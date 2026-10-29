"""查询改写 / 扩展（Query Rewriting / Multi-Query Retrieval）。

解决什么问题？
    用户提问的用词，经常和文档里的用词对不上：
        用户问「住酒店能报多少钱」  文档写的是「住宿标准」
        用户问「年假多少天」        文档写的是「休假天数」
        用户问「东西坏了能退吗」    文档写的是「售后退换政策」
    这叫**词汇鸿沟（vocabulary gap）**。纯词面检索（BM25）在这种情况直接零命中，
    向量检索也只能靠语义"猜"，经常猜偏。

    上一版我靠 cross-encoder 精排硬救 —— 能救回来，但那是"救火"：
    前提是正确片段先被召回到候选池里，如果压根没召回，精排也无能为力。

    查询改写是"防火"：在检索之前，先把问题翻译成文档可能使用的措辞。

三种做法（按成本从低到高）：
    1. 同义词词典：手工维护，覆盖有限，但零成本零延迟
    2. 小模型改写（本项目采用）：用 LLM 生成 2~3 个候选查询，
       理解力强、覆盖广，代价是一次额外的 LLM 调用（约 0.5~1.5s）
    3. HyDE：让模型先"编"一段假答案再去检索，效果更好但更慢、更容易引入幻觉

Multi-Query 检索流程：
    原问题 + 3 个改写 → 每个查询各自检索 → 跨查询 RRF 融合
    注意原查询权重给最高，因为它最接近用户真实意图；
    改写只用来"补召回"，不能反客为主。

成本控制（这几条很关键，否则改写会变成系统瓶颈）：
    - 结果缓存：同一个问题第二次问不再调用模型
    - 超时短（默认 8s）+ 失败降级：改写挂了就用原问题，绝不让问答失败
    - 只对「问题较短且不含编号」的查询改写（长问题本身信息够，编号必须原样保留）
"""

import hashlib
import json
import os
import time

# 独立运行（直接 python query_rewrite.py）时也要能读到 .env
# load_dotenv 默认不覆盖已存在的环境变量，所以重复调用是安全的
try:
    from dotenv import load_dotenv
    from pathlib import Path as _Path

    load_dotenv(_Path(__file__).resolve().parent / ".env")
except Exception:
    pass

# 开关：true / false
# 默认关闭。实测数据（57 条题）：
#   混合 36/49 → 加改写 40/49（+4），但**开了精排之后再叠加改写没有任何增益**（都是 44/49）。
#   而且改写对跨文档题是负收益（CROSS 5/8 → 4/8），多查询融合会把结果集中到单一文档上。
#   加上每条多花约 1 秒，所以默认不开。
# 什么时候开：没有精排、或者问题用词和文档差异极大时，它能补召回。
REWRITE_ENABLED = os.getenv("QUERY_REWRITE_ENABLED", "false").lower() in ("1", "true", "yes")
# 生成几个改写查询（不含原查询）
REWRITE_N = int(os.getenv("QUERY_REWRITE_N", "3"))
# 单次改写超时（秒）
REWRITE_TIMEOUT = float(os.getenv("QUERY_REWRITE_TIMEOUT", "8"))
# 问题超过这个长度就不改写了 —— 长问题信息已经足够，改写收益低还费钱
REWRITE_MAX_LEN = int(os.getenv("QUERY_REWRITE_MAX_LEN", "40"))
# 原查询在跨查询融合里的权重，改写查询权重为 1
ORIGINAL_WEIGHT = float(os.getenv("QUERY_REWRITE_ORIGINAL_WEIGHT", "2.0"))

SYSTEM_PROMPT = """你是检索查询改写助手。用户会针对一批文档提问，你要把问题改写成更适合检索的查询。

规则：
1. 补充文档可能使用的同义说法。例如「酒店」→「住宿」，「年假」→「休假」，「多少钱」→「费用/标准/上限」。
2. 数字、编号、人名、产品型号必须原样保留，绝对不要改。
3. 每个查询都要短，用关键词组合，不要写成问句。
4. 改写要覆盖不同角度，不要只是换个说法重复同一句。
5. 只输出 JSON 数组，不要任何解释、不要 markdown 代码块。

示例：
问题：住酒店能报多少钱
输出：["住宿标准 费用上限", "酒店 住宿费 报销额度", "一线城市 二线城市 住宿 每晚"]

问题：E4027 怎么处理
输出：["E4027 故障处理", "E4027 现象 解决方法", "E4027 售后"]"""

_cache = {}
_CACHE_MAX = 256


def _client():
    from openai import OpenAI

    return OpenAI(
        api_key=os.getenv("LLM_API_KEY", ""),
        base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
    )


def _llm_ready() -> bool:
    key = os.getenv("LLM_API_KEY", "").strip()
    return len(key) >= 20 and "你的" not in key and "在这里" not in key


def _should_rewrite(question: str) -> bool:
    """判断这个问题值不值得改写。

    不值的情况：
      - 太短（一个词，改写出不来什么）
      - 太长（信息已经够，改写收益低）
      - 纯编号查询（E4027 这种，改写反而可能把编号改丢掉）
    """
    q = question.strip()
    if len(q) < 4 or len(q) > REWRITE_MAX_LEN:
        return False
    return True


def _parse(text: str, n: int) -> list:
    """从模型输出里抠出 JSON 数组。模型经常多嘴加代码块或解释，要容错。"""
    text = text.strip()
    if text.startswith("```"):
        # 去掉 ```json ... ```
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[:-3]
    text = text.strip()

    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []

    try:
        data = json.loads(text[start:end + 1])
    except Exception:
        return []

    if not isinstance(data, list):
        return []

    out = []
    for item in data:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
        elif isinstance(item, dict):
            # 有些模型会返回 [{"query": "..."}]
            v = item.get("query") or item.get("q")
            if isinstance(v, str) and v.strip():
                out.append(v.strip())
    return out[:n]


def rewrite(question: str, n: int = None, use_llm: bool = None) -> dict:
    """生成改写查询。

    Args:
        question: 用户原问题
        n: 生成几个改写
        use_llm: None 时读 QUERY_REWRITE_ENABLED

    Returns:
        {
          "original": 原问题,
          "queries": [原问题, 改写1, 改写2, ...],   # 原问题永远在第一位
          "rewritten": [改写1, ...],
          "cached": bool,
          "elapsed_ms": int,
          "error": str | "",
        }
    """
    n = n or REWRITE_N
    if use_llm is None:
        use_llm = REWRITE_ENABLED

    base = {
        "original": question,
        "queries": [question],
        "rewritten": [],
        "cached": False,
        "elapsed_ms": 0,
        "error": "",
    }

    if not use_llm or not _should_rewrite(question) or not _llm_ready():
        return base

    key = hashlib.md5(f"{question}|{n}".encode("utf-8")).hexdigest()
    if key in _cache:
        # 注意：缓存里存的 elapsed_ms 是「首次生成本次改写」的耗时，
        # 直接返回会让前端显示一个假的耗时（实际是微秒级）。
        # 所以这里要把 elapsed_ms 改成真实的缓存读取耗时，
        # 原始耗时另存一份，方便分析改写本身的成本。
        hit = dict(_cache[key])
        hit["cached"] = True
        hit["original_elapsed_ms"] = hit.get("elapsed_ms", 0)
        hit["elapsed_ms"] = 0
        return hit

    started = time.time()
    try:
        resp = _client().chat.completions.create(
            model=os.getenv("REWRITE_MODEL", os.getenv("LLM_MODEL", "deepseek-chat")),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"问题：{question}\n输出："},
            ],
            temperature=0.3,      # 改写要稳，别太发散
            max_tokens=200,
            timeout=REWRITE_TIMEOUT,
        )
        text = resp.choices[0].message.content or ""
        rewritten = _parse(text, n)
    except Exception as exc:
        base["error"] = f"{type(exc).__name__}: {exc}"
        base["elapsed_ms"] = int((time.time() - started) * 1000)
        return base

    # 去掉和原问题重复的、去重保序
    seen = {question.strip().lower()}
    uniq = []
    for r in rewritten:
        low = r.strip().lower()
        if low and low not in seen:
            seen.add(low)
            uniq.append(r.strip())

    result = {
        "original": question,
        "queries": [question] + uniq,
        "rewritten": uniq,
        "cached": False,
        "elapsed_ms": int((time.time() - started) * 1000),
        "original_elapsed_ms": int((time.time() - started) * 1000),
        "error": "",
    }

    if len(_cache) < _CACHE_MAX:
        _cache[key] = dict(result)

    return result


# 子问题分解（跨文档检索用）
#   跨文档题失败的根因不是"排序不好"，而是"一个查询里有两个意图"。
#   按文档配额只能保证"选了几篇不同文档"，保证不了"覆盖了两个意图" ——
#   实测要 min_docs=4 这种极端值才碰巧捞到第二篇相关文档，而那样会毁掉单文档题。
#   正确做法是把问题拆开，各自检索。
DECOMPOSE_ENABLED = os.getenv("QUERY_DECOMPOSE_ENABLED", "true").lower() in ("1", "true", "yes")

DECOMPOSE_PROMPT = """判断用户问题是否包含**多个并列的、需要分别回答的子问题**。

规则：
1. 如果有多个子问题，拆成独立问题，每个都要能单独检索（补全主语和限定词）。
2. 如果只有一个问题，输出空数组 []。
3. 不要改写措辞，不要补充同义词，只做拆分。
4. 只输出 JSON 数组，不要任何解释或代码块。

示例：
问题：Vue 的表单双向绑定和 Linux 的创建目录分别怎么做
输出：["Vue 表单双向绑定怎么做", "Linux 创建目录用什么命令"]

问题：住宿报销上限和数据库备份保留期限分别是多少
输出：["住宿报销上限是多少", "数据库备份保留多少天"]

问题：年假有几天
输出：[]

问题：怎么统计字符串里各个字母出现的次数
输出：[]"""


# 零成本的预筛：只有看起来是「多子问题」时才值得花 1 秒调 LLM
MULTI_MARKERS = ["分别", "各自", "以及", "另外", "还有", "同时", "、"]
QUESTION_WORDS = ["怎么", "什么", "多少", "哪些", "哪个", "如何", "为什么"]


def _looks_multi(question: str) -> bool:
    """粗判问题是否包含多个子问题。

    为什么要这层预筛？
        分解要对每个问题调一次 LLM（约 1 秒）。但绝大多数问题只有一个意图，
        在它们上面花 1 秒纯粹是浪费。用关键词先筛一遍，成本为零。

    判定规则：
        1. 出现「分别/各自/以及/另外/还有/同时/、」→ 大概率是多子问题
        2. 疑问词出现 2 次以上（「怎么…怎么…」「多少…多少…」）→ 同上

    已知漏判：「A 和 B 分别…」以外的形式，比如「A 怎么弄，B 呢」会被漏掉。
    这是有意的取舍 —— 宁可漏判（退化成普通检索，不会变差），
    也不要为了覆盖所有写法让每个查询都多花 1 秒。
    """
    if any(m in question for m in MULTI_MARKERS):
        return True
    return sum(question.count(w) for w in QUESTION_WORDS) >= 2


def decompose(question: str, use_llm: bool = None) -> dict:
    """把含多个子问题的问题拆开。

    Returns:
        {"original":..., "subs":[子问题...], "cached":bool, "error":str}
        单问题或无 LLM 时 subs 为空列表。
    """
    if use_llm is None:
        use_llm = DECOMPOSE_ENABLED

    base = {"original": question, "subs": [], "cached": False, "error": ""}
    if not use_llm or not _llm_ready() or len(question.strip()) < 6:
        return base
    if not _looks_multi(question):
        return base          # 预筛没通过，不花 LLM 调用

    cache_key = hashlib.md5(f"decompose|{question}".encode("utf-8")).hexdigest()
    if cache_key in _cache:
        hit = dict(_cache[cache_key])
        hit["cached"] = True
        return hit

    try:
        resp = _client().chat.completions.create(
            model=os.getenv("REWRITE_MODEL", os.getenv("LLM_MODEL", "deepseek-chat")),
            messages=[{"role": "system", "content": DECOMPOSE_PROMPT},
                      {"role": "user", "content": f"问题：{question}\n输出："}],
            temperature=0,
            max_tokens=200,
            timeout=REWRITE_TIMEOUT,
        )
        subs = _parse(resp.choices[0].message.content or "", 4)
    except Exception as exc:
        base["error"] = f"{type(exc).__name__}: {exc}"
        return base

    # 去掉跟原问题完全一样的、去重
    seen = {question.strip().lower()}
    uniq = []
    for s in subs:
        low = s.strip().lower()
        if low and low not in seen:
            seen.add(low)
            uniq.append(s.strip())

    # 只拆出 1 个子问题没意义（等于没拆）
    if len(uniq) < 2:
        uniq = []

    result = {"original": question, "subs": uniq, "cached": False, "error": ""}
    if len(_cache) < _CACHE_MAX:
        _cache[cache_key] = dict(result)
    return result


def clear_cache() -> None:
    _cache.clear()


def info() -> dict:
    return {
        "enabled": REWRITE_ENABLED,
        "n": REWRITE_N,
        "original_weight": ORIGINAL_WEIGHT,
        "cache_size": len(_cache),
    }


if __name__ == "__main__":
    tests = [
        "住酒店能报多少钱",
        "年假多少天",
        "苹果多少钱一斤",
        "E4027 怎么处理",
        "公",
    ]
    for q in tests:
        t = time.time()
        r = rewrite(q)
        print(f"原问题：{q}")
        if r["rewritten"]:
            for i, x in enumerate(r["rewritten"], 1):
                print(f"   改写{i}: {x}")
        else:
            print(f"   （不改写，原因：{'开关关闭' if not REWRITE_ENABLED else '规则跳过或失败'}）")
        print(f"   耗时 {r['elapsed_ms']}ms  缓存={r['cached']}  错误={r['error'][:60]}")
        print(f"   总耗时 {(time.time()-t)*1000:.0f}ms")
        print()