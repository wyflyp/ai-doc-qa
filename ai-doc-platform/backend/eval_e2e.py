"""端到端评估：测的是「答案对不对」，不是「片段找没找到」。

为什么必须有这一层？
    Hit@K 只能告诉你检索质量。但用户关心的是最终答案。
    这两件事经常不一致 —— 我实测就遇到过：
    问「v-show 和 v-if 有什么区别」，`v-show` 这个词确实在文档里（Hit@K 算命中），
    但它只出现在「实验要求：v-show 和 v-if 有什么区别？」这个**问题列表**里，
    文档根本没给答案。检索"成功"，答案只能是「未提及」。

    所以端到端评估要回答的是：**瓶颈到底在检索还是在生成？**

核心产出：2×2 归因矩阵
                    答案正确        答案错误
    检索命中        正常            生成/上下文问题 ← 最值得查
    检索未命中      模型靠先验蒙对    检索问题

    这个矩阵比一个笼统的「准确率 80%」有用得多 —— 它告诉你下一步该改哪里。

判分方式（两层，都要看）：
    1. 规则判分：答案里有没有事实要点关键词（客观、可复现，但粗糙）
    2. LLM 裁判：让模型判断答案是否覆盖要点、有没有编造（更接近人判断，但有噪声）
    两者不一致的题会被单独标出来，建议人工复核 —— 这就是「人工评 20 个问题」的
    可落地版本：机器先过一遍，人只复核有争议的。

同时测防幻觉：
    题目里刻意放了 2 条**文档里根本没有答案**的题。理想行为是明确说「文档中未提及」。
    如果模型硬编一个答案，就记为 HALF（编造），这比答错更严重。

用法：
    python eval_e2e.py            # 全量 22 题
    python eval_e2e.py --limit 5  # 只跑前 5 题（调试用）
    python eval_e2e.py --no-judge # 只跑规则判分，省 LLM 调用

前提：后端服务得先启动（默认 127.0.0.1:8000）。
"""

import csv
import json
import os
import sys
import time
from pathlib import Path

# 独立运行时也要读到 .env（LLM_API_KEY 在这里）
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except Exception:
    pass

import requests

BASE = os.getenv("EVAL_BASE_URL", "http://127.0.0.1:8000")
TOP_K = 5

# trust_env=False：不读系统代理。
# 本机开着代理时，requests 会把 127.0.0.1 也走代理，导致每次请求多等好几秒
# （实测 4 题跑了 105 秒，关掉代理环境变量后快很多）。
_session = requests.Session()
_session.trust_env = False

# (类型, 问题, 事实要点关键词, 参考要点, 是否文档中无答案)
CASES = [
    # ---- FACT：事实查找 ----
    ("fact", "密码输入框怎么控制提交按钮能不能点",
     ["disabled"], "应说明用一个响应式变量控制按钮的 disabled 属性（v-bind 绑定即可，不要求一定提 ref）", False),
    ("fact", "复制多行内容用哪组按键",
     ["nyy"], "文档写明命令模式下 nyy 用于复制多行内容（示例为复制 4 行到文件尾部）", False),
    ("fact", "删除最后两行用什么命令",
     ["ndd"], "文档写明命令模式下 ndd 删除多行内容（示例为删除最后 2 行）", False),
    ("fact", "创建目录用什么命令",
     ["mkdir"], "Linux 中用 mkdir 创建目录", False),
    ("fact", "查看当前所在路径",
     ["pwd"], "Linux 中用 pwd 显示当前工作目录", False),
    ("fact", "统计字母出现次数用的字典叫什么",
     ["count_letter"], "用 count_letter 字典统计每个字母出现次数，先初始化为 0 再遍历累加", False),
    ("fact", "怎么找出最长的单词",
     ["longest"], "遍历单词列表，用 len 比较长度，记录最长的那个", False),
    ("fact", "判断一个正整数是不是丑数",
     ["is_ugly"], "丑数是只含 2、3、5 因子的正整数，1 算特殊丑数；is_ugly 返回 True/False", False),

    # ---- SYN：同义 / 缩写 ----
    ("syn", "住酒店一晚上最多能报多少",
     ["八百", "六百"],
     "住宿上限：制度汇编为一类城市八百元，员工手册为一线城市六百元（两份文档确实冲突）。"
     "给出任一口径、或明确指出冲突，都算正确", False),
    ("syn", "公积金按什么比例缴",
     ["百分之十二"], "住房公积金缴存比例为百分之十二", False),
    ("syn", "主动辞职要提前多久",
     ["三十日"], "员工主动离职须提前三十日提交书面申请", False),
    ("syn", "SLA 要求全年可用性达到多少",
     ["百分之九十九点九五"], "核心业务系统可用性不低于百分之九十九点九五", False),
    ("syn", "OA 系统什么时候做例行维护",
     ["凌晨两点"], "每日凌晨两点至四点例行维护", False),
    ("syn", "E4015 是什么故障",
     ["E4015", "重启通讯模组"], "现象为设备无法启动、指示灯闪烁 5 次后常亮；处理方式是重启通讯模组", False),
    ("syn", "苹果一斤多少钱",
     ["3.5"], "bom.csv 中苹果的 price 为 3.5", False),

    # ---- CROSS：跨文档 ----
    ("cross", "Vue 的表单双向绑定和 Linux 的创建目录分别怎么做",
     ["v-model", "mkdir"], "应同时给出 v-model（Vue）和 mkdir（Linux）两个答案", False),
    ("cross", "住宿报销上限和数据库备份保留期限分别是多少",
     ["八百", "九十天"],
     "住宿上限：制度汇编为八百元、员工手册为六百元（两份文档确实冲突）；"
     "只要能给出其中一种口径、或明确指出两份文档标准不一致，都算正确。备份保留九十天", False),
    ("cross", "Python 怎么找最长单词，Linux 怎么查看当前路径",
     ["longest", "pwd"],
     "Python 部分：用 max(words, key=len) 找最长单词。"
     "Linux 部分：文档只把 pwd 列为「需掌握的目录命令」，并未说明它的用途 —— "
     "答出 pwd、或明确指出文档未说明其用途，都算正确", False),

    # ---- 无答案：测防幻觉 ----
    ("none", "公司今年的营业收入是多少",
     [], "文档中没有任何营收数据，正确行为是明确说未提及", True),
    ("none", "实验里用的深度学习框架是哪个",
     [], "文档里没有深度学习框架相关内容，正确行为是说未提及", True),
]


# ---------------- 调用后端 ----------------
def ask(question: str) -> dict:
    """调 /api/chat/stream，收集完整答案和引用片段。"""
    resp = _session.post(
        f"{BASE}/api/chat/stream",
        json={"question": question, "top_k": TOP_K, "use_rag": True},
        stream=True,
        timeout=120,
    )
    resp.raise_for_status()

    answer, sources, meta = [], [], {}
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data:"):
            continue
        payload = raw[5:].strip()
        if payload == "[DONE]":
            break
        try:
            evt = json.loads(payload)
        except Exception:
            continue
        t = evt.get("type")
        if t == "delta":
            answer.append(evt.get("content", ""))
        elif t == "meta":
            sources = evt.get("sources", [])
            meta = evt
        elif t == "error":
            answer.append(f"[错误] {evt.get('message')}")

    return {"answer": "".join(answer).strip(), "sources": sources, "meta": meta}


def retrieve(question: str) -> list:
    """调 /api/rag/debug 拿完整片段内容。

    为什么不直接用 meta 里的 sources？
        那里的 preview 只有前 120 字。判断「检索有没有命中」必须看全文，
        否则关键词在片段后半段就会漏判 —— 我第一版就踩了这个坑，
        明明检索命中了却判成未命中，把归因矩阵算错了。
    """
    try:
        resp = _session.post(
            f"{BASE}/api/rag/debug",
            json={"question": question, "top_k": TOP_K},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("results", [])
    except Exception:
        return []


# ---------------- 拒答检测（规则，优先于裁判）----------------
# 为什么要用规则兜底？
#     LLM 裁判不稳定：实测有一条「文档中未提及公司今年的营业收入」，
#     裁判给出的理由是「正确指出文档未提及」，但 verdict 却返回了 pass，
#     导致被误判成"编造"。拒答是有明确文本特征的，用规则判断比问模型可靠。
REFUSAL_MARKERS = [
    "未提及", "没有提及", "未涉及", "没有涉及", "未包含", "没有包含",
    "文档中没有", "文档中无", "无法从文档", "文档未提供", "没有相关",
    "未找到相关", "不包含相关",
]


def is_refusal(answer: str) -> bool:
    """答案是否**整体上**在说「文档里没有」。

    这里有个坑，第一版判错了 4 条：
        「文档中未提及公司今年的营业收入。」            → 真拒答 ✅
        「苹果一斤 3.5。…但片段未明确说明 price 的单位」 → 部分作答，不是拒答
        「文档只涉及 Vue…Linux 创建目录的做法未提及」    → 部分作答，不是拒答
    四个都含「未提及」，但后两个明明给了答案。所以不能只看关键词，要结合：
        1. 是否**开场就拒答**（前 60 字出现）
        2. 篇幅是否够短（真拒答通常很短，给了内容的答案会展开）

    规则：开场 60 字内出现拒答词，且总长不超过 200 字 → 判定为拒答。
    另外前 20 字就出现拒答词的，无论多长都算（这是非常明确的拒答开场）。
    """
    if not answer:
        return False
    head60, head20 = answer[:60], answer[:20]
    hit20 = any(m in head20 for m in REFUSAL_MARKERS)
    if hit20:
        return True
    return len(answer) <= 200 and any(m in head60 for m in REFUSAL_MARKERS)


def retrieval_hit_of(kind: str, kws: list, chunks: list) -> bool:
    """判断检索是否「够用」。

    ⚠️ 跨文档题必须用 all，不能用 any：
        问「Vue 的表单绑定和 Linux 的创建目录分别怎么做」，
        top-5 里全是 Vue 文档、一个 mkdir 都没有，
        但只要有一个关键词（v-model）命中，用 any 就会判成「检索命中」，
        于是把「检索失败」错记成「生成失败」，归因完全反了。
    """
    if not kws:
        return False
    joined = "\n".join(c.get("content", "") for c in chunks)
    if kind == "cross":
        return all(k in joined for k in kws)
    return any(k in joined for k in kws)


# ---------------- LLM 裁判 ----------------
JUDGE_PROMPT = """你是严格的答案质量评估员。请判断系统回答是否正确。

判定标准（只能选一个）：
- pass     完全正确：覆盖了参考要点，且没有编造与要点矛盾的内容
- partial  部分正确：提到了部分要点，但缺少关键信息
- wrong    错误：与参考要点矛盾，或答非所问
- refused  明确表示文档中未提及（这类问题本身可能就没有答案，要看是否该拒答）

注意：
- 不要因为答案简短就判 partial，只要要点齐了就是 pass
- 答案里出现参考要点之外的具体数字、人名、命令时，要警惕编造

输出严格的 JSON，不要任何解释文字：
{"verdict":"pass|partial|wrong|refused","reason":"不超过 20 字的理由"}"""


def judge(question: str, rubric: str, answer: str, no_answer: bool) -> dict:
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("LLM_API_KEY", ""),
        base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
    )
    user = (
        f"问题：{question}\n"
        f"参考要点：{rubric}\n"
        f"文档中是否有答案：{'没有，正确行为是拒答' if no_answer else '有'}\n"
        f"系统回答：{answer[:1500]}"
    )
    try:
        resp = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "deepseek-chat"),
            messages=[{"role": "system", "content": JUDGE_PROMPT},
                      {"role": "user", "content": user}],
            temperature=0,
            max_tokens=120,
            timeout=60,
        )
        text = (resp.choices[0].message.content or "").strip()
        s, e = text.find("{"), text.rfind("}")
        data = json.loads(text[s:e + 1]) if s >= 0 else {}
        verdict = str(data.get("verdict", "")).lower()
        if verdict not in ("pass", "partial", "wrong", "refused"):
            verdict = "unknown"
        return {"verdict": verdict, "reason": str(data.get("reason", ""))[:40]}
    except Exception as exc:
        return {"verdict": "unknown", "reason": f"裁判失败 {type(exc).__name__}"}


# ---------------- 主流程 ----------------
def normalize(verdict: str, no_answer: bool, answer: str = "", kws: list = None) -> str:
    """把裁判结论归一成「答案对不对」。

    拒答优先用规则判断（LLM 裁判在这件事上不稳定，见 REFUSAL_MARKERS），
    但有一条硬约束：**只要答案里给出了预期要点，就绝不判成拒答**。

    为什么加这条？
        实测「统计字母出现次数用的字典叫什么」，答案写的是
        「文档中未提及该字典的专门名称，只给出了变量名 count_letter」——
        这句话开头带「未提及」，被规则判成拒答，可它明明答出了 count_letter。
        要点命中就该算作答。

    无答案题的规则和普通题相反：
        拒答 = 正确（模型没有编造）
        给了具体答案 = 编造，记为 HALLUCINATED
    """
    has_fact = bool(kws) and any(k in answer for k in (kws or []))
    refused = is_refusal(answer) and not has_fact

    if no_answer:
        if refused:
            return "CORRECT"
        if verdict in ("pass", "partial", "wrong"):
            return "HALLUCINATED"
        return "UNKNOWN"

    if refused:
        # 文档里明明有答案却拒答 —— 说明上下文没送到模型手里
        return "REFUSED"
    if verdict == "pass":
        return "CORRECT"
    if verdict == "partial":
        return "PARTIAL"
    if verdict == "wrong":
        return "WRONG"
    return "UNKNOWN"


def main():
    limit = None
    use_judge = "--no-judge" not in sys.argv
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    cases = CASES[:limit] if limit else CASES

    # 探活
    try:
        _session.get(f"{BASE}/api/rag/stats", timeout=5).raise_for_status()
    except Exception:
        print(f"连不上后端 {BASE}，请先启动：双击 1-启动后端.bat")
        return 1

    print(f"端到端评估：{len(cases)} 题（规则判分{' + LLM 裁判' if use_judge else ''}）\n")

    rows = []
    started = time.time()
    for i, (kind, q, kws, rubric, no_answer) in enumerate(cases, 1):
        t0 = time.time()
        try:
            out = ask(q)
        except Exception as exc:
            print(f"{i:>2}. {q[:26]:28} 调用失败 {type(exc).__name__}")
            rows.append({"kind": kind, "question": q, "answer": "", "error": str(exc),
                         "retrieval_hit": False, "rule_hit": False,
                         "verdict": "unknown", "result": "UNKNOWN", "reason": "调用失败"})
            continue

        answer = out["answer"]
        # 用完整片段内容判断检索命中（不能用 meta 里的 120 字 preview）
        hits = retrieve(q)
        retrieval_hit = retrieval_hit_of(kind, kws, hits)
        rule_hit = any(k in answer for k in kws) if kws else False

        if use_judge:
            jr = judge(q, rubric, answer, no_answer)
        else:
            jr = {"verdict": "pass" if rule_hit else "wrong", "reason": "仅规则判分"}

        result = normalize(jr["verdict"], no_answer, answer, kws)
        ms = int((time.time() - t0) * 1000)

        rows.append({
            "kind": kind, "question": q, "answer": answer,
            "retrieval_hit": retrieval_hit, "rule_hit": rule_hit,
            "verdict": jr["verdict"], "result": result, "reason": jr["reason"],
            "sources": len(out["sources"]), "ms": ms,
        })

        flag = {"CORRECT": "✅", "PARTIAL": "🟡", "WRONG": "❌",
                "REFUSED": "🚫", "HALLUCINATED": "⚠️", "UNKNOWN": "?"}[result]
        print(f"{i:>2}. {flag} [{kind:5}] {q[:26]:28} 检索{'命中' if retrieval_hit else '未中'} "
              f"→ {result:12} {jr['reason'][:24]}")

    # ---------------- 汇总 ----------------
    report(rows, elapsed=int(time.time() - started))
    return 0


def report(rows: list, elapsed: int = 0, use_judge: bool = True):
    """打印汇总并落盘。被主流程和 --rejudge 共用。"""
    total = len(rows)
    correct = sum(1 for r in rows if r["result"] == "CORRECT")
    print(f"\n{'='*62}")
    print(f"答案正确率：{correct}/{total}  ({correct/total*100:.0f}%)")
    if elapsed:
        print(f"耗时：{elapsed}s")

    # 2×2 归因矩阵（无答案题不参与，因为"检索命中"对它们没意义）
    graded = [r for r in rows if r["kind"] != "none"]
    print(f"\n{'='*62}")
    print("归因矩阵（有答案的题）")
    print(f"{'':14}{'答案正确':>10}{'答案错误':>10}")
    for rh in (True, False):
        ok = sum(1 for r in graded if r["retrieval_hit"] == rh and r["result"] == "CORRECT")
        bad = sum(1 for r in graded if r["retrieval_hit"] == rh and r["result"] != "CORRECT")
        label = "检索命中" if rh else "检索未命中"
        print(f"{label:14}{ok:>10}{bad:>10}")

    hit_bad = [r for r in graded if r["retrieval_hit"] and r["result"] != "CORRECT"]
    if hit_bad:
        print(f"\n⚠️  检索命中但答案不对（{len(hit_bad)} 条）—— 瓶颈在生成，不在检索：")
        for r in hit_bad:
            print(f"    {r['question'][:30]}  [{r['result']}] {r['reason'][:30]}")

    miss_good = [r for r in graded if not r["retrieval_hit"] and r["result"] == "CORRECT"]
    if miss_good:
        print(f"\n⚠️  检索没命中但答案对了（{len(miss_good)} 条）—— 模型靠先验知识蒙的，不可靠：")
        for r in miss_good:
            print(f"    {r['question'][:30]}")

    # 防幻觉
    none_rows = [r for r in rows if r["kind"] == "none"]
    if none_rows:
        refused = sum(1 for r in none_rows if r["result"] == "CORRECT")
        hallu = sum(1 for r in none_rows if r["result"] == "HALLUCINATED")
        print(f"\n防幻觉（{len(none_rows)} 条无答案题）：老实拒答 {refused}，编造 {hallu}")

    # 规则判分与裁判不一致的，建议人工复核
    if use_judge:
        diff = [r for r in rows
                if r["kind"] != "none"
                and str(r["rule_hit"]) != str(r["result"] == "CORRECT")]
        if diff:
            print(f"\n需要人工复核（规则判分与裁判结论不一致）{len(diff)} 条：")
            for r in diff:
                print(f"    {r['question'][:28]:30} "
                      f"规则{'命中' if str(r['rule_hit']) == 'True' else '未中'} "
                      f"裁判{r['result']}")

    # 存 CSV 供人工复核
    out_path = Path(__file__).resolve().parent / "e2e_report.csv"
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=["kind", "question", "retrieval_hit", "rule_hit",
                           "verdict", "result", "reason", "answer", "ms"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n明细已保存：{out_path.name}（可用 Excel 打开人工复核）")


def rejudge():
    """只重判，不重新生成答案。

    为什么需要这个？
        调过判分规则或参考要点之后，如果重新跑一遍生成，
        20 题要 8 分多钟，而且答案本身会变，反而不好对比。
        答案已经存在 CSV 里了，直接复用，只重跑裁判 —— 40 秒搞定。
    """
    path = Path(__file__).resolve().parent / "e2e_report.csv"
    if not path.exists():
        print(f"没有找到 {path.name}，先跑一次完整评估")
        return 1

    by_q = {c[1]: c for c in CASES}
    rows = []
    print("重新判分（复用已有答案）\n")

    for r in csv.DictReader(path.open(encoding="utf-8-sig")):
        case = by_q.get(r["question"])
        if not case:
            rows.append(r)
            continue
        _, q, kws, rubric, no_answer = case
        answer = r.get("answer", "")
        jr = judge(q, rubric, answer, no_answer)
        result = normalize(jr["verdict"], no_answer, answer, kws)
        # 检索命中要重算：修正跨文档判定后，旧 CSV 里的值是错的
        rh = retrieval_hit_of(case[0], kws, retrieve(q)) if kws else False
        r.update({
            "verdict": jr["verdict"], "result": result, "reason": jr["reason"],
            "rule_hit": str(any(k in answer for k in kws)) if kws else "False",
            # CSV 读回来是字符串，report() 里要按布尔比较，这里统一转回 bool
            "retrieval_hit": rh,
        })
        flag = {"CORRECT": "✅", "PARTIAL": "🟡", "WRONG": "❌",
                "REFUSED": "🚫", "HALLUCINATED": "⚠️", "UNKNOWN": "?"}[result]
        print(f"  {flag} {q[:28]:30} → {result:12} {jr['reason'][:26]}")
        rows.append(r)

    report(rows)
    return 0


if __name__ == "__main__":
    if "--rejudge" in sys.argv:
        sys.exit(rejudge())
    sys.exit(main())