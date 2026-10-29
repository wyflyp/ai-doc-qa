"""重排（Rerank）—— 检索链路最后一道裁判。

⚠️ 重要结论（实测，两次结论相反 —— 这才是重点）：
    同一套代码、同一个语料，只是评估集从 16 条扩到 57 条，结论完全翻转：
        16 条题：混合 15/16  →  加精排 14/16   （负收益）
        57 条题：混合 36/49  →  加精排 44/49   （+8，显著正收益）
    按分类看（57 条）：FACT 19→24，SYN 17→20，CROSS 5→5（跨文档无改善）。

    所以现在**默认开启**。但要记住：16 条题时它看起来是有害的。
    **小样本评估得出的结论不可信**，这条比精排本身更值得记住。

    另一个发现：开精排之后 RRF 权重变得不敏感
    —— 1:1 和 1:3 的结果完全相同（都是 44/49），
    因为精排会重新打分，直接覆盖融合顺序。

为什么混合检索之后还要 rerank？
    RRF 是「无监督」融合：它只看两路的名次，不知道谁对。
    实测过的翻车现场：问「E4027 怎么处理」时，向量把 E4013 排第一，
    BM25 把 E4027 排第一，RRF 只能折中 —— 它没有能力判断
    「E4027 和查询的相关性比 E4013 高」，因为它从没读过原始文本。

    Cross-encoder 不一样：它把 (query, 文档) **拼在一起**送进模型，
    让模型直接判断这一对的匹配程度。计算量比 bi-encoder 大得多
    （每个候选都要跑一次完整推理），所以只能用在最后一步：
    先粗排召回 20 个，再精排选 4 个。

架构对比（面试重点）：
    bi-encoder（我们检索用的 bge-small-zh）
        问题 → 向量 ┐
        文档 → 向量 ┴→ 算余弦
        优点：文档向量可以**离线预计算**，查询只算一次，快
        缺点：query 和 doc 在编码阶段互相看不见，精度有上限

    cross-encoder（rerank 用的 bge-reranker）
        (问题, 文档) → 模型 → 一个分数
        优点：query 和 doc 在注意力层里充分交互，精度高得多
        缺点：无法预计算，每个候选都要推理一次，慢
        所以只能做 rerank，不能做检索

降级设计（很重要）：
    rerank 是「锦上添花」，不是必需品。模型没下载、内存不够、
    推理报错，都必须能退回 RRF 原始顺序继续工作，
    而不是让用户看到 500。
"""

import os

# none / local / api
RERANK_PROVIDER = os.getenv("RERANK_PROVIDER", "auto").strip().lower()
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
# 粗排召回多少个候选去精排。太少没得选，太多太慢
# 提速主要靠下面的 RERANK_MAX_CHARS 截断（已验证不影响准确率），
# 候选数这里保守保留 20，避免改动带来的精度波动
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "20"))
# 精排后留下多少个
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "4"))
# 分数下限（bge-reranker 输出 logits，>0 表示相关）
RERANK_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", "-100"))

MODEL_CACHE = os.getenv(
    "EMBEDDING_CACHE_DIR",
    str(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".model_cache")),
)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

_model = None
_load_failed = False


def available() -> bool:
    """模型是否真的能用。检查一次就记住结果，避免每次请求都重试加载。"""
    global _load_failed
    if RERANK_PROVIDER == "none":
        return False
    if _load_failed:
        return False
    if RERANK_PROVIDER == "api":
        return bool(os.getenv("RERANK_API_KEY") or os.getenv("LLM_API_KEY"))
    try:
        import fastembed.rerank.cross_encoder  # noqa: F401
        return True
    except ImportError:
        return False


def _get_model():
    global _model, _load_failed
    if _model is not None:
        return _model
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        os.makedirs(MODEL_CACHE, exist_ok=True)
        _model = TextCrossEncoder(model_name=RERANK_MODEL, cache_dir=MODEL_CACHE)
        return _model
    except Exception as exc:
        # 加载失败就永久降级，不要每次请求都卡几秒重试
        print(f"[rerank] 模型加载失败（{exc}），本次进程内停用 rerank")
        _load_failed = True
        return None


def _score_local(query: str, docs: list) -> list:
    model = _get_model()
    if model is None:
        return []
    return [float(s) for s in model.rerank(query, docs)]


def _score_api(query: str, docs: list) -> list:
    """OpenAI 兼容的 rerank 接口（如硅基流动 /v1/rerank）。"""
    import requests

    base = os.getenv("RERANK_BASE_URL", "https://api.siliconflow.cn/v1")
    key = os.getenv("RERANK_API_KEY", os.getenv("LLM_API_KEY", ""))
    resp = requests.post(
        f"{base.rstrip('/')}/rerank",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": RERANK_MODEL, "query": query,
              "documents": docs, "top_n": len(docs)},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()["results"]
    # 接口返回的是 {index, relevance_score} 且已排序，这里还原成按原顺序的分数
    scores = [0.0] * len(docs)
    for item in data:
        scores[item["index"]] = float(item["relevance_score"])
    return scores


# 送给 cross-encoder 的文本最大长度。
# 这是个很重要的性能参数：实测 20 条 469 字的片段要 8.6 秒，
# 截断到 200 字只要 4.0 秒 —— cross-encoder 的耗时随文本长度增长得比线性更快。
#
# 但截断不能太狠：实测截到 300 字时，故障代码 E4015 那个片段因为编号在
# 片段后半段被切掉，精排给了低分、掉出 top-5，检索直接失败。
# 400 字能保住它，耗时代价只有 0.8 秒。**这类回归只有端到端评估能发现，
# 单看检索指标是看不出来的。**
RERANK_MAX_CHARS = int(os.getenv("RERANK_MAX_CHARS", "400"))


def _rerank_text(row: dict) -> str:
    """把片段转成「适合 cross-encoder 读」的文本。

    为什么不能直接把 Markdown 表格喂给 reranker？
        实测踩过：问「苹果多少钱一斤」，正确答案在 bom.csv 的表格片段里，
        内容长这样：
            | product | qty | price |
            | --- | --- | --- |
            | 苹果 | 10 | 3.5 |
        cross-encoder 是在自然语言句对上训练的，看到一堆竖线和英文表头
        完全读不懂，给了 -6.3 的负分，结果把无关的报销章节排到了第一。

        转成「product=苹果，qty=10，price=3.5」之后，模型就能正常判断了。
    这个细节说明：**检索链路里每一环的输入格式都会影响效果**，
    不是换个更强的模型就能自动解决。
    """
    if row.get("kind") != "table":
        return row["content"]

    lines = [ln for ln in row["content"].splitlines() if ln.strip()]
    if len(lines) < 3:
        return row["content"]

    def cells(ln: str) -> list:
        return [c.strip() for c in ln.strip().strip("|").split("|")]

    header = cells(lines[0])
    if not header:
        return row["content"]

    pairs = []
    for ln in lines[2:22]:                # 最多 20 行，够 model 判断主题了
        row_cells = cells(ln)
        kv = "，".join(
            f"{h}={v}" for h, v in zip(header, row_cells) if v
        )
        if kv:
            pairs.append(kv)

    if not pairs:
        return row["content"]

    return f"表格数据，列：{'、'.join(header)}。" + "；".join(pairs) + "。"


def rerank(query: str, results: list, top_n: int = None) -> list:
    """对候选片段重排。

    Args:
        query: 用户问题
        results: 检索结果（已按 RRF 排好序）
        top_n: 保留几条，默认读 RERANK_TOP_N

    Returns:
        重排后的结果。每个元素会多出 rerank_score 和 rerank_rank 字段。
        失败时原样返回（并在每项标记 rerank_skipped=True）。
    """
    if not results:
        return results

    top_n = top_n or RERANK_TOP_N

    # 只对前 N 个候选做精排，控制延迟
    head = results[:RERANK_CANDIDATES]
    tail = results[RERANK_CANDIDATES:]

    if not available():
        for r in results:
            r["rerank_skipped"] = True
        return results[:top_n]

    docs = [_rerank_text(r)[:RERANK_MAX_CHARS] for r in head]

    try:
        if RERANK_PROVIDER == "api":
            scores = _score_api(query, docs)
        else:
            scores = _score_local(query, docs)
    except Exception as exc:
        print(f"[rerank] 打分失败（{exc}），退回 RRF 顺序")
        for r in results:
            r["rerank_skipped"] = True
        return results[:top_n]

    if not scores or len(scores) != len(head):
        for r in results:
            r["rerank_skipped"] = True
        return results[:top_n]

    for r, s in zip(head, scores):
        r["rerank_score"] = round(float(s), 4)

    ordered = sorted(head, key=lambda x: -x.get("rerank_score", 0))
    ordered = [r for r in ordered if r.get("rerank_score", 0) >= RERANK_MIN_SCORE]

    for i, r in enumerate(ordered, start=1):
        r["rerank_rank"] = i

    out = ordered[:top_n]
    # 精排砍掉的候选还留着（tail 本来就不参与），但不再返回
    _ = tail
    return out


def info() -> dict:
    """给 /api/rag/stats 用的状态信息。"""
    return {
        "provider": RERANK_PROVIDER,
        "model": RERANK_MODEL if RERANK_PROVIDER != "none" else None,
        "available": available(),
        "candidates": RERANK_CANDIDATES,
        "top_n": RERANK_TOP_N,
    }


if __name__ == "__main__":
    q = "公积金缴存比例是多少"
    cands = [
        {"content": "公司每年组织两次技术分享会，鼓励员工参加外部行业会议。"},
        {"content": "公司为员工缴纳五险一金，公积金缴存比例为百分之十二。"},
        {"content": "员工每日需保证八小时工作时长，迟到三次扣全勤奖。"},
    ]
    print("未重排顺序：")
    for i, c in enumerate(cands, 1):
        print(f"  {i}. {c['content'][:24]}")
    out = rerank(q, [dict(c) for c in cands], top_n=3)
    print("\n重排后：")
    for r in out:
        print(f"  [{r.get('rerank_score')}] {r['content'][:24]}")