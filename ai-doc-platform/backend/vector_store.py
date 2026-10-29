"""向量存储与混合检索。

三种检索模式（`RETRIEVAL_MODE` 控制）：

    1. `hybrid`（默认）向量 + BM25，用 RRF 融合
    2. `vector`       只用语义向量
    3. `bm25`         只用关键词

为什么要混合？
    两路各有一个致命的短板：
      - 纯向量：查「2026 年」「百分之十二」「错误码 E4021」这类精确内容时，
        向量把数字和人名都模糊成"语义相近"，经常召回错的片段
      - 纯 BM25：查「出差住酒店能报多少钱」时，文档里写的是"住宿标准"，
        词面完全对不上，直接零命中
    实测同一个问题，两路召回的第一名常常不同 —— 融合才稳。

为什么用 RRF 而不是加权平均分数？
    余弦相似度是 0~1，BM25 是无上界的正数（可以到 10+），两者**量纲不可比**。
    归一化后再加权？归一化依赖当前候选集的分布，语料一变权重就漂。
    RRF（Reciprocal Rank Fusion）只用**排名**不用分数：
        score(d) = Σ  1 / (k + rank_i(d))     k 经验值 60
    好处：免调参、跨模型通用、对分数尺度完全不敏感。
    这是 RAG 混合检索的事实标准做法。

为什么没上 Chroma / FAISS / Milvus？
    - 数据量：几千个切片，暴力算余弦是毫秒级
    - 依赖成本：Chroma 在 Windows + Python 3.13 上容易出问题，
      为几千条数据引重型依赖不划算
    - 可解释性：自己写一遍，能讲清「向量怎么存、相似度怎么算、为什么这么设计」
    代价与边界：暴力检索是 O(n·d)。到十万级再换 ANN（HNSW / IVF），
    那时候才需要 Milvus、Qdrant 或 pgvector。
"""

import os
import time

import bm25
import db
import embedding
import query_rewrite
import rerank as reranker

RRF_K = int(os.getenv("RRF_K", "60"))
# 每路各取多少个候选参与融合。候选太少融合没意义，太多会引入噪声
CANDIDATE_DEPTH = int(os.getenv("CANDIDATE_DEPTH", "30"))

# RRF 权重。默认等权（1:1），但**这个值必须在你自己的语料上标定**。
#
# 实测数据（同一个系统，两类语料，差异很大）：
#   语料 A：19 份课程实验报告（Vue/Linux/Python），112 个切片
#       1:3 → 13/16    1:1 → 14/16    1:0.3 → 15/16
#       这类语料大量重复「实验」「报告」「截图」等词，BM25 权重高反而添乱
#
#   语料 B：60 条故障代码手册，每条只有编号不同
#       1:3 → 7/7      1:1 → 5/7      1:0.3 → 3/7
#       这类语料编号是唯一标识，BM25 的词面精确匹配不可替代
#
#   混合语料合计：1:3 → 20/23（最优），1:1 → 19/23，1:0.3 → 18/23
#
# 结论：没有通用最优值，取决于「唯一标识类内容」占多少。
# 换语料后跑 `python eval_real.py --tune` 自动扫一遍，别抄经验值。
RRF_WEIGHT_VECTOR = float(os.getenv("RRF_WEIGHT_VECTOR", "1.0"))
RRF_WEIGHT_BM25 = float(os.getenv("RRF_WEIGHT_BM25", "1.0"))

# 跨文档分散：按文档配额强制多样化。
#   实测这个方向行不通，默认关闭（=1），保留实现作为对照：
#   问「Vue 表单绑定和 Linux 创建目录分别怎么做」时，候选池前两名文档都是 Vue，
#   配额只能保证「两篇不同文档」，保证不了「覆盖两个意图」，
#   要 min_docs=4 这种极端值才碰巧捞到第二篇相关文档，而那样会毁掉单文档题。
#   真正有效的是下面的子问题分解。
DIVERSITY_MIN_DOCS = int(os.getenv("DIVERSITY_MIN_DOCS", "1"))
DIVERSITY_MAX_PER_DOC = int(os.getenv("DIVERSITY_MAX_PER_DOC", "3"))
DIVERSITY_RECALL = int(os.getenv("DIVERSITY_RECALL", "20"))

# 向量召回的绝对下限（默认 0 = 关闭）。
# 为什么需要它：语义向量对**任何**文本都会给出非零相似度，即使完全不相关。
# 这是 embedding 的固有特性，不是 bug。bge 系列实测：相关 0.5~0.7，
# 无关 0.2~0.4。数据量小的时候无所谓（反正都会进候选），
# 但文档多了以后弱召回会挤掉真正相关的片段，可以调到 0.35 左右。
VECTOR_MIN_SCORE = float(os.getenv("VECTOR_MIN_SCORE", "0"))


def index_document(doc_id: str, text: str, ext: str = "",
                   chunk_size: int = None, overlap: int = None) -> int:
    """把文档切片 + 向量化 + 入库。返回切片数量。

    先删后插：重复索引同一文档不会留下脏数据。
    """
    import chunking

    kw = {}
    if chunk_size:
        kw["size"] = chunk_size
    if overlap is not None:
        kw["overlap"] = overlap

    chunks = chunking.chunk_document(text, ext, **kw)
    if not chunks:
        return 0

    vectors = embedding.embed_texts([c["content"] for c in chunks])

    db.delete_chunks(doc_id)
    rows = []
    for c, v in zip(chunks, vectors):
        rows.append({
            "doc_id": doc_id,
            "chunk_index": c["index"],
            "content": c["content"],
            "kind": c.get("kind", "text"),
            "meta": c.get("meta", ""),
            "tokens": len(embedding.tokenize(c["content"])),
            "vec": embedding.to_blob(v),
            "dim": len(v),
        })

    db.insert_chunks(rows)
    bm25.invalidate()          # 语料变了，BM25 内存索引必须重建
    return len(rows)


def rrf_fuse(rankings: dict, k: int = RRF_K, weights: dict = None) -> list:
    """Reciprocal Rank Fusion。

    Args:
        rankings: {"vector": [(key, score), ...], "bm25": [(key, score), ...]}
                  每个列表必须已按相关性降序排列
        k: 平滑常数。k 越大，排名差异的影响越平缓。60 是论文经验值
        weights: 可选，给某一路加权（默认等权）

    Returns:
        [(key, {"rrf": float, "ranks": {...}, "scores": {...}})] 按 rrf 降序
    """
    weights = weights or {}
    fused = {}

    for name, ranked in rankings.items():
        w = float(weights.get(name, 1.0))
        for rank, (key, score) in enumerate(ranked, start=1):
            e = fused.setdefault(key, {"rrf": 0.0, "ranks": {}, "scores": {}})
            # 只依赖名次，不依赖分数大小 —— 这是 RRF 能跨模型通用的原因
            e["rrf"] += w / (k + rank)
            e["ranks"][name] = rank
            e["scores"][name] = float(score)

    return sorted(fused.items(), key=lambda x: -x[1]["rrf"])


def _rank_single(query: str, rows: list, mode: str, weights: dict,
                 recall_n: int, min_score: float) -> list:
    """单查询检索：跑向量 + BM25，内部融合，返回 [(key, info)]。

    抽成独立函数是为了让「多查询改写」能复用 —— 每个改写查询各跑一遍，
    再做跨查询融合。原查询权重最高，改写的只用来补召回。
    """
    # ---------- 第一路：向量检索 ----------
    vec_ranked = []
    if mode in ("hybrid", "vector"):
        qvec = embedding.embed_texts([query])[0]

        # 维度必须一致，否则是换了 embedding 模型但没重建索引
        dims = {r["dim"] for r in rows}
        if len(dims) > 1 or dims.pop() != len(qvec):
            raise RuntimeError(
                "向量维度不一致：索引里的向量和当前 embedding 模型不匹配。"
                "换了 EMBEDDING_PROVIDER 或 EMBEDDING_MODEL 之后，需要重建索引"
                "（调用 POST /api/reindex）。"
            )

        matrix = [embedding.from_blob(r["vec"]) for r in rows]
        scores = embedding.similarity_scores(qvec, matrix)

        pairs = [(bm25.chunk_key(r), s) for r, s in zip(rows, scores) if s > 0]
        pairs.sort(key=lambda x: -x[1])
        if mode == "vector":
            # 纯向量模式保留 min_score 过滤
            pairs = [p for p in pairs if p[1] >= min_score]
            vec_ranked = pairs[:recall_n]
        else:
            # 混合模式也支持全局阈值，压掉"什么都能召回"的噪声
            if VECTOR_MIN_SCORE > 0:
                pairs = [p for p in pairs if p[1] >= VECTOR_MIN_SCORE]
            vec_ranked = pairs[:max(CANDIDATE_DEPTH, recall_n)]

    # ---------- 第二路：BM25 关键词检索 ----------
    bm_ranked = []
    if mode in ("hybrid", "bm25"):
        index = bm25.get_index(rows, embedding.tokenize)
        depth = max(CANDIDATE_DEPTH, recall_n) if mode == "hybrid" else recall_n
        bm_ranked = index.search(query, embedding.tokenize, top_k=depth)

    # ---------- 单查询内部融合 ----------
    if mode == "vector":
        return [(key, {"rrf": s, "ranks": {"vector": i}, "scores": {"vector": s}})
                for i, (key, s) in enumerate(vec_ranked, start=1)]
    if mode == "bm25":
        return [(key, {"rrf": s, "ranks": {"bm25": i}, "scores": {"bm25": s}})
                for i, (key, s) in enumerate(bm_ranked, start=1)]

    rankings = {}
    if vec_ranked:
        rankings["vector"] = vec_ranked
    if bm_ranked:
        rankings["bm25"] = bm_ranked
    fused_weights = weights or {
        "vector": RRF_WEIGHT_VECTOR,
        "bm25": RRF_WEIGHT_BM25,
    }
    return rrf_fuse(rankings, weights=fused_weights)[:recall_n]


def spread_by_doc(results: list, top_k: int, min_docs: int = 2,
                  max_per_doc: int = 3) -> list:
    """按文档分散选取，避免 top_k 被单篇文档占满。

    为什么需要它？
        跨文档题的典型失败长这样：问「Vue 的表单绑定和 Linux 的创建目录分别怎么做」，
        前 20 个候选里 8 条是 Vue 文档、mkdir 那条排第 6 —— 而 top_k=5 被 Vue 占满，
        模型只看到 Vue，只能答一半。
        实测这条题在端到端评估里是唯一的失败项，归因矩阵显示「检索未命中」。
        **精排救不了它**：精排只在给定候选里重新排序，不会把第 6 名之外的东西变进来。

    策略（MMR 的简化版，不做向量相似度惩罚，只做文档配额）：
        1. 先给每篇文档保底 1 条（按各文档最高分的先后取前 min_docs 篇）
        2. 剩余槽位按全局相关性补满，但单篇文档不超过 max_per_doc 条
        3. 最后仍按原始相关性排序 —— 多样性只影响「选谁」，不改变「谁更相关」

    为什么不用真正的 MMR？
        MMR 要算片段两两之间的相似度，O(n²) 且需要向量；而这里的问题
        只是"同质内容挤占名额"，按文档配额就能解决，简单可解释。

    风险控制：
        如果问题只涉及一篇文档，强制第 2 篇会挤掉一个相关片段。
        所以 max_per_doc 默认 3（不是 2），保证主文档仍能占多数名额；
        而且 min_docs 只保底 1 条，代价可控。
    """
    if not results:
        return []
    if min_docs <= 1 or len(results) <= top_k:
        return results[:top_k]

    # 1. 各文档的最高分片段（results 已按相关性降序，所以首次出现即最高分）
    first_of_doc = {}
    for r in results:
        if r["filename"] not in first_of_doc:
            first_of_doc[r["filename"]] = r

    selected, used = [], set()
    for r in list(first_of_doc.values())[:min(min_docs, top_k)]:
        selected.append(r)
        used.add(id(r))

    # 2. 剩余槽位按全局相关性补，单文档数量设上限
    per_doc = {}
    for r in selected:
        per_doc[r["filename"]] = per_doc.get(r["filename"], 0) + 1

    for r in results:
        if len(selected) >= top_k:
            break
        if id(r) in used:
            continue
        if per_doc.get(r["filename"], 0) >= max_per_doc:
            continue
        selected.append(r)
        used.add(id(r))
        per_doc[r["filename"]] = per_doc.get(r["filename"], 0) + 1

    # 3. 恢复相关性顺序
    order = {id(r): i for i, r in enumerate(results)}
    selected.sort(key=lambda r: order[id(r)])
    return selected[:top_k]


def merge_sub_queries(subs: list, base_results: list, top_k: int,
                      doc_ids: list, mode: str) -> list:
    """子问题各检索一遍再合并。

    每个子问题至少贡献 1 条，保证「两个意图都覆盖到」；
    剩余槽位用原问题的结果补满。

    子查询检索刻意关掉精排：
        精排是按**整句**打分的，拿整句去评「Linux 创建目录」这个片段会打低分，
        反而把正确答案压下去 —— 这正是跨文档题之前失败的原因。
        子查询只在各自的小范围内找最相关的 1-2 条，不需要精排。
    """
    per = max(1, top_k // len(subs))

    merged, seen = [], set()

    for s in subs:
        for r in search(s, top_k=per, doc_ids=doc_ids, mode=mode,
                        use_rerank=False, use_rewrite=False,
                        min_docs=1, _no_decompose=True):
            key = (r["filename"], r["chunk_index"])
            if key in seen:
                continue
            seen.add(key)
            r["matched_sub"] = s          # 标明是为哪个子问题召回的
            merged.append(r)
            if len(merged) >= top_k:
                break
        if len(merged) >= top_k:
            break

    # 补位分两轮：先补「还没覆盖的文档」，再随便补满。
    # 为什么要有第一轮？
    #   子查询的 top 结果不一定覆盖到目标文档 —— 比如把「查看当前路径」
    #   加上「Linux」前缀后，含 pwd 的片段掉到第 4 名，子查询只带回了别的文档。
    #   这时如果按原顺序补位，补进来的很可能还是已覆盖的文档，
    #   白占名额。优先补未覆盖文档，等于用原查询的结果兜底。
    covered_docs = {r["filename"] for r in merged}

    for r in base_results:
        if len(merged) >= top_k:
            break
        key = (r["filename"], r["chunk_index"])
        if key in seen or r["filename"] in covered_docs:
            continue
        seen.add(key)
        merged.append(r)
        covered_docs.add(r["filename"])

    for r in base_results:
        if len(merged) >= top_k:
            break
        key = (r["filename"], r["chunk_index"])
        if key in seen:
            continue
        seen.add(key)
        merged.append(r)

    return merged[:top_k]


def search(query: str, top_k: int = 4, doc_ids: list = None,
           min_score: float = 0.0, mode: str = None,
           weights: dict = None, use_rerank: bool = None,
           use_rewrite: bool = None, use_decompose: bool = None,
           min_docs: int = None,
           _no_decompose: bool = False) -> list:
    """检索入口。

    Args:
        query: 用户问题
        top_k: 最终返回片段数
        doc_ids: 限定文档范围；None 表示全部（多文档检索）
        min_score: 向量相似度下限，只在纯向量模式下生效
        mode: hybrid / vector / bm25，默认读环境变量 RETRIEVAL_MODE
        weights: RRF 权重，如 {"vector":1.0,"bm25":0.8}。默认等权
        use_rerank: 是否用 cross-encoder 精排。None 时读 RERANK_ENABLED
        use_rewrite: 是否做查询改写 + 多查询检索。None 时读 QUERY_REWRITE_ENABLED

    Returns:
        [{"doc_id","filename","chunk_index","content","kind","meta",
          "score","rrf","ranks","matched_by","vector_score","bm25_score",
          "rerank_score","rerank_rank","query_hits"}]
    """
    mode = (mode or os.getenv("RETRIEVAL_MODE", "hybrid")).strip().lower()
    if mode not in ("hybrid", "vector", "bm25"):
        mode = "hybrid"

    if use_rerank is None:
        # 默认开启。57 条题的评估里精排把 Hit@1 从 36/49 提到 44/49。
        # 注意：16 条题时它是负收益（-1），题量扩到 57 条后才显出真实效果。
        # 小样本评估容易得出相反结论，这点写进 README 了。
        use_rerank = os.getenv("RERANK_ENABLED", "true").lower() in ("1", "true", "yes")

    if use_rewrite is None:
        use_rewrite = query_rewrite.REWRITE_ENABLED

    if min_docs is None:
        min_docs = DIVERSITY_MIN_DOCS

    if use_decompose is None:
        use_decompose = query_rewrite.DECOMPOSE_ENABLED

    # 要 rerank 就得先多召回一些候选给它精排；
    # 要做跨文档分散也得有候选池，否则第 6 名的片段永远进不来
    if use_rerank or min_docs > 1:
        recall_n = max(top_k, reranker.RERANK_CANDIDATES)
        if min_docs > 1:
            recall_n = max(recall_n, DIVERSITY_RECALL)
    else:
        recall_n = top_k

    rows = db.load_chunk_vectors(doc_ids)
    if not rows:
        return []

    row_by_key = {bm25.chunk_key(r): r for r in rows}

    # ---------- 查询改写 ----------
    if use_rewrite is None:
        use_rewrite = query_rewrite.REWRITE_ENABLED

    rw = {"original": query, "queries": [query], "rewritten": [], "cached": False}
    if use_rewrite:
        # 显式传 use_llm=True：调用方明确要求改写时，不能被 .env 的默认关闭值覆盖
        rw = query_rewrite.rewrite(query, use_llm=True)

    queries = rw["queries"] or [query]

    if len(queries) == 1:
        final = _rank_single(query, rows, mode, weights, recall_n, min_score)
        q_infos = {0: dict(final)}
    else:
        # 每个查询各检索一遍，再跨查询融合
        rankings, q_weights, q_infos = {}, {}, {}
        for i, q in enumerate(queries):
            ranked = _rank_single(q, rows, mode, weights, recall_n, min_score)
            name = f"q{i}"
            rankings[name] = [(k, info["rrf"]) for k, info in ranked]
            # 原查询权重更高 —— 改写只是补召回，不能反客为主
            q_weights[name] = (
                query_rewrite.ORIGINAL_WEIGHT if i == 0 else 1.0
            )
            q_infos[i] = dict(ranked)

        final = rrf_fuse(rankings, weights=q_weights)[:recall_n]

    # ---------- 组装结果 ----------
    results = []
    for key, info in final:
        r = row_by_key.get(key)
        if not r:
            continue

        # 命中来源：优先取原查询的通道信息（vector / bm25），
        # 原查询没召回到的，再找哪个改写查询召回它
        src = None
        for i in range(len(queries)):
            if key in q_infos.get(i, {}):
                src = q_infos[i][key]
                break

        hits = sum(1 for i in range(len(queries)) if key in q_infos.get(i, {}))

        results.append({
            "doc_id": r["doc_id"],
            "filename": r["filename"],
            "chunk_index": r["chunk_index"],
            "content": r["content"],
            "kind": r["kind"],
            "meta": r["meta"],
            # score 统一用 RRF 分数；纯模式下落回原始分数
            "score": round(info["rrf"], 6),
            "rrf": round(info["rrf"], 6),
            "ranks": info["ranks"],
            "matched_by": sorted(src["ranks"].keys()) if src else [],
            "vector_score": round(src["scores"]["vector"], 4)
            if src and "vector" in src["scores"] else None,
            "bm25_score": round(src["scores"]["bm25"], 4)
            if src and "bm25" in src["scores"] else None,
            # 有几个查询召回了它 —— 多查询都命中说明更可靠
            "query_hits": hits,
        })

    # ---------- 精排 ----------
    if use_rerank and results:
        # 注意保留 recall_n 条，不能直接截到 top_k ——
        # 否则后面的跨文档分散没有候选可挑，第 6 名的片段永远进不来
        results = reranker.rerank(query, results, top_n=recall_n)

    # 把改写信息挂到结果上，方便上层展示和调试
    for r in results:
        r["rewrite"] = rw

    # ---------- 跨文档分散 ----------
    if min_docs > 1:
        final = spread_by_doc(results, top_k, min_docs, DIVERSITY_MAX_PER_DOC)
    else:
        final = results[:top_k]

    # ---------- 子问题分解（跨文档题的关键）----------
    # 放在最后：先拿到原问题的正常结果，再用子问题去补齐缺失的意图
    if use_decompose and not _no_decompose and top_k >= 2:
        dec = query_rewrite.decompose(query)
        subs = dec.get("subs") or []
        if subs:
            final = merge_sub_queries(subs, final, top_k, doc_ids, mode)
            for r in final:
                r["decode"] = dec        # 供上层展示「问题被拆成了什么」

    return final


def build_context(results: list, max_chars: int = None) -> str:
    """把检索结果拼成给模型的上下文。

    每个片段带来源标注，模型回答时可以引用「片段2」，
    前端也能显示依据，这是 RAG 可解释性的关键。
    """
    budget = max_chars or int(os.getenv("RAG_CONTEXT_BUDGET", "8000"))
    parts, used = [], 0

    for i, r in enumerate(results, start=1):
        header = f"【片段{i}｜来源：{r['filename']}"
        if r.get("meta"):
            header += f"｜{r['meta']}"
        header += "】"

        block = f"{header}\n{r['content']}"
        if used + len(block) > budget:
            break
        parts.append(block)
        used += len(block)

    return "\n\n".join(parts)
