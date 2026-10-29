"""检索效果评估。

为什么需要这个？
    改切片参数、调 RRF 权重、换 embedding 模型之后，必须有客观指标判断
    到底是变好还是变坏。靠肉眼看几个 case 很容易自我欺骗。

指标说明（面试可以直接讲）：
    这里用的是 **Hit@K**：top-K 个片段里是否包含答案关键词。
    这是 Recall 的简化版 —— 严格做法需要人工标注每个问题对应哪些片段，
    算 Recall@K 和 MRR。课程项目用 Hit@K 足够说明问题，成本也低。

局限（要主动说）：
    关键词命中不等于答案可用。真正上线要人工标注 + 端到端评估
    （答案正确率），而不只是检索指标。

用法：
    python eval.py              # 测默认（混合）
    python eval.py vector       # 只测向量
    python eval.py all          # 三种模式对比
"""

import sys

import db
import vector_store

# (问题, 答案必须包含的关键词, 限定文档名 或 None=全库)
CASES = [
    ("年假有几天", ["十五天", "15"], "员工手册"),
    ("公积金缴存比例是多少", ["十二"], "员工手册"),
    ("一线城市住宿每晚多少钱", ["六百", "600"], "员工手册"),
    ("出差报销要在多久内提交", ["十个工作日"], "员工手册"),
    ("离职后保密义务持续多久", ["两年"], "员工手册"),
    ("入职培训多长时间", ["两周"], "员工手册"),
    ("绩效工资什么时候发", ["季度"], "员工手册"),
    ("体检机构是哪家", ["美年"], "员工手册"),
    ("张三多大年纪", ["28"], "gbk_semicolon"),
    ("苹果多少钱一斤", ["3.5"], "bom"),
    ("市区内打车能报多少", ["五十", "50"], "员工手册"),
    ("迟到几次扣全勤奖", ["三次"], "员工手册"),
    # 精确编号类：语义向量对「E4013」和「E4027」几乎分不开，
    # 这类必须靠 BM25 的词面精确匹配
    ("E4013 出现这个故障怎么处理", ["E4013"], "故障代码"),
    ("E4027 出现这个故障怎么处理", ["E4027"], "故障代码"),
    ("E4042 出现这个故障怎么处理", ["E4042"], "故障代码"),
    ("E4008 出现这个故障怎么处理", ["E4008"], "故障代码"),
]


def doc_ids_by_name(name: str):
    if not name:
        return None
    for d in db.list_documents_for_reindex():
        if name in d["filename"]:
            return [d["id"]]
    return None


def run(mode: str, top_k: int = 3, use_rerank: bool = False):
    hits, total = 0, 0
    misses = []
    for q, keywords, doc_name in CASES:
        ids = doc_ids_by_name(doc_name)
        try:
            results = vector_store.search(q, top_k=top_k, doc_ids=ids,
                                          mode=mode, use_rerank=use_rerank)
        except Exception as exc:
            misses.append((q, f"报错: {exc}"))
            total += 1
            continue

        total += 1
        joined = "\n".join(r["content"] for r in results)
        if any(k in joined for k in keywords):
            hits += 1
        else:
            misses.append((q, "未命中"))

    return hits, total, misses


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"

    plans = [
        ("vector", "vector", False),
        ("bm25", "bm25", False),
        ("hybrid", "hybrid", False),
        ("hybrid+rerank", "hybrid", True),
    ]
    if arg != "all":
        plans = [p for p in plans if p[0] == arg] or [("hybrid+rerank", "hybrid", True)]

    for k in (1, 3):
        print(f"########## Hit@{k} ##########")
        for label, mode, rr in plans:
            hits, total, misses = run(mode, top_k=k, use_rerank=rr)
            rate = hits / total * 100 if total else 0
            print(f"  {label:14} {hits}/{total} ({rate:.0f}%)")
            if misses and arg != "all":
                for q, why in misses:
                    print(f"        未命中: {q}  ({why})")
        print()


if __name__ == "__main__":
    main()