"""针对「词汇鸿沟」的评估集。

普通评估集（eval_retrieval.CASES）测的是整体检索质量，
但混合检索在上面已经满分了，看不出查询改写的价值。

这里的题目专门设计成**用户用词和文档用词对不上**：
    用户问「住酒店」      文档写「住宿标准」
    用户问「打车」        文档写「市内交通费」
    用户问「奖金」        文档写「绩效工资」
    用户问「吃饭有补贴吗」文档写「餐补」
这就是查询改写要解决的问题 —— 词面对不上时，BM25 零命中，
向量只能靠语义猜。

用法：
    python eval_hard_cases.py           # 对比改写开关
"""

import sys

import vector_store

# (问题, 答案关键词, 文档名片段)
# 关键点：问题里的词在文档中**都不出现**
GAP_CASES = [
    ("住酒店能报多少", ["六百", "600"], "员工手册"),
    ("打车能报销吗", ["五十", "50"], "员工手册"),
    ("奖金怎么算", ["季度", "绩效"], "员工手册"),
    ("吃饭有补贴吗", ["三百", "300"], "员工手册"),
    ("离职后能去竞争对手公司吗", ["两年", "保密"], "员工手册"),
    ("体检去哪儿做", ["美年"], "员工手册"),
    ("迟到会扣钱吗", ["全勤"], "员工手册"),
    ("请个假要谁批", ["主管"], "员工手册"),
    ("出差住一晚上限多少", ["六百", "600"], "员工手册"),
    ("坐地铁的钱能不能报", ["五十", "50"], "员工手册"),
]


def run(use_rewrite: bool, top_k: int = 1, use_rerank: bool = False):
    hits, total, fails = 0, 0, []
    import eval_retrieval as ev

    for q, kws, dn in GAP_CASES:
        ids = ev.doc_ids_by_name(dn)
        try:
            rows = vector_store.search(q, top_k=top_k, doc_ids=ids,
                                       mode="hybrid", use_rerank=use_rerank,
                                       use_rewrite=use_rewrite)
        except Exception as exc:
            fails.append((q, f"报错 {exc}"))
            total += 1
            continue
        total += 1
        joined = "\n".join(r["content"] for r in rows)
        if any(k in joined for k in kws):
            hits += 1
        else:
            fails.append((q, "未命中"))
    return hits, total, fails


def main():
    print(f"词汇鸿沟评估集（{len(GAP_CASES)} 条，均关闭精排以隔离改写效果）\n")

    b_hits, total, b_fails = run(use_rewrite=False)
    r_hits, _, r_fails = run(use_rewrite=True)

    print(f"Hit@1  不改写  {b_hits}/{total}  ({b_hits/total*100:.0f}%)")
    print(f"Hit@1  有改写  {r_hits}/{total}  ({r_hits/total*100:.0f}%)")
    print(f"提升：{r_hits - b_hits:+d} 条\n")

    if b_fails:
        print("不改写时失败的题：")
        for q, why in b_fails:
            fixed = q not in [x[0] for x in r_fails]
            print(f"  {q}  ({why})  → {'改写后修好' if fixed else '改写也没救'}")

    if len(sys.argv) > 1 and sys.argv[1] == "-v":
        print("\n=== 逐条详情（改写版）===")
        import eval_retrieval as ev
        for q, kws, dn in GAP_CASES:
            rows = vector_store.search(q, top_k=1, doc_ids=ev.doc_ids_by_name(dn),
                                       mode="hybrid", use_rerank=False,
                                       use_rewrite=True)
            rw = rows[0].get("rewrite", {}) if rows else {}
            ok = bool(rows and any(k in rows[0]["content"] for k in kws))
            print(f"\n问题：{q}")
            print(f"  改写：{rw.get('rewritten')}")
            print(f"  top1：{'正确' if ok else '错误'}  hits={rows[0].get('query_hits') if rows else 0}")


if __name__ == "__main__":
    main()