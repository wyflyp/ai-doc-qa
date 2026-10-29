"""真实语料评估：用你自己的 19 份课程实验报告（Vue / Linux / Python）做检索评测。

为什么这份评估比之前的有意义？
    之前的测试文档都是我自己造的，总共 25 个切片，而候选深度就是 20 ——
    几乎所有片段都进了候选池，任何检索优化都测不出差别（评测环境饱和）。

    换成真实语料后有 112 个切片、26 个文档，且不同课程之间有大量相似词汇
    （「实验」「截图」「运行结果」「Vue」「程序」），检索必须真正做选择。

    而且这里**不加文档过滤**（全库检索），比指定文档更难。

指标的局限（要说清楚）：
    用「答案关键词是否出现在 top-K」判断命中，这是 Recall 的简化版。
    关键词出现不等于答案好，严格要做人工标注。但用来对比不同检索配置
    的相对高低是够的。

用法：
    python eval_real.py            # 全部配置对比
    python eval_real.py -v         # 加逐条详情
"""

import sys

import vector_store

# (问题, 答案关键词, 说明)
# 关键词都预先验证过确实存在于文档中，避免出「文档里根本没有」的废题
REAL_CASES = [
    # ---- Vue 实验（8 份文档）----
    ("v-show 和 v-if 有什么区别", ["v-show"], "Vue 实验二"),
    ("子组件怎么向父组件传值", ["emit", "$emit"], "Vue 实验五"),
    ("作用域插槽怎么用", ["作用域插槽"], "Vue 实验六"),
    ("动态路由参数怎么接收", ["params"], "Vue 实验七"),
    ("axios 怎么发请求", ["axios"], "Vue 实验八"),
    ("表单输入怎么双向绑定", ["v-model"], "Vue 实验四"),
    ("列表渲染用哪个指令", ["v-for"], "Vue 实验二"),
    ("侦听器怎么用", ["watch"], "Vue 实验一"),
    ("组件生命周期有哪些钩子", ["生命周期"], "Vue 实验六"),
    # ---- Linux 实验（5 份文档）----
    ("vim 有哪几种模式", ["命令模式"], "Linux 实验四"),
    ("实验用的什么操作系统", ["Ubuntu"], "Linux 实验一"),
    ("实验环境是虚拟机吗", ["虚拟机"], "Linux 实验一"),
    # ---- Python 讲义与作业 ----
    ("元组和列表有什么区别", ["元组"], "实验五讲义"),
    ("列表推导式怎么写", ["推导式"], "实验五讲义"),
    ("怎么统计字符串里各字母出现的次数", ["count_letter", "count"], "Python 作业三"),
    ("1234 能组成多少个不重复的三位数", ["不重复", "count"], "Python 作业二"),
]

PLANS = [
    ("vector", "vector", False, False),
    ("bm25", "bm25", False, False),
    ("hybrid", "hybrid", False, False),
    ("hybrid+改写", "hybrid", False, True),
    ("hybrid+精排", "hybrid", True, False),
    ("全开", "hybrid", True, True),
]


def run(mode: str, use_rerank: bool, use_rewrite: bool, top_k: int = 3):
    hits, total, fails = 0, 0, []
    for q, kws, note in REAL_CASES:
        try:
            rows = vector_store.search(q, top_k=top_k, doc_ids=None,
                                       mode=mode, use_rerank=use_rerank,
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
            fails.append((q, note))
    return hits, total, fails


def tune_weights(k: int = 1):
    """在当前语料上扫描 RRF 权重，找出最适合的配置。

    为什么需要这个？
        权重不该拍脑袋定。我实测过：
          报告类语料（大量重复的「实验」「报告」「截图」）→ BM25 权重要低
          编号类语料（E4027 这种唯一标识）        → BM25 权重要高
        同一组权重在两类语料上能差 6 个百分点。
        换语料后跑一次这个，比抄别人的经验值可靠。

    用法：python eval_real.py --tune
    """
    print(f"权重扫描（{len(REAL_CASES)} 条问题，全库检索，Hit@{k}）\n")
    print(f"{'vector:bm25':>14}  命中")
    print("-" * 30)

    best = (None, -1)
    for wv, wb in [(1, 3), (1, 2), (1, 1), (1, 0.7), (1, 0.5), (1, 0.3), (2, 1), (3, 1)]:
        hits = 0
        for q, kws, _ in REAL_CASES:
            rows = vector_store.search(q, top_k=k, doc_ids=None, mode="hybrid",
                                       use_rerank=False, use_rewrite=False,
                                       weights={"vector": wv, "bm25": wb})
            if any(x in "\n".join(r["content"] for r in rows) for x in kws):
                hits += 1
        bar = "█" * hits + "░" * (len(REAL_CASES) - hits)
        mark = ""
        if hits > best[1]:
            best = ((wv, wb), hits)
            mark = "  ← 当前最优"
        print(f"{wv}:{wb:<12} {hits:>2}/{len(REAL_CASES)}  {bar}{mark}")

    print(f"\n最优：vector={best[0][0]} bm25={best[0][1]}（{best[1]}/{len(REAL_CASES)}）")
    print("写进 .env：")
    print(f"  RRF_WEIGHT_VECTOR={best[0][0]}")
    print(f"  RRF_WEIGHT_BM25={best[0][1]}")
    print("\n注意：这只是在这批问题上最优，不代表通用。语料变了要重跑。")


def main():
    if "--tune" in sys.argv:
        return tune_weights()

    verbose = "-v" in sys.argv
    n = len(REAL_CASES)
    print(f"真实语料评估：{n} 条问题，全库检索（不加文档过滤）\n")

    results = {}
    for k in (1, 3):
        print(f"{'配置':14} Hit@{k}")
        print("-" * 34)
        for label, mode, rr, rw in PLANS:
            hits, total, fails = run(mode, rr, rw, top_k=k)
            results[(label, k)] = (hits, fails)
            bar = "█" * hits + "░" * (total - hits)
            print(f"{label:14} {hits:>2}/{total}  {bar}")
        print()

    # 找出各配置的失败题，帮助判断下一步该优化什么
    if verbose:
        print("=== 各配置未命中的题（Hit@1）===")
        for label, _, _, _ in PLANS:
            hits, fails = results[(label, 1)]
            if fails:
                print(f"\n{label}（{hits}/{n}）")
                for q, note in fails:
                    print(f"   ✗ {q}   [{note}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())