"""完整评估套件：57 条题，分三类，覆盖真实语料的典型场景。

为什么要分类？
    一个笼统的 Hit@K 数字说明不了什么。面试官问「怎么评估」时，
    你得能说清「哪类问题表现好、哪类差」。所以这里拆成三类：

    FACT  事实查找题 —— 答案在一个文档里，关键词独特（函数名、命令、组件名）
          考的是「精确定位能力」

    SYN   同义/缩写题 —— 用户的说法和文档用词不一样
          「住酒店」vs 文档的「住宿」、「SLA」vs 文档的「服务等级协议」
          考的是「跨词汇鸿沟能力」，这是 RAG 最真实的难点

    CROSS 跨文档题 —— 一个问题要两篇以上文档的信息
          比如「Vue 的表单绑定和 Linux 的创建目录分别用什么」
          考的是「多文档覆盖能力」，单靠一个片段答不了

判分规则的区别（重要）：
    FACT/SYN  看 top-K 里有没有答案关键词
    CROSS     要求**所有**关键词都出现，且来自 **≥2 个不同文件**
              —— 否则一个文档里碰巧有两个词就算对，那不叫跨文档

指标口径：
    Hit@1  第一条片段就够用（衡量「用户看完第一条能不能得到答案」）
    Hit@5  前五条里能找到（衡量「召回是否覆盖」）
    CROSS 只看 Hit@5 —— 要求 top1 就覆盖两篇文档不现实

用法：
    python eval_suite.py              # 六种配置全跑
    python eval_suite.py --quick      # 只跑 vector / bm25 / hybrid
    python eval_suite.py -v           # 加逐条失败明细
"""

import sys

import vector_store

# 题型, 问题, 答案关键词, 来源说明
CASES = [
    # ---------------- FACT：事实查找（28 条）----------------
    ("fact", "密码输入框怎么让提交按钮不可点击", ["disabled"], "Vue实验1"),
    ("fact", "成绩及格和不及格怎么分别显示", ["showPassed"], "Vue实验2"),
    ("fact", "阻止默认行为用哪个修饰符", ["prevent"], "Vue实验3"),
    ("fact", "阻止事件冒泡用哪个修饰符", ["stop"], "Vue实验3"),
    ("fact", "希望事件只触发一次用什么修饰符", ["once"], "Vue实验3"),
    ("fact", "下拉选择框的选项怎么渲染", ["option"], "Vue实验4"),
    ("fact", "父组件给子组件传值用什么", ["props"], "Vue实验5"),
    ("fact", "子组件给父组件传值靠什么", ["emit"], "Vue实验5"),
    ("fact", "具名插槽怎么定义", ["slot"], "Vue实验6"),
    ("fact", "博客详情页组件叫什么名字", ["BlogDetail"], "Vue实验7"),
    ("fact", "页面跳转用哪个标签", ["router-link"], "Vue实验7"),
    ("fact", "前端发网络请求用哪个库", ["axios"], "Vue实验8"),
    ("fact", "装虚拟机用什么软件", ["VMware"], "Linux实验1"),
    ("fact", "创建目录用什么命令", ["mkdir"], "Linux实验2"),
    ("fact", "查看当前所在路径用什么命令", ["pwd"], "Linux实验2"),
    ("fact", "在文件内容里搜关键字用什么命令", ["grep"], "Linux实验3"),
    ("fact", "新建一个空文件用什么命令", ["touch"], "Linux实验3"),
    ("fact", "复制整行用哪组按键", ["nyy"], "Linux实验4"),
    ("fact", "删除整行用哪组按键", ["ndd"], "Linux实验4"),
    ("fact", "shell 里怎么读入用户输入", ["read"], "Linux实验5"),
    ("fact", "shell 多分支判断用什么语句", ["case"], "Linux实验5"),
    ("fact", "判断一个数是不是丑数", ["is_ugly"], "提高题三"),
    ("fact", "怎么找出最长的单词", ["longest"], "提高题2"),
    ("fact", "最高分怎么求", ["max_score"], "提高题2"),
    ("fact", "统计字母出现次数用的字典变量叫什么", ["count_letter"], "作业三"),
    ("fact", "用户密码校验用的变量叫什么", ["user_pwd"], "作业三"),
    ("fact", "企业利润提成怎么计算", ["profit"], "作业二"),
    ("fact", "备份文件默认保留多少天", ["九十天"], "IT规范"),

    # ---------------- SYN：同义 / 缩写（21 条）----------------
    ("syn", "住酒店一晚上最多能报多少", ["八百"], "制度汇编"),
    ("syn", "公积金按什么比例缴", ["百分之十二"], "制度汇编"),
    ("syn", "年假能休几天", ["十五天"], "制度汇编"),
    ("syn", "登录密码多久必须换一次", ["九十天"], "制度汇编"),
    ("syn", "主动辞职要提前多久说", ["三十日"], "制度汇编"),
    ("syn", "加班之后怎么折算调休", ["一点五倍"], "制度汇编"),
    ("syn", "体检安排在哪家机构", ["美年"], "制度汇编"),
    ("syn", "出差住宿一晚上限多少", ["六百", "八百"], "员工手册/汇编"),
    ("syn", "打车费能报销多少", ["五十", "一百"], "员工手册/汇编"),
    ("syn", "商业秘密需要保守多久", ["三年", "两年"], "员工手册/汇编"),
    ("syn", "SLA 要求全年可用性达到多少", ["百分之九十九点九五"], "IT规范"),
    ("syn", "OKR 每季度定几个目标", ["三到五个"], "IT规范"),
    ("syn", "VPN 账号可以借给同事用吗", ["不得共享"], "IT规范"),
    ("syn", "OA 系统什么时候做维护", ["凌晨两点至四点"], "IT规范"),
    ("syn", "E4015 报错怎么处理", ["E4015"], "故障代码"),
    ("syn", "E4033 是什么故障", ["E4033"], "故障代码"),
    ("syn", "苹果一斤多少钱", ["3.5"], "bom.csv"),
    ("syn", "元组和列表有什么不一样", ["元组"], "实验五讲义"),
    ("syn", "随机数用哪个库生成", ["random"], "实验五讲义"),
    ("syn", "列表推导式怎么写", ["推导式"], "实验五讲义"),
    ("syn", "怎么把列表截取一部分出来", ["切片"], "实验五讲义"),

    # ---------------- CROSS：跨文档（8 条）----------------
    ("cross", "Vue 的表单双向绑定和 Linux 的创建目录分别怎么做", ["v-model", "mkdir"], "Vue4+Linux2"),
    ("cross", "Python 怎么统计字母出现次数，Vue 怎么做页面跳转", ["count_letter", "router-link"], "作业三+Vue7"),
    ("cross", "住宿报销上限和数据库备份保留期限分别是多少", ["八百", "九十天"], "汇编+IT规范"),
    ("cross", "公积金缴存比例和 SLA 可用性要求分别是多少", ["百分之十二", "百分之九十九点九五"], "汇编+IT规范"),
    ("cross", "vim 复制整行和 Vue 阻止事件冒泡分别怎么做", ["nyy", "stop"], "Linux4+Vue3"),
    ("cross", "Python 怎么找最长单词，Linux 怎么查看当前路径", ["longest", "pwd"], "提高题2+Linux2"),
    ("cross", "年假天数和 OKR 目标数量分别是多少", ["十五天", "三到五个"], "汇编+IT规范"),
    ("cross", "子组件怎么给父组件传值，Linux 查看文件内容用什么命令", ["emit", "cat"], "Vue5+Linux3"),
]

PLANS = [
    ("仅向量", "vector", False, False, False),
    ("仅BM25", "bm25", False, False, False),
    ("混合", "hybrid", False, False, False),
    ("混合+改写", "hybrid", False, True, False),
    ("混合+精排", "hybrid", True, False, False),
    ("混合+分解", "hybrid", False, False, True),
    ("精排+分解", "hybrid", True, False, True),
]

TOP_K = 5


def evaluate(mode: str, use_rerank: bool, use_rewrite: bool,
             use_decompose: bool = False, only_cross: bool = False) -> dict:
    """跑一遍全部题目，返回按类别拆分的命中统计。"""
    stat = {c: {"n": 0, "hit1": 0, "hit5": 0} for c in ("fact", "syn", "cross")}
    fails = []
    cases = [c for c in CASES if c[0] == "cross"] if only_cross else CASES

    for kind, q, kws, note in cases:
        try:
            rows = vector_store.search(q, top_k=TOP_K, doc_ids=None,
                                       mode=mode, use_rerank=use_rerank,
                                       use_rewrite=use_rewrite,
                                       use_decompose=use_decompose)
        except Exception as exc:
            fails.append((kind, q, f"报错 {exc}"))
            stat[kind]["n"] += 1
            continue

        stat[kind]["n"] += 1
        joined = "\n".join(r["content"] for r in rows)
        top1 = rows[0]["content"] if rows else ""

        if kind == "cross":
            # 要求所有关键词都出现，且来自 ≥2 个不同文件
            files_hit = set()
            ok_all = True
            for kw in kws:
                hit_files = {r["filename"] for r in rows if kw in r["content"]}
                if not hit_files:
                    ok_all = False
                    break
                files_hit |= hit_files
            if ok_all and len(files_hit) >= 2:
                stat[kind]["hit5"] += 1
                stat[kind]["hit1"] += 1     # cross 不看 hit1，这里同步仅为方便
            else:
                fails.append((kind, q, note))
        else:
            if any(k in top1 for k in kws):
                stat[kind]["hit1"] += 1
            if any(k in joined for k in kws):
                stat[kind]["hit5"] += 1
            else:
                fails.append((kind, q, note))

    return {"stat": stat, "fails": fails}


def tune_weights():
    """在 57 条题上扫 RRF 权重。

    为什么值得单做一次？
        16 条题上标定出的最优值是 bm25=0.3；题目扩到 57 条后答案可能变。
        题目构成变了，最优权重就变了 —— 这本身就是个结论。
    """
    print(f"权重扫描（{len(CASES)} 条题，Hit@1 = FACT+SYN 的 top1 命中，Hit@5 = 全部含 CROSS）\n")
    print(f"{'vector:bm25':>14}  Hit@1   Hit@5")
    print("-" * 34)

    best_h1, best_h5 = (None, -1), (None, -1)
    for wv, wb in [(1, 3), (1, 2), (1, 1), (1, 0.7), (1, 0.5), (1, 0.3), (2, 1), (3, 1)]:
        stat = {c: {"n": 0, "hit1": 0, "hit5": 0} for c in ("fact", "syn", "cross")}
        for kind, q, kws, note in CASES:
            rows = vector_store.search(q, top_k=TOP_K, doc_ids=None, mode="hybrid",
                                       use_rerank=False, use_rewrite=False,
                                       weights={"vector": wv, "bm25": wb})
            stat[kind]["n"] += 1
            if kind == "cross":
                files_hit, ok = set(), True
                for kw in kws:
                    hf = {r["filename"] for r in rows if kw in r["content"]}
                    if not hf:
                        ok = False
                        break
                    files_hit |= hf
                if ok and len(files_hit) >= 2:
                    stat[kind]["hit5"] += 1
            else:
                top1 = rows[0]["content"] if rows else ""
                if any(k in top1 for k in kws):
                    stat[kind]["hit1"] += 1
                if any(k in "\n".join(r["content"] for r in rows) for k in kws):
                    stat[kind]["hit5"] += 1

        h1 = stat["fact"]["hit1"] + stat["syn"]["hit1"]
        h1n = stat["fact"]["n"] + stat["syn"]["n"]
        h5 = stat["fact"]["hit5"] + stat["syn"]["hit5"] + stat["cross"]["hit5"]

        mark = ""
        if h1 > best_h1[1]:
            best_h1 = ((wv, wb), h1)
            mark = "  ← Hit@1 最优"
        if h5 > best_h5[1]:
            best_h5 = ((wv, wb), h5)
            mark += "  ← Hit@5 最优"
        print(f"{wv}:{wb:<12} {h1:>3}/{h1n}  {h5:>3}/{len(CASES)}{mark}")

    print(f"\nHit@1 最优：vector={best_h1[0][0]} bm25={best_h1[0][1]}（{best_h1[1]}/{h1n}）")
    print(f"Hit@5 最优：vector={best_h5[0][0]} bm25={best_h5[0][1]}（{best_h5[1]}/{len(CASES)}）")
    print("写进 .env 的示例：")
    print(f"  RRF_WEIGHT_VECTOR={best_h5[0][0]}")
    print(f"  RRF_WEIGHT_BM25={best_h5[0][1]}")


def main():
    if "--tune" in sys.argv:
        return tune_weights()

    quick = "--quick" in sys.argv
    slow = "--slow" in sys.argv
    cross_only = "--cross-only" in sys.argv
    verbose = "-v" in sys.argv

    if cross_only:
        plans = PLANS
    elif slow:
        plans = [p for p in PLANS if p[0] in ("混合+改写", "混合+精排")]
    elif quick:
        plans = [p for p in PLANS if p[0] in ("仅向量", "仅BM25", "混合")]
    else:
        plans = PLANS

    n_total = 8 if cross_only else len(CASES)
    scope = "CROSS 跨文档 8 题" if cross_only else "FACT 28 / SYN 21 / CROSS 8"
    print(f"评估套件：{scope}，全库检索，top_k={TOP_K}\n")

    if cross_only:
        # 只看跨文档：这是分解策略唯一要证明的地方，跑得快也能直接对比
        print(f"{'配置':16} CROSS Hit@5")
        print("-" * 34)
        for label, mode, rr, rw, dec in plans:
            res = evaluate(mode, rr, rw, use_decompose=dec, only_cross=True)
            c = res["stat"]["cross"]
            bar = "█" * c["hit5"] + "░" * (c["n"] - c["hit5"])
            print(f"{label:16} {c['hit5']:>2}/{c['n']}  {bar}")
        print()
        return 0

    all_results = {}
    for label, mode, rr, rw, dec in plans:
        res = evaluate(mode, rr, rw, use_decompose=dec)
        all_results[label] = res
        s = res["stat"]

        fact, syn, cross = s["fact"], s["syn"], s["cross"]
        h1 = fact["hit1"] + syn["hit1"]
        h1n = fact["n"] + syn["n"]
        h5 = fact["hit5"] + syn["hit5"] + cross["hit5"]

        print(f"── {label} ──")
        print(f"   FACT  Hit@1 {fact['hit1']:>2}/{fact['n']}   Hit@5 {fact['hit5']:>2}/{fact['n']}")
        print(f"   SYN   Hit@1 {syn['hit1']:>2}/{syn['n']}   Hit@5 {syn['hit5']:>2}/{syn['n']}")
        print(f"   CROSS （要求覆盖 2 篇文档）  Hit@5 {cross['hit5']:>2}/{cross['n']}")
        print(f"   合计  Hit@1 {h1}/{h1n}   Hit@5 {h5}/{n_total}")
        print()

    if verbose:
        print("=" * 52)
        print("各配置未命中的题（用于定位问题）")
        for label, res in all_results.items():
            if not res["fails"]:
                continue
            print(f"\n【{label}】{len(res['fails'])} 条未命中")
            for kind, q, note in res["fails"]:
                print(f"   [{kind}] {q}  ({note})")
    return 0


if __name__ == "__main__":
    sys.exit(main())