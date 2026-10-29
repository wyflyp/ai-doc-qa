"""BM25 关键词检索。

为什么向量检索之外还要 BM25？
    语义向量擅长「意思相近」，但对**精确匹配**不敏感：
      - 数字：「2026 年」「百分之十二」「五千元」
      - 专有名词：型号、人名、代码、错误码
      - 短查询：只有两三个词时，向量反而容易漂
    BM25 是经典的概率检索模型，本质是「词频 + 逆文档频率 + 长度归一化」，
    对精确词面匹配极其可靠，正好补上向量的短板。

为什么自己写而不装 rank_bm25？
    1. rank_bm25 默认用英文分词，中文要自己接分词器，收益不明显
    2. 核心公式只有十几行，自己写一遍面试能讲清每个参数的作用
    3. 少一个依赖

BM25 公式（每个查询词 t 对文档 d 的贡献求和）：

    IDF(t) = ln(1 + (N - n_t + 0.5) / (n_t + 0.5))
    score  = Σ IDF(t) · f(t,d)·(k1+1) / ( f(t,d) + k1·(1 - b + b·|d|/avgdl) )

参数含义（面试要能说）：
    k1 = 1.2~2.0  控制词频饱和速度。词出现 10 次不该比 2 次重要 5 倍，
                  所以 f/(f+k1) 会饱和。经验值 1.5
    b  = 0.75     控制长度归一化强度。长文档天然容易命中更多词，
                  要按长度打折扣。0=不归一化，1=完全归一化
"""

import math
from collections import Counter

K1 = 1.5
B = 0.75

# 单字（中文 unigram）权重打折。
# 原因：字符级分词下，「源代码」的「码」会被匹配到查询「错误码」的「码」上，
# 单字歧义太大，容易造成误召回。双字（bigram）可信度高得多。
UNIGRAM_WEIGHT = 0.3


def _term_weight(term: str) -> float:
    """双字及以上 = 1.0，中文单字 = 0.3。"""
    return UNIGRAM_WEIGHT if len(term) == 1 else 1.0


class BM25Index:
    """内存版 BM25 索引。

    为什么放内存而不是落库？
        BM25 需要全语料的 DF 和平均文档长度，这些是全局统计量，
        每次查询都从数据库重算太浪费。切片数量在几千级别时，
        内存索引构建只要几十毫秒，查询是亚毫秒级。
    """

    def __init__(self, k1: float = K1, b: float = B):
        self.k1 = k1
        self.b = b
        self.keys = []          # 与 rows 一一对应的唯一标识
        self.tfs = []           # 每个文档的词频 Counter
        self.lengths = []       # 每个文档的 token 数
        self.df = Counter()     # 每个词出现在多少个文档里
        self.avgdl = 0.0
        self.n = 0

    def fit(self, items: list, tokenizer) -> "BM25Index":
        """
        Args:
            items: [{"key": 唯一标识, "text": 原文}]
            tokenizer: 分词函数（复用 embedding.tokenize，保证和向量侧一致）
        """
        self.keys, self.tfs, self.lengths = [], [], []
        self.df = Counter()

        for it in items:
            tokens = tokenizer(it["text"])
            tf = Counter(tokens)
            self.keys.append(it["key"])
            self.tfs.append(tf)
            self.lengths.append(len(tokens))
            # DF 按「文档是否包含该词」计数，不是词频
            for term in tf:
                self.df[term] += 1

        self.n = len(self.keys)
        self.avgdl = (sum(self.lengths) / self.n) if self.n else 0.0
        return self

    def _idf(self, term: str) -> float:
        n_t = self.df.get(term, 0)
        if n_t == 0:
            return 0.0
        # 加 1 再取对数：保证 IDF 恒为正。
        # 经典 BM25 在词出现在超过半数文档时会算出负值，会倒扣分，实测有问题
        return math.log(1 + (self.n - n_t + 0.5) / (n_t + 0.5))

    def search(self, query: str, tokenizer, top_k: int = None) -> list:
        """返回 [(key, score)]，按分数降序。

        加了「强匹配」门槛：至少命中一个双字以上的词才算有效召回。
        否则单字碰撞会让完全不相关的片段也拿到分数
        （实测「E4021 错误码」曾把「源代码」的「码」匹配上）。
        """
        if not self.n:
            return []

        q_terms = Counter(tokenizer(query))
        if not q_terms:
            return []

        # 查询里有没有可用于强匹配的词（长度 >= 2）
        strong_query_terms = {t for t in q_terms if len(t) >= 2}

        scores = []
        for i, tf in enumerate(self.tfs):
            dl = self.lengths[i]
            norm = self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
            s = 0.0
            strong_hit = not strong_query_terms   # 查询里没有双字词时不做限制

            for term, q_count in q_terms.items():
                f = tf.get(term)
                if not f:
                    continue
                if len(term) >= 2:
                    strong_hit = True
                contrib = self._idf(term) * (f * (self.k1 + 1)) / (f + norm) * q_count
                s += contrib * _term_weight(term)

            if s > 0 and strong_hit:
                scores.append((self.keys[i], s))

        scores.sort(key=lambda x: -x[1])
        return scores[:top_k] if top_k else scores


# ---------------- 索引缓存 ----------------
# 每次请求都重建索引太浪费。用一个极简的版本号判断是否需要重建：
# 切片数量变化、或最大自增 id 变化，就认为语料变了。
_cache = {"sig": None, "index": None}


def get_index(rows: list, tokenizer, doc_ids: list = None) -> BM25Index:
    """按 (文档范围, 切片数, 最大 id) 缓存索引。"""
    if not rows:
        return BM25Index()

    sig = (tuple(sorted(doc_ids)) if doc_ids else None,
           len(rows),
           max(r.get("id") or 0 for r in rows))

    if _cache["sig"] == sig and _cache["index"] is not None:
        return _cache["index"]

    items = [
        {"key": chunk_key(r), "text": r["content"]}
        for r in rows
    ]
    index = BM25Index().fit(items, tokenizer)
    _cache["sig"] = sig
    _cache["index"] = index
    return index


def chunk_key(row: dict) -> tuple:
    """切片的唯一标识。用 (doc_id, chunk_index) 而不是自增 id，
    这样重建索引后 id 变了也能对得上。"""
    return (row["doc_id"], row["chunk_index"])


def invalidate() -> None:
    """上传 / 删除 / 重建索引后调用，强制下次重建。"""
    _cache["sig"] = None
    _cache["index"] = None


if __name__ == "__main__":
    import embedding

    corpus = [
        {"key": ("d1", 0), "text": "公司年假：满一年五天，满三年十天，满五年十五天"},
        {"key": ("d1", 1), "text": "公积金缴存比例为百分之十二，每年免费体检一次"},
        {"key": ("d2", 0), "text": "张三 28 岁，南昌；李四 31 岁，杭州"},
    ]
    idx = BM25Index().fit(corpus, embedding.tokenize)

    for q in ["满五年多少天", "百分之十二", "张三多少岁", "电脑坏了怎么办"]:
        print(f"查询：{q}")
        hits = idx.search(q, embedding.tokenize)
        if not hits:
            print("  （无命中）")
        for key, s in hits[:3]:
            print(f"  {s:.4f}  {key}")
        print()