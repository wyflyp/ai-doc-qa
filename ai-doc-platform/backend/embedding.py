"""向量化（Embedding）—— 把文本变成可以算相似度的向量。

为什么要做 provider 抽象？
    这个项目的 LLM 用的是 DeepSeek，但 **DeepSeek 不提供 embedding 接口**，
    而 HuggingFace 直连在国内不通（实测超时，只能用 hf-mirror 镜像）。
    所以这里做了一层可切换的抽象：换向量模型只改 .env，业务代码不动。

三种 provider（默认 auto）：

1. `auto`（默认）：装了 fastembed 就用本地语义模型，否则自动退回 hashing。
   同一份代码在「装得动模型」和「装不动」的机器上都能跑，不会因为缺依赖崩掉。

2. `local`：fastembed + BAAI/bge-small-zh-v1.5（onnxruntime 后端，不需要 torch）
   - 512 维中文语义向量，模型约 90MB
   - 优势：真正的语义匹配，「年龄最大的」能召回「28 岁」，纯词面匹配做不到
   - 首次运行从 hf-mirror 下载，之后走本地缓存

3. `hashing`：字符 n-gram + hashing trick（零依赖兜底）
   - 优点：不下载模型、不怕断网、确定性强
   - 缺点：只是词面匹配，同义词召不回
   - 中文用字符 bigram，词面召回能力其实不错

4. `api`：任何 OpenAI 兼容的 /v1/embeddings
   - 阿里 DashScope: base_url=https://dashscope.aliyuncs.com/compatible-mode/v1, model=text-embedding-v3
   - 硅基流动:      base_url=https://api.siliconflow.cn/v1, model=BAAI/bge-m3

面试怎么讲这个取舍：
    「DeepSeek 没有 embedding 接口，HF 直连又不通，所以我做了 provider 抽象：
     本地用 bge-small-zh 走 onnxruntime（不依赖 torch，装起来轻），
     同时保留 hashing 兜底，保证换台机器、断网也能跑。
     维度不一致会直接报错而不是静默算错——这个我在检索层做了校验。」
"""

import hashlib
import math
import os
import re
import zlib
from pathlib import Path

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:  # pragma: no cover
    HAS_NUMPY = False

PROVIDER = os.getenv("EMBEDDING_PROVIDER", "auto").strip().lower()
DIM = int(os.getenv("EMBEDDING_DIM", "8192"))

# 模型缓存放项目内，别放系统 Temp —— Temp 被清理后每次启动都要重下 90MB
MODEL_CACHE = os.getenv(
    "EMBEDDING_CACHE_DIR",
    str(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".model_cache")),
)

# HuggingFace 直连国内基本不通，默认指向镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# Windows 下普通用户没权限建符号链接，huggingface_hub 会静默失败，
# 导致模型文件下载了但快照目录是空的（我踩过这个坑）。改成复制。
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

CJK = re.compile(r"[\u4e00-\u9fff]+")
WORD = re.compile(r"[a-zA-Z0-9_]+")

# 中文高频虚词：几乎出现在每句话里，对区分度没有贡献，反而稀释权重。
# 注意只过滤「单字」，双字词（如「中国」）保留 —— 过滤会把语义切碎。
STOPWORDS = set("""
的 了 在 是 有 和 与 就 都 而 及 或 这 那 之 也 但 并 对 从 到 把 被 让 给 为 以 于 其
我 你 他 她 它 们 谁 什 么 怎 哪 些 多 少 几 吗 呢 啊 吧 呀 哦 嗯 个 上 下 里 中
会 能 要 可 该 还 又 再 很 太 更 最 第 每 各 该 此 些 且 却 则 若 因 故 由 等
the a an of to in on at is are was were be been for and or not with this that it as by from
""".split())


def tokenize(text: str) -> list:
    """中英混合分词，不依赖 jieba。

    - 中文：连续汉字串 → 单字（过滤虚词）+ 相邻双字（bigram 全保留）
      单字保证召回广度，双字近似词，比纯单字准很多
    - 英文/数字：整词小写

    关键细节：bigram 是从**原始串**上生成的，不受停用词影响。
    否则「中国」会被拆成「国」，语义就没了。
    """
    tokens = []

    for run in CJK.findall(text):
        # bigram 先做，用完整 run
        tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
        # unigram 过滤虚词
        tokens.extend(ch for ch in run if ch not in STOPWORDS)

    for w in WORD.findall(text):
        low = w.lower()
        if low in STOPWORDS:
            continue
        # 纯数字太长（如 id、金额）容易污染向量，截断处理
        tokens.append(low if len(low) <= 12 else low[:12])

    return tokens


def _hash(token: str) -> int:
    """稳定哈希。不能用内置 hash() —— Python 的字符串哈希每次启动都随机化，
    重启进程后同一个词会落到不同维度，已入库的向量全部失效。
    """
    return zlib.crc32(token.encode("utf-8")) % DIM


def _embed_hashing(text: str) -> list:
    """hashing trick + 亚线性 TF + L2 归一化。"""
    vec = [0.0] * DIM
    counts = {}
    for t in tokenize(text):
        idx = _hash(t)
        counts[idx] = counts.get(idx, 0) + 1

    for idx, c in counts.items():
        # 亚线性 TF：1 + log(tf)，避免某个词重复太多次主导向量
        vec[idx] = 1.0 + math.log(c)

    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def _embed_api(texts: list) -> list:
    """调用 OpenAI 兼容的 embeddings 接口。"""
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("EMBEDDING_API_KEY", os.getenv("LLM_API_KEY", "")),
        base_url=os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1"),
    )
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    out = []
    # 很多接口单次最多 10~32 条，分批更安全
    batch = 16
    for i in range(0, len(texts), batch):
        resp = client.embeddings.create(model=model, input=texts[i:i + batch])
        out.extend(d.embedding for d in resp.data)

    return [_l2_normalize(v) for v in out]


_local_model = None


def local_available() -> bool:
    """fastembed 是否可用（装了库 + 能加载模型）。"""
    try:
        import fastembed  # noqa: F401
        return True
    except ImportError:
        return False


def _embed_local(texts: list) -> list:
    """fastembed 本地模型（onnxruntime 后端，不依赖 torch）。

    默认 BAAI/bge-small-zh-v1.5：中文语义向量，512 维，模型约 90MB。
    """
    global _local_model
    if _local_model is None:
        from fastembed import TextEmbedding

        model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
        Path(MODEL_CACHE).mkdir(parents=True, exist_ok=True)
        _local_model = TextEmbedding(model_name=model_name, cache_dir=MODEL_CACHE)

    return [_l2_normalize(list(v)) for v in _local_model.embed(texts)]


def _l2_normalize(vec) -> list:
    vec = [float(x) for x in vec]
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm > 0 else vec


_resolved = None


def resolve_provider() -> str:
    """把 auto 解析成具体 provider。

    auto 的规则：装了 fastembed 就用本地语义模型，否则退回 hashing。
    这样同一份代码在「有模型」和「没模型」的机器上都能跑。
    """
    global _resolved
    if _resolved:
        return _resolved

    if PROVIDER == "auto":
        _resolved = "local" if local_available() else "hashing"
    else:
        _resolved = PROVIDER

    return _resolved


def embed_texts(texts: list) -> list:
    """批量向量化，统一返回 List[List[float]]（已 L2 归一化）。"""
    if not texts:
        return []

    provider = resolve_provider()

    if provider == "api":
        vecs = _embed_api(texts)
    elif provider == "local":
        try:
            vecs = _embed_local(texts)
        except Exception as exc:
            # 模型加载失败不能让整个系统挂掉，降级到 hashing
            print(f"[embedding] 本地模型不可用（{exc}），降级为 hashing 向量")
            global _resolved
            _resolved = "hashing"
            vecs = [_embed_hashing(t) for t in texts]
    else:
        vecs = [_embed_hashing(t) for t in texts]

    return vecs


def current_dim() -> int:
    """当前 provider 的输出维度。"""
    if resolve_provider() == "hashing":
        return DIM
    probe = embed_texts(["测试"])
    return len(probe[0]) if probe else DIM


def cosine(a, b) -> float:
    """余弦相似度。两者都已 L2 归一化，所以点积就是余弦值。"""
    if HAS_NUMPY:
        return float(np.dot(a, b))
    return sum(x * y for x, y in zip(a, b))


def similarity_scores(query_vec, matrix) -> list:
    """一次算一个 query 对多个 chunk 的相似度。

    matrix: List[List[float]]
    返回 List[float]
    """
    if not matrix:
        return []
    if HAS_NUMPY:
        q = np.asarray(query_vec, dtype="float32")
        m = np.asarray(matrix, dtype="float32")
        return (m @ q).tolist()
    return [sum(x * y for x, y in zip(query_vec, row)) for row in matrix]


def to_blob(vec) -> bytes:
    """向量序列化成 float32 二进制，比 JSON 省 3~4 倍空间，读写也快。"""
    if HAS_NUMPY:
        return np.asarray(vec, dtype="float32").tobytes()
    import struct

    return struct.pack(f"<{len(vec)}f", *vec)


def from_blob(blob: bytes) -> list:
    if not blob:
        return []
    if HAS_NUMPY:
        return np.frombuffer(blob, dtype="float32").tolist()
    import struct

    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


if __name__ == "__main__":
    q = "表格里谁的年龄最大"
    docs = [
        "张三 28 岁，在南昌工作",
        "李四 31 岁，在杭州工作",
        "这份文档讲的是项目部署流程，用 Nginx 反向代理",
    ]
    qv = embed_texts([q])[0]
    dv = embed_texts(docs)
    scores = similarity_scores(qv, dv)
    print(f"provider={PROVIDER} dim={len(qv)} numpy={HAS_NUMPY}")
    for d, s in sorted(zip(docs, scores), key=lambda x: -x[1]):
        print(f"  {s:.4f}  {d}")