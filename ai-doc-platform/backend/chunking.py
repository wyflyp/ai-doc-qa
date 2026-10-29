"""文档切片（Chunking）—— RAG 里最影响效果的一环。

设计原则：
    1. 切片要「自包含」：一个片段单独拿出来也能读懂
    2. 表格切片必须带表头，否则片段脱离列名就失去意义
    3. 相邻切片要有重叠（overlap），避免答案正好被切断在边界上
    4. 按字符切而不是按 token —— 中文 1 字约 1 token，估算够用，
       而且不引入 tokenizer 依赖

常见错误做法（会被面试官追问）：
    - 固定长度硬切：把一个句子、一行表格切成两半，检索出来读不懂
    - 切片过大：召回不精准，白烧 token
    - 切片过小：丢失上下文，模型答不全
    - 无重叠：关键信息跨边界时两边都召回不到
"""

import re

# 默认参数：字符数
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 80
CSV_ROWS_PER_CHUNK = 20


def _split_long_text(text: str, size: int) -> list:
    """超长段落按句子边界切成不超过 size 的块。

    优先在句号/问号/感叹号/分号处断开，实在找不到就硬切。
    """
    sentences = re.split(r"(?<=[。！？；!?;])|(?<=\n)", text)
    units, cur = [], ""
    for s in sentences:
        if not s:
            continue
        if len(cur) + len(s) <= size:
            cur += s
        else:
            if cur:
                units.append(cur)
            # 单句本身超长，硬切
            while len(s) > size:
                units.append(s[:size])
                s = s[size:]
            cur = s
    if cur:
        units.append(cur)
    return [u.strip() for u in units if u.strip()]


def _is_markdown_table(text: str) -> bool:
    """判断是否为 Markdown 表格（CSV 解析后会转成这种格式）。"""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    return lines[0].startswith("|") and re.match(r"^\|[\s:\-|]+\|$", lines[1].strip()) is not None


def chunk_table(text: str, rows_per_chunk: int = CSV_ROWS_PER_CHUNK) -> list:
    """表格切片：每 N 行一个 chunk，**每个 chunk 都重复表头**。

    这是表格类 RAG 最关键的一点：表头就是列的语义。
    不带表头的片段，模型看到「| 张三 | 28 | 南昌 |」根本不知道 28 是年龄还是工号。
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 3:
        return [{"index": 0, "content": text, "kind": "table"}]

    header, separator, body = lines[0], lines[1], lines[2:]
    chunks = []
    for start in range(0, len(body), rows_per_chunk):
        block = body[start:start + rows_per_chunk]
        content = "\n".join([header, separator] + block)
        chunks.append({
            "index": len(chunks),
            "content": content,
            "kind": "table",
            "meta": f"第 {start + 1}-{start + len(block)} 行",
        })
    return chunks


def chunk_text(text: str, size: int = DEFAULT_CHUNK_SIZE,
               overlap: int = DEFAULT_OVERLAP) -> list:
    """文本切片：按段落聚合到 size，相邻块保留 overlap 字符重叠。"""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return []

    units = []
    for p in paragraphs:
        if len(p) <= size:
            units.append(p)
        else:
            units.extend(_split_long_text(p, size))

    chunks, cur = [], ""
    for u in units:
        if not cur:
            cur = u
        elif len(cur) + len(u) + 1 <= size:
            cur = cur + "\n" + u
        else:
            chunks.append(cur)
            # 重叠：把上一块尾部 overlap 个字符拼到新块开头
            tail = cur[-overlap:] if overlap else ""
            cur = (tail + "\n" + u) if tail else u
            # 防止重叠后超长
            if len(cur) > size + overlap:
                cur = cur[-size:]

    if cur:
        chunks.append(cur)

    return [
        {"index": i, "content": c, "kind": "text"}
        for i, c in enumerate(chunks)
        if c.strip()
    ]


def chunk_document(text: str, ext: str = "",
                   size: int = DEFAULT_CHUNK_SIZE,
                   overlap: int = DEFAULT_OVERLAP) -> list:
    """统一入口：自动判断表格还是文本。

    Returns:
        [{"index": int, "content": str, "kind": "table"|"text", "meta": str}]
    """
    if not text or not text.strip():
        return []

    if ext == ".csv" and _is_markdown_table(text):
        return chunk_table(text)

    # 非 CSV 但内容恰好是 Markdown 表格（比如 .md 文件），也走表格逻辑
    if _is_markdown_table(text) and len(text.splitlines()) > 8:
        return chunk_table(text)

    return chunk_text(text, size, overlap)


if __name__ == "__main__":
    # 自测
    demo = """第一段内容，讲的是项目背景。这个项目要做文档问答。

第二段内容，讲技术选型。用了 Vue3 和 FastAPI。

第三段内容，说明为什么要用 SSE。因为需要流式输出。
"""
    for c in chunk_document(demo, ".txt", size=60, overlap=15):
        print(f"[{c['index']}] ({c['kind']})")
        print(repr(c["content"]))
        print()

    table = """| 姓名 | 年龄 | 城市 |
| --- | --- | --- |
| 张三 | 28 | 南昌 |
| 李四 | 31 | 杭州 |
| 王五 | 25 | 上海 |"""
    print("=== 表格切片 ===")
    for c in chunk_table(table, rows_per_chunk=2):
        print(f"[{c['index']}] {c.get('meta')}")
        print(c["content"])
        print()