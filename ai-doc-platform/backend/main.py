"""AI 文档智能处理平台 —— 后端主服务

对外接口：
    GET    /api/health              健康检查
    POST   /api/upload              上传文档 -> 解析清洗 -> 入库
    GET    /api/documents           文档列表
    GET    /api/documents/{doc_id}  文档详情（含预览文本）
    DELETE /api/documents/{doc_id}  删除文档（同时删文件）
    POST   /api/chat/stream         基于文档问答，SSE 流式返回  ★核心亮点
    GET    /api/chat/history/{doc_id}  某文档的历史问答记录
"""

import json
import os
import threading
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import OpenAI
from pydantic import BaseModel, Field

import bm25
import db
import embedding
import query_rewrite
import rerank
import vector_store
from file_utils import parse_and_clean

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# 允许上传的类型
ALLOWED_EXT = {".txt", ".csv", ".md", ".markdown", ".docx"}

app = FastAPI(title="AI 文档智能处理平台", version="0.2.0")

# ---- 跨域：前端 Vue(5173) 调后端(8000) 必须开 ----
# 注意 allow_origins=["*"] 时 allow_credentials 必须为 False，否则浏览器直接拒绝
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _get_client() -> OpenAI:
    """按需创建客户端。

    不放在模块顶层，是因为顶层只在进程启动时执行一次 ——
    用户改了 .env 里的 Key 之后必须重启整个进程才生效，容易踩坑。
    """
    return OpenAI(
        api_key=os.getenv("LLM_API_KEY", ""),
        base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
    )

SYSTEM_PROMPT = """你是一个严谨的文档问答助手。请严格遵守以下规则：
1. 只依据【文档内容】回答问题，文档里没有的信息一律回答"文档中未提及"。
2. 需要引用原文时，用「」把原文片段包起来。
3. 回答用简洁的中文，先给结论再给细节，能用列表就用列表。
4. 不要编造数字、日期、名称。如果文档信息不足以回答，直接说明缺什么。"""


@app.on_event("startup")
def on_startup() -> None:
    """启动时建表（幂等），并后台预热向量/精排模型。

    为什么要在后台预热？
        cross-encoder 首次推理要加载 ONNX 会话（几秒钟）。
        不预热的话用户第一次提问会莫名其妙卡 6 秒。
        放到后台线程里做，既不阻塞服务启动，又能让首个请求变快。
    """
    db.init_db()

    def warmup() -> None:
        try:
            embedding.embed_texts(["预热"])
            if rerank.available():
                rerank.rerank("预热", [{"content": "预热文本", "kind": "text"}], top_n=1)
        except Exception as exc:   # 预热失败不影响主流程
            print(f"[warmup] {exc}")

    threading.Thread(target=warmup, daemon=True).start()


# ---------------- 健康检查 ----------------
def _llm_ready() -> bool:
    """判断 Key 是否真的配好了 —— 只判非空会把占位符当成有效值。"""
    key = os.getenv("LLM_API_KEY", "").strip()
    return len(key) >= 20 and "你的" not in key and "在这里" not in key


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "llm": _llm_ready(),
        "message": "" if _llm_ready() else "LLM_API_KEY 未配置，上传可用但问答会失败",
    }


# ---------------- 上传 + 入库 ----------------
@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"只支持 {', '.join(sorted(ALLOWED_EXT))}，收到 {ext}")

    raw = await file.read()
    if not raw:
        raise HTTPException(400, "文件是空的")

    # 落盘：用 uuid 重命名，避免同名覆盖 + 路径穿越
    doc_id = uuid.uuid4().hex[:12]
    stored_name = f"{doc_id}{ext}"
    stored_path = UPLOAD_DIR / stored_name
    stored_path.write_bytes(raw)

    # 解析清洗（编码探测 + CSV 结构化）
    try:
        parsed = parse_and_clean(raw, ext)
    except Exception as exc:  # 解析失败也要删掉落盘的文件，不能留垃圾
        stored_path.unlink(missing_ok=True)
        raise HTTPException(400, f"文件解析失败：{exc}") from exc

    if len(parsed["text"].strip()) < 10:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(400, "解析后内容为空，可能是文件损坏或不是有效文本")

    doc = db.insert_document(
        doc_id=doc_id,
        filename=file.filename,
        stored_name=stored_name,
        size=len(raw),
        ext=ext,
        encoding=parsed["encoding"],
        text=parsed["text"],
        summary=parsed["summary"],
        rows=parsed["rows"],
        cols=parsed["cols"],
        preview=parsed["preview"],
    )

    # 建索引：切片 + 向量化。
    # 索引失败不应该让整个上传失败 —— 文档已经在库里了，
    # 没有索引还能走整篇塞入的降级路径，用户可以稍后点「重建索引」补上。
    indexed = 0
    index_error = ""
    try:
        indexed = vector_store.index_document(doc_id, parsed["text"], ext)
    except Exception as exc:
        index_error = str(exc)

    return {
        "ok": True,
        "document": doc,
        "chunks": indexed,
        "index_error": index_error,
    }


# ---------------- 文档列表 / 详情 ----------------
@app.get("/api/documents")
def list_documents():
    return {"items": db.list_documents()}


@app.get("/api/documents/{doc_id}")
def get_document(doc_id: str):
    doc = db.get_document(doc_id)
    if not doc:
        raise HTTPException(404, "文档不存在")
    return doc


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    doc = db.get_document(doc_id)
    if not doc:
        raise HTTPException(404, "文档不存在")
    (UPLOAD_DIR / doc["stored_name"]).unlink(missing_ok=True)
    db.delete_document(doc_id)
    bm25.invalidate()   # 切片被级联删除，BM25 内存索引必须失效
    return {"ok": True}


# ---------------- ★ RAG 流式文档问答（核心模块）----------------
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)

    # 兼容两种调用方式：
    #   单文档：{"doc_id": "xxx", "question": "..."}   ← 老前端不用改
    #   多文档：{"doc_ids": ["a","b"], "question": "..."}  ← 新增
    doc_id: str | None = Field(None, description="单文档 id（兼容旧接口）")
    doc_ids: list[str] | None = Field(None, description="多文档 id 列表")
    use_rag: bool = Field(True, description="True=检索片段；False=整篇塞入")
    top_k: int = Field(4, ge=1, le=10, description="召回片段数")
    mode: str | None = Field(
        None,
        description="检索模式：hybrid（默认，向量+BM25+RRF）/ vector / bm25",
    )
    use_rerank: bool | None = Field(
        None,
        description="是否用 cross-encoder 精排。None 时读 .env 的 RERANK_ENABLED",
    )
    use_rewrite: bool | None = Field(
        None,
        description="是否做查询改写 + 多查询检索。None 时读 .env 的 QUERY_REWRITE_ENABLED",
    )
    use_decompose: bool | None = Field(
        None,
        description="是否把多子问题拆开分别检索（跨文档题用）。None 时读 QUERY_DECOMPOSE_ENABLED",
    )


RAG_SYSTEM_PROMPT = """你是严谨的文档问答助手，只能依据给定的文档片段回答。

规则：
1. 只使用【片段】中出现的信息。片段里没有的，一律回答"文档中未提及"，不要凭常识补充。
2. 回答结尾用一行标注依据，格式：依据：片段1、片段3
3. 引用原文时用「」把原文包起来。
4. 用简洁中文，先给结论再给细节，能用列表就用列表。
5. 不要编造数字、日期、人名、金额。如果片段之间互相矛盾，要明确指出冲突。"""


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):
    """RAG 问答：先检索相关片段，再让模型基于片段回答。

    为什么要 RAG 而不是把整篇文档塞进去？
      - 上下文窗口有限，长文档塞不下
      - 整篇塞入时模型容易被无关内容干扰，注意力分散
      - 成本：每次问答都送全部内容，token 费用高好几倍
      - 可解释：RAG 能说清「依据哪几段」，纯塞入做不到

    流程：问题向量化 → 与所有片段算余弦 → 取 top_k → 拼上下文 → 流式生成
    """
    if not _llm_ready():
        raise HTTPException(
            503,
            "LLM_API_KEY 未配置。请打开 backend/.env 把 LLM_API_KEY 填成真实的 DeepSeek Key。",
        )

    # 统一成列表，去重保序
    resolved = list(dict.fromkeys(req.doc_ids or ([req.doc_id] if req.doc_id else [])))

    docs = []
    for did in resolved:
        d = db.get_document(did)
        if not d:
            raise HTTPException(404, f"文档不存在：{did}")
        docs.append(d)

    # ---------- 1. 检索 ----------
    sources = []
    if req.use_rag:
        try:
            sources = vector_store.search(req.question, top_k=req.top_k,
                                          doc_ids=resolved or None, mode=req.mode,
                                          use_rerank=req.use_rerank,
                                          use_rewrite=req.use_rewrite,
                                          use_decompose=req.use_decompose)
        except RuntimeError as exc:
            # 换了 embedding 模型但没重建索引
            raise HTTPException(409, str(exc)) from exc

    # ---------- 2. 拼上下文 ----------
    rewrite_info = sources[0].get("rewrite", {}) if sources else {}
    decode_info = sources[0].get("decode", {}) if sources else {}

    if sources:
        context = vector_store.build_context(sources)
        mode = "rag"
    else:
        # 没索引或一条都没命中，退回整篇塞入（保证功能不残废）
        if not docs:
            raise HTTPException(
                400,
                "没有可用的文档。请先上传文档，或等待索引构建完成。",
            )
        budget = int(os.getenv("DOC_CHAR_BUDGET", "12000"))
        parts = []
        for d in docs:
            t = d["content"] or ""
            if len(t) > budget:
                head = budget * 3 // 4
                tail = budget - head
                t = t[:head] + "\n\n...（中间内容已省略）...\n\n" + t[-tail:]
            parts.append(f"【文档：{d['filename']}】\n{t}")
        context = "\n\n".join(parts)
        mode = "fulltext"

    user_prompt = f"【文档片段】\n{context}\n\n【问题】{req.question}"
    messages = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT if mode == "rag" else SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # 多文档回答要记录到哪篇文档下：取第一个，没有就不记
    history_doc_id = resolved[0] if resolved else None

    def event_stream():
        answer = []
        started = time.time()
        try:
            # 第一帧把检索结果推给前端，让用户看到「依据了哪些片段」
            yield sse({
                "type": "meta",
                "mode": mode,
                "retrieval": req.mode or os.getenv("RETRIEVAL_MODE", "hybrid"),
                "top_k": req.top_k,
                # 查询改写结果推给前端，让用户看到系统是怎么理解问题的
                "queries": rewrite_info.get("queries", [req.question]),
                "rewritten": rewrite_info.get("rewritten", []),
                "rewrite_ms": rewrite_info.get("elapsed_ms", 0),
                "rewrite_cached": rewrite_info.get("cached", False),
                "rewrite_error": rewrite_info.get("error", ""),
                # 子问题拆解结果 —— 多子问题会被拆开分别检索
                "sub_questions": decode_info.get("subs", []),
                "sources": [
                    {
                        "filename": s["filename"],
                        "chunk_index": s["chunk_index"],
                        "score": s["score"],
                        "kind": s["kind"],
                        "matched_by": s.get("matched_by", []),
                        "ranks": s.get("ranks", {}),
                        "vector_score": s.get("vector_score"),
                        "bm25_score": s.get("bm25_score"),
                        "rerank_score": s.get("rerank_score"),
                        "rerank_rank": s.get("rerank_rank"),
                        # 这条片段是为哪个子问题召回的（跨文档题才有）
                        "matched_sub": s.get("matched_sub"),
                        "preview": s["content"][:120],
                    }
                    for s in sources
                ],
            })

            stream = _get_client().chat.completions.create(
                model=os.getenv("LLM_MODEL", "deepseek-chat"),
                messages=messages,
                stream=True,
                temperature=0.2,   # 文档问答要稳，温度调低
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    answer.append(delta)
                    yield sse({"type": "delta", "content": delta})

            full = "".join(answer)
            cost = int((time.time() - started) * 1000)
            if history_doc_id:
                db.insert_chat_message(history_doc_id, req.question, full, cost)
            yield sse({"type": "done", "elapsed_ms": cost, "mode": mode})

        except Exception as exc:
            # 流已经开始就不能再改 HTTP 状态码了，只能在流内推送错误事件
            yield sse({"type": "error", "message": str(exc)})
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Nginx 反向代理 SSE 必须关缓冲，否则前端会一次性收到全部内容而不是流式
            "X-Accel-Buffering": "no",
        },
    )


def sse(payload: dict) -> str:
    """SSE 协议要求：每行以 'data: ' 开头，消息之间用一个空行分隔"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ---------------- RAG 索引管理 ----------------
@app.post("/api/reindex")
def reindex(doc_id: str | None = None):
    """重建向量索引。

    什么时候需要：
      - 换了 EMBEDDING_PROVIDER / EMBEDDING_MODEL（维度变了）
      - 改了切片参数（CHUNK_SIZE / CHUNK_OVERLAP）
      - 老数据在加 RAG 之前上传的，还没有切片
    """
    targets = ([db.get_document(doc_id)] if doc_id else db.list_documents_for_reindex())
    targets = [t for t in targets if t]

    if not targets:
        return {"ok": True, "indexed_docs": 0, "indexed_chunks": 0}

    total, failed = 0, []
    for d in targets:
        try:
            n = vector_store.index_document(d["id"], d["content"] or "", d["ext"] or "")
            total += n
        except Exception as exc:
            failed.append({"doc_id": d["id"], "filename": d["filename"], "error": str(exc)})

    return {
        "ok": not failed,
        "indexed_docs": len(targets) - len(failed),
        "indexed_chunks": total,
        "failed": failed,
    }


@app.get("/api/rag/stats")
def rag_stats():
    """给前端展示「索引里有多少知识」，也方便你调试。"""
    stats = db.chunk_stats()
    provider = embedding.resolve_provider()
    return {
        "provider": provider,
        "model": os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5") if provider == "local" else None,
        "dim": embedding.DIM if provider == "hashing" else embedding.current_dim(),
        "numpy": embedding.HAS_NUMPY,
        "retrieval_mode": os.getenv("RETRIEVAL_MODE", "hybrid"),
        "rrf_k": vector_store.RRF_K,
        "rrf_weights": {
            "vector": vector_store.RRF_WEIGHT_VECTOR,
            "bm25": vector_store.RRF_WEIGHT_BM25,
        },
        "rerank": rerank.info(),
        "query_rewrite": query_rewrite.info(),
        "documents": stats["docs"],
        "chunks": stats["chunks"],
        "tokens": int(stats["tokens"] or 0),
    }


@app.get("/api/documents/{doc_id}/chunks")
def document_chunks(doc_id: str, limit: int = 100):
    """查看某文档的切片，用来验证切片策略是否合理。"""
    if not db.get_document(doc_id):
        raise HTTPException(404, "文档不存在")
    rows = db.load_chunk_vectors([doc_id])[:limit]
    return {
        "items": [
            {
                "chunk_index": r["chunk_index"],
                "kind": r["kind"],
                "meta": r["meta"],
                "tokens": r["tokens"],
                "content": r["content"],
            }
            for r in rows
        ]
    }


@app.post("/api/rag/debug")
def rag_debug(req: ChatRequest):
    """只看检索结果，不调大模型。调试召回质量时非常省事，也不花钱。

    传 mode 可指定只看某一路（vector / bm25 / hybrid），不传用默认配置。
    """
    try:
        results = vector_store.search(req.question, top_k=req.top_k,
                                      doc_ids=req.doc_ids or None, mode=req.mode,
                                      use_rerank=req.use_rerank,
                                      use_rewrite=req.use_rewrite,
                                      use_decompose=req.use_decompose)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    rw = results[0].get("rewrite", {}) if results else {}
    dc = results[0].get("decode", {}) if results else {}
    return {
        "question": req.question,
        "mode": req.mode or os.getenv("RETRIEVAL_MODE", "hybrid"),
        "rewritten": rw.get("rewritten", []),
        "rewrite_ms": rw.get("elapsed_ms", 0),
        "rewrite_error": rw.get("error", ""),
        "sub_questions": dc.get("subs", []),
        "count": len(results),
        "results": results,
    }


@app.post("/api/rag/compare")
def rag_compare(req: ChatRequest):
    """把几种配置的结果并排返回，用于消融对比（ablation）。

    vector / bm25 / hybrid / hybrid+rewrite / hybrid+rerank /
    hybrid+rewrite+rerank（全开）

    这个接口是给面试演示用的 —— 能直观看出每一层各自贡献了什么。
    改写那一档刻意关掉精排，隔离变量，否则看不出是谁起的作用。
    """
    plans = [
        ("vector", "vector", False, False),
        ("bm25", "bm25", False, False),
        ("hybrid", "hybrid", False, False),
        ("hybrid_rewrite", "hybrid", False, True),
        ("hybrid_rerank", "hybrid", True, False),
        ("full", "hybrid", True, True),
    ]
    out = {}
    rewritten = []
    for label, m, use_rr, use_rw in plans:
        try:
            hits = vector_store.search(req.question, top_k=req.top_k,
                                       doc_ids=req.doc_ids or None,
                                       mode=m, use_rerank=use_rr, use_rewrite=use_rw)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        if use_rw and hits:
            rewritten = hits[0].get("rewrite", {}).get("rewritten", [])
        out[label] = [
            {
                "filename": h["filename"],
                "chunk_index": h["chunk_index"],
                "score": h["score"],
                "vector_score": h["vector_score"],
                "bm25_score": h["bm25_score"],
                "rerank_score": h.get("rerank_score"),
                "matched_by": h["matched_by"],
                "query_hits": h.get("query_hits", 1),
                "preview": h["content"][:100].replace("\n", " "),
            }
            for h in hits
        ]

    def top1(key):
        return (out[key][0]["filename"], out[key][0]["chunk_index"]) if out.get(key) else None

    v1, b1 = top1("vector"), top1("bm25")
    h1, r1 = top1("hybrid"), top1("hybrid_rerank")
    return {
        "question": req.question,
        "rewritten": rewritten,
        "rerank_available": rerank.available(),
        # 两路第一名是否一致 —— 不一致说明确实互补
        "top1_same": v1 == b1 if (v1 and b1) else None,
        # 精排有没有改变融合的第一名 —— 改变了说明 rerank 起了作用
        "rerank_changed_top1": h1 != r1 if (h1 and r1) else None,
        # 改写有没有改变融合的第一名
        "rewrite_changed_top1": h1 != top1("hybrid_rewrite")
        if (h1 and top1("hybrid_rewrite")) else None,
        "results": out,
    }


# ---------------- 问答历史 ----------------
@app.get("/api/chat/history/{doc_id}")
def chat_history(doc_id: str):
    return {"items": db.list_chat_messages(doc_id)}


if __name__ == "__main__":
    import uvicorn

    # host=0.0.0.0 才能被虚拟机/局域网其他机器访问到
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)