"""MySQL 数据访问层。

用 PyMySQL + 连接池，而不是裸 MySQLdb：
  - PyMySQL 是纯 Python 实现，pip 装完即用，不需要编译
  - 密码不写死在代码里，统一从 .env 读
"""

import os

import pymysql
from dotenv import load_dotenv
from pymysql.cursors import DictCursor

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "127.0.0.1"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "ai_doc_platform"),
    "charset": "utf8mb4",
    "cursorclass": DictCursor,
    "autocommit": True,
}


def get_conn():
    """新建一个数据库连接。用完必须关闭（或用 with 自动关）。"""
    return pymysql.connect(**DB_CONFIG)


def init_db() -> None:
    """建表。CREATE TABLE IF NOT EXISTS 保证重复执行不报错。"""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document (
                    id           VARCHAR(32)  NOT NULL PRIMARY KEY COMMENT '文档唯一 id',
                    filename     VARCHAR(255) NOT NULL COMMENT '原始文件名',
                    stored_name  VARCHAR(255) NOT NULL COMMENT '磁盘上的存储名',
                    ext          VARCHAR(16)  NOT NULL COMMENT '扩展名',
                    size         INT          NOT NULL DEFAULT 0 COMMENT '字节数',
                    encoding     VARCHAR(32)  NOT NULL DEFAULT '' COMMENT '探测出的编码',
                    summary      VARCHAR(500) NOT NULL DEFAULT '' COMMENT '结构摘要',
                    rows_count   INT          NOT NULL DEFAULT 0 COMMENT 'CSV 行数',
                    cols_count   INT          NOT NULL DEFAULT 0 COMMENT 'CSV 列数',
                    content      MEDIUMTEXT   NOT NULL COMMENT '清洗后的全文',
                    preview      TEXT                  COMMENT '前 2000 字符预览',
                    created_at   DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='上传的文档'
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_message (
                    id          BIGINT       NOT NULL AUTO_INCREMENT PRIMARY KEY,
                    doc_id      VARCHAR(32)  NOT NULL COMMENT '关联文档 id',
                    question    TEXT         NOT NULL COMMENT '用户提问',
                    answer      MEDIUMTEXT   NOT NULL COMMENT '模型回答全文',
                    elapsed_ms  INT          NOT NULL DEFAULT 0 COMMENT '耗时毫秒',
                    created_at  DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_doc_id (doc_id),
                    CONSTRAINT fk_chat_doc FOREIGN KEY (doc_id)
                        REFERENCES document(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='文档问答记录'
            """)
            # RAG 切片表：一行 = 一个切片 + 它的向量
            # vec 存 float32 二进制而不是 JSON，省 3~4 倍空间，读写也快
            cur.execute("""
                CREATE TABLE IF NOT EXISTS doc_chunk (
                    id           BIGINT       NOT NULL AUTO_INCREMENT PRIMARY KEY,
                    doc_id       VARCHAR(32)  NOT NULL COMMENT '所属文档',
                    chunk_index  INT          NOT NULL DEFAULT 0 COMMENT '在文档内的序号',
                    content      MEDIUMTEXT   NOT NULL COMMENT '片段文本',
                    kind         VARCHAR(16)  NOT NULL DEFAULT 'text' COMMENT 'text/table',
                    meta         VARCHAR(255) NOT NULL DEFAULT '' COMMENT '如第几行',
                    tokens       INT          NOT NULL DEFAULT 0 COMMENT '分词数量',
                    vec          LONGBLOB     NOT NULL COMMENT 'float32 向量',
                    dim          INT          NOT NULL DEFAULT 0 COMMENT '向量维度',
                    created_at   DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_doc_id (doc_id),
                    INDEX idx_doc_chunk (doc_id, chunk_index),
                    CONSTRAINT fk_chunk_doc FOREIGN KEY (doc_id)
                        REFERENCES document(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='RAG 文档切片与向量'
            """)
    finally:
        conn.close()


# ---------------- document 表 ----------------

def insert_document(doc_id: str, filename: str, stored_name: str, size: int,
                    ext: str, encoding: str, text: str, summary: str,
                    rows: int, cols: int, preview: str) -> dict:
    sql = """
        INSERT INTO document
            (id, filename, stored_name, ext, size, encoding, summary,
             rows_count, cols_count, content, preview)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (doc_id, filename, stored_name, ext, size, encoding,
                              summary, rows, cols, text, preview))
    finally:
        conn.close()
    return get_document(doc_id)


def list_documents(limit: int = 100) -> list:
    """列表不返回 content 大字段，节省内存和网络。"""
    sql = """
        SELECT id, filename, ext, size, encoding, summary,
               rows_count, cols_count, created_at
        FROM document ORDER BY created_at DESC LIMIT %s
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (limit,))
            return cur.fetchall()
    finally:
        conn.close()


def get_document(doc_id: str):
    sql = """
        SELECT id, filename, stored_name, ext, size, encoding, summary,
               rows_count, cols_count, content, preview, created_at
        FROM document WHERE id = %s
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (doc_id,))
            return cur.fetchone()
    finally:
        conn.close()


def delete_document(doc_id: str) -> None:
    """聊天记录靠 ON DELETE CASCADE 自动级联删除，不用手动清。"""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM document WHERE id = %s", (doc_id,))
    finally:
        conn.close()


# ---------------- chat_message 表 ----------------

def insert_chat_message(doc_id: str, question: str, answer: str,
                        elapsed_ms: int = 0) -> None:
    sql = """
        INSERT INTO chat_message (doc_id, question, answer, elapsed_ms)
        VALUES (%s, %s, %s, %s)
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (doc_id, question, answer, elapsed_ms))
    finally:
        conn.close()


def list_chat_messages(doc_id: str, limit: int = 50) -> list:
    sql = """
        SELECT id, question, answer, elapsed_ms, created_at
        FROM chat_message WHERE doc_id = %s
        ORDER BY id DESC LIMIT %s
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (doc_id, limit))
            return cur.fetchall()
    finally:
        conn.close()


# ---------------- doc_chunk 表（RAG）----------------

def insert_chunks(rows: list) -> None:
    """批量写入切片。用 executemany，几千条也就一次往返。"""
    if not rows:
        return
    sql = """
        INSERT INTO doc_chunk
            (doc_id, chunk_index, content, kind, meta, tokens, vec, dim)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    params = [
        (r["doc_id"], r["chunk_index"], r["content"], r["kind"],
         r["meta"], r["tokens"], r["vec"], r["dim"])
        for r in rows
    ]
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.executemany(sql, params)
    finally:
        conn.close()


def delete_chunks(doc_id: str) -> None:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM doc_chunk WHERE doc_id = %s", (doc_id,))
    finally:
        conn.close()


def load_chunk_vectors(doc_ids: list = None) -> list:
    """把切片和向量一次性读进内存，交给 numpy 批量算余弦。

    doc_ids 为 None 时检索全部文档（多文档 RAG）。

    注意：这里刻意不带 content 之外的冗余字段，也不要在这里做过滤，
    因为过滤逻辑放在检索层更灵活。
    """
    sql = """
        SELECT c.id, c.doc_id, c.chunk_index, c.content, c.kind, c.meta,
               c.tokens, c.vec, c.dim, d.filename
        FROM doc_chunk c
        JOIN document d ON d.id = c.doc_id
    """
    params = ()
    if doc_ids:
        placeholders = ",".join(["%s"] * len(doc_ids))
        sql += f" WHERE c.doc_id IN ({placeholders})"
        params = tuple(doc_ids)
    sql += " ORDER BY c.doc_id, c.chunk_index"

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()
    finally:
        conn.close()


def chunk_stats() -> dict:
    """切片统计，用于前端展示「索引了多少知识」。"""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) AS chunks,
                       COUNT(DISTINCT doc_id) AS docs,
                       COALESCE(SUM(tokens), 0) AS tokens
                FROM doc_chunk
            """)
            return cur.fetchone()
    finally:
        conn.close()


def list_documents_for_reindex() -> list:
    """重建索引用：需要全文，所以不能复用 list_documents（它省掉了 content）。"""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, filename, ext, content FROM document")
            return cur.fetchall()
    finally:
        conn.close()