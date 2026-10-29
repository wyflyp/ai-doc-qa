"""环境自检脚本。双击 3-检查环境.bat 运行，或者：
    .venv\\Scripts\\python.exe check.py

会检查 4 件事，任何一项失败都会告诉你具体怎么修。
"""

import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent


def line(title: str) -> None:
    print(f"\n{'=' * 50}\n{title}\n{'=' * 50}")


def main() -> int:
    problems = []

    # ---------- 1. .env ----------
    line("1/4  配置文件 .env")
    env_file = BASE / ".env"
    if not env_file.exists():
        print("[X] .env 不存在")
        print("    修复：复制 .env.example 改名为 .env，然后填入 DB_PASSWORD 和 LLM_API_KEY")
        return 1

    from dotenv import load_dotenv

    load_dotenv(env_file, override=True)

    db_pwd = os.getenv("DB_PASSWORD", "")
    llm_key = os.getenv("LLM_API_KEY", "")
    db_name = os.getenv("DB_NAME", "ai_doc_platform")

    print(f"[OK] .env 已找到")
    print(f"     DB_USER     = {os.getenv('DB_USER')}")
    print(f"     DB_NAME     = {db_name}")
    print(f"     DB_PASSWORD = {'已填写' if db_pwd and '你的' not in db_pwd else '【未填写】'}")
    print(f"     LLM_API_KEY = {'已填写' if llm_key and '你的' not in llm_key and 'sk-在' not in llm_key else '【未填写】'}")
    print(f"     LLM_MODEL   = {os.getenv('LLM_MODEL')}")

    if not db_pwd or "你的" in db_pwd:
        problems.append("DB_PASSWORD 还没填")
    if not llm_key or "你的" in llm_key or "sk-在" in llm_key:
        problems.append("LLM_API_KEY 还没填（能上传文档，但问答会报错）")

    # ---------- 2. MySQL 连接 ----------
    line("2/4  MySQL 连接")
    try:
        import pymysql

        conn = pymysql.connect(
            host=os.getenv("DB_HOST", "127.0.0.1"),
            port=int(os.getenv("DB_PORT", "3306")),
            user=os.getenv("DB_USER", "root"),
            password=db_pwd,
            charset="utf8mb4",
        )
        print("[OK] MySQL 连接成功")

        with conn.cursor() as cur:
            cur.execute("SHOW DATABASES LIKE %s", (db_name,))
            if cur.fetchone():
                print(f"[OK] 数据库 {db_name} 已存在")
            else:
                print(f"[..] 数据库 {db_name} 不存在，正在自动创建 ...")
                cur.execute(
                    f"CREATE DATABASE `{db_name}` "
                    "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                )
                print(f"[OK] 数据库 {db_name} 创建完成")
        conn.close()
    except Exception as exc:
        print(f"[X] MySQL 连接失败：{str(exc)[:120]}")
        print("    修复：检查 .env 里的 DB_PASSWORD 是否正确")
        print("    忘记密码？看 init_db.sql 文件末尾的重置方法")
        problems.append("MySQL 连不上")

    # ---------- 3. 建表 ----------
    line("3/4  数据表")
    if not problems or "MySQL 连不上" not in problems:
        try:
            import db

            db.init_db()
            print("[OK] 数据表已就绪（document / chat_message）")

            conn = db.get_conn()
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) AS c FROM document")
                docs = cur.fetchone()["c"]
                cur.execute("SELECT COUNT(*) AS c FROM chat_message")
                chats = cur.fetchone()["c"]
            conn.close()
            print(f"     当前已有 {docs} 个文档、{chats} 条问答记录")
        except Exception as exc:
            print(f"[X] 建表失败：{str(exc)[:120]}")
            problems.append("建表失败")

    # ---------- 4. 依赖 ----------
    line("4/5  Python 依赖")
    mods = {
        "fastapi": "FastAPI",
        "uvicorn": "Uvicorn",
        "pymysql": "PyMySQL",
        "openai": "OpenAI SDK",
        "dotenv": "python-dotenv",
        "multipart": "python-multipart",
    }
    missing = []
    for mod, name in mods.items():
        try:
            __import__(mod)
            print(f"[OK] {name}")
        except ImportError:
            print(f"[X] {name} 缺失")
            missing.append(name)
    if missing:
        problems.append("依赖缺失：" + "、".join(missing))

    # ---------- 5. RAG 索引 ----------
    line("5/5  RAG 向量索引")
    try:
        import embedding

        provider = embedding.resolve_provider()
        if provider == "hashing":
            print("[!] 使用 hashing 词面向量（fastembed 未安装或加载失败）")
            print("    想要语义检索：pip install fastembed -i "
                  "https://mirrors.aliyun.com/pypi/simple/")
        else:
            model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
            print(f"[OK] 向量模型：{model}（{provider}）")
            print(f"     维度：{embedding.current_dim()}  numpy 加速：{embedding.HAS_NUMPY}")

        stats = db.chunk_stats()
        print(f"[OK] 已索引 {stats['docs']} 个文档，共 {stats['chunks']} 个切片")

        if stats["docs"] and not stats["chunks"]:
            problems.append("有文档但没有切片，请点界面的「重建索引」")

        # 维度一致性检查：换了模型没重建索引，检索时会直接报错
        rows = db.load_chunk_vectors()
        if rows:
            dims = {r["dim"] for r in rows}
            cur = embedding.current_dim()
            if dims != {cur}:
                print(f"[X] 索引维度 {dims} 与当前模型 {cur} 不一致")
                problems.append("向量维度不一致，需要重建索引（POST /api/reindex）")
            else:
                print(f"[OK] 向量维度一致（{cur}）")

        import vector_store

        mode = os.getenv("RETRIEVAL_MODE", "hybrid")
        print(f"[OK] 检索模式：{mode}")
        if mode == "hybrid":
            print(f"     RRF 权重 vector={vector_store.RRF_WEIGHT_VECTOR} "
                  f"bm25={vector_store.RRF_WEIGHT_BM25} k={vector_store.RRF_K}")

        import rerank

        rinfo = rerank.info()
        if rinfo["available"]:
            print(f"[OK] 精排模型：{rinfo['model']}（默认开启，57 条题上 Hit@1 +8）")
        else:
            print("[!] 精排未安装（不影响问答，会自动跳过）")

        import query_rewrite

        qinfo = query_rewrite.info()
        state = "开启" if qinfo["enabled"] else "关闭（默认，叠加精排后无增益）"
        print(f"[OK] 查询改写：{state}")

        print("[提示] 默认链路 = 混合检索 + 精排 + 子问题分解。")
        print("       实测：本地 CPU 上检索约 9 秒（首次加载模型约 15 秒），关掉精排降到 0.2 秒。")
        print("       关精排的代价：57 条题上 Hit@1 从 44/49 掉到 36/49。")
    except Exception as exc:
        print(f"[X] RAG 检查失败：{str(exc)[:120]}")
        problems.append("RAG 索引异常")

    # ---------- 总结 ----------
    line("总结")
    if not problems:
        print("全部通过！现在双击 1-启动后端.bat，再双击 2-启动前端.bat")
        return 0

    print("还有以下问题需要处理：")
    for i, p in enumerate(problems, 1):
        print(f"  {i}. {p}")
    return 1


if __name__ == "__main__":
    sys.exit(main())