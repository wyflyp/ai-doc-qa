"""批量导入真实文档，用于在真实语料上评估检索效果。

为什么需要它？
    前面所有评估都建立在我自己造的测试文档上（总共 25 个切片），
    候选深度 20 几乎全覆盖，导致任何检索优化都测不出差别。
    只有导入真实文档、让切片数到几百个，评估才有意义。

安全设计（重要）：
    批量导入最大的风险是把不该进库的文件也导进去。
    这里做了三层防护：
      1. 扩展名白名单：只认 .docx / .md / .txt / .csv
      2. 文件名黑名单：含 key / 密码 / 简历 / 名片 / 身份证 等直接跳过
      3. 临时文件过滤：以 ~$ 开头的是 Word 打开时生成的锁文件，内容是空的
    另外跳过小于 200 字节的文件（基本是空壳）。

用法：
    python import_corpus.py <目录或文件> [更多路径...]
    python import_corpus.py "C:/Users/xxx/OneDrive/Desktop/vue实验"
    python import_corpus.py --dry-run "路径"     # 只扫描不导入
"""

import sys
import uuid
from pathlib import Path

import db
import vector_store
from file_utils import parse_and_clean

ALLOWED = {".docx", ".md", ".txt", ".csv", ".markdown"}

# 命中任一关键词就跳过 —— 这些是凭证或隐私文件，绝不能进知识库
BLOCK_WORDS = [
    "api-key", "apikey", "api_key", "key.txt", "secret", "password", "密码",
    "简历", "名片", "身份证", "身份证号", "银行卡", "工资条",
]

MIN_BYTES = 200

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"


def collect(paths: list, exclude_keywords: list = None) -> tuple:
    """收集候选文件，返回 (待导入, 被跳过及原因)。"""
    block = BLOCK_WORDS + (exclude_keywords or [])
    todo, skipped = [], []

    for raw in paths:
        p = Path(raw)
        files = [p] if p.is_file() else sorted(p.rglob("*"))

        for f in files:
            if not f.is_file():
                continue
            name, ext = f.name, f.suffix.lower()
            low = name.lower()

            if ext not in ALLOWED:
                if ext in (".doc", ".pdf"):
                    skipped.append((str(f), f"{ext} 格式不支持（需转 docx/txt）"))
                continue
            if name.startswith("~$"):
                skipped.append((str(f), "Word 临时锁文件"))
                continue
            if any(w in low for w in block):
                skipped.append((str(f), "命中敏感词黑名单"))
                continue
            try:
                size = f.stat().st_size
            except OSError:
                skipped.append((str(f), "读取失败"))
                continue
            if size < MIN_BYTES:
                skipped.append((str(f), f"文件过小（{size} 字节）"))
                continue

            todo.append(f)

    return todo, skipped


def import_files(files: list, skip_existing: bool = True) -> dict:
    db.init_db()
    UPLOAD_DIR.mkdir(exist_ok=True)

    existing = {d["filename"] for d in db.list_documents_for_reindex()}

    imported, failed, skipped_dup = [], [], []
    total_chunks = 0

    for f in files:
        if skip_existing and f.name in existing:
            skipped_dup.append(f.name)
            continue

        try:
            raw = f.read_bytes()
            parsed = parse_and_clean(raw, f.suffix.lower())
        except Exception as exc:
            failed.append((f.name, str(exc)[:100]))
            continue

        if len(parsed["text"].strip()) < 50:
            failed.append((f.name, "提取出的文本过少（可能是纯图片文档）"))
            continue

        doc_id = uuid.uuid4().hex[:12]
        stored_name = f"{doc_id}{f.suffix.lower()}"
        (UPLOAD_DIR / stored_name).write_bytes(raw)

        try:
            db.insert_document(
                doc_id=doc_id, filename=f.name, stored_name=stored_name,
                size=len(raw), ext=f.suffix.lower(), encoding=parsed["encoding"],
                text=parsed["text"], summary=parsed["summary"],
                rows=parsed["rows"], cols=parsed["cols"], preview=parsed["preview"],
            )
            n = vector_store.index_document(doc_id, parsed["text"], f.suffix.lower())
        except Exception as exc:
            failed.append((f.name, str(exc)[:100]))
            continue

        total_chunks += n
        imported.append((f.name, n, len(parsed["text"])))

    return {
        "imported": imported,
        "failed": failed,
        "skipped_dup": skipped_dup,
        "total_chunks": total_chunks,
    }


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry-run" in sys.argv

    if not args:
        print(__doc__)
        return 1

    todo, skipped = collect(args)

    print(f"扫描到 {len(todo)} 个可导入文件，跳过 {len(skipped)} 个\n")

    if skipped:
        print("--- 被跳过的文件 ---")
        for path, why in skipped:
            short = Path(path).name
            print(f"  {short[:46]:48} {why}")
        print()

    if dry:
        print("--- 待导入 ---")
        for f in todo:
            print(f"  {f.name}  ({f.stat().st_size // 1024}KB)")
        return 0

    print("--- 开始导入 ---")
    result = import_files(todo)

    for name, chunks, chars in result["imported"]:
        print(f"  ✅ {name[:46]:48} {chunks:>3} 切片  {chars:>6} 字")
    for name, why in result["failed"]:
        print(f"  ❌ {name[:46]:48} {why}")
    if result["skipped_dup"]:
        print(f"  ⏭  已存在跳过 {len(result['skipped_dup'])} 个")

    print(f"\n导入 {len(result['imported'])} 份，共新增 {result['total_chunks']} 个切片")
    stats = db.chunk_stats()
    print(f"当前知识库：{stats['docs']} 个文档、{stats['chunks']} 个切片、{int(stats['tokens'])} 个词元")
    return 0


if __name__ == "__main__":
    sys.exit(main())