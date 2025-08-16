"""
Basic indexer for the latest Godot documentation (tutorials + class reference).

It downloads the official docs from GitHub, parses and chunks them, generates
embeddings using OpenAI, and writes rows to the existing BigQuery vector table
used by the backend (schema-compatible with CloudVectorManager).

Corpus identifiers (can be overridden via env/CLI):
  user_id = "public_docs"
  project_id = "godot_docs_latest"

Usage (local):
  python -m backend.scripts.index_godot_docs --gcp-project $GCP_PROJECT_ID --openai-key $OPENAI_API_KEY

Notes:
  - Minimal dependencies (google-cloud-bigquery, openai) are already in requirements.txt
  - This script avoids modifying backend code; it writes directly to the same BigQuery table
  - To refresh, run again with --force
"""

import argparse
import json
import concurrent.futures
import datetime as dt
import hashlib
import io
import os
import re
import sys
import tarfile
import time
import zipfile
from typing import Dict, Iterable, List, Tuple

import requests
from google.cloud import bigquery
from google.cloud.exceptions import NotFound

try:
    import openai  # openai>=1.50.0
except Exception:  # pragma: no cover
    openai = None


# Defaults
DEFAULT_DATASET = os.getenv("EMBED_DATASET", "godot_embeddings")
DEFAULT_TABLE = os.getenv("EMBED_TABLE", "embeddings")
DEFAULT_USER_ID = os.getenv("DOCS_USER_ID", "public_docs")
DEFAULT_PROJECT_ID = os.getenv("DOCS_PROJECT_ID", "godot_docs_latest")
DEFAULT_BRANCH = os.getenv("DOCS_BRANCH", "latest")  # godot-docs branch
DEFAULT_ENGINE_BRANCH = os.getenv("GODOT_ENGINE_BRANCH", "master")

EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "128"))
EMBED_MAX_PARALLEL = int(os.getenv("EMBED_MAX_PARALLEL", "8"))
CHUNK_MAX_CHARS = int(os.getenv("DOCS_CHUNK_MAX_CHARS", "2000"))
CHUNK_OVERLAP_CHARS = int(os.getenv("DOCS_CHUNK_OVERLAP", "200"))


def _d(root: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", root))


def ensure_dataset_and_table(bq: bigquery.Client, gcp_project: str, dataset_id: str, table_id: str) -> None:
    dataset_ref = f"{gcp_project}.{dataset_id}"
    try:
        bq.get_dataset(dataset_ref)
    except NotFound:
        ds = bigquery.Dataset(dataset_ref)
        ds.location = "US"
        bq.create_dataset(ds)
        print(f"Created dataset {dataset_ref}")

    table_ref = f"{dataset_ref}.{table_id}"
    schema = [
        bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("user_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("project_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("file_path", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("chunk_index", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("content", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("start_line", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("end_line", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("file_hash", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("indexed_at", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
    ]
    try:
        bq.get_table(table_ref)
    except NotFound:
        table = bigquery.Table(table_ref, schema=schema)
        bq.create_table(table)
        print(f"Created table {table_ref}")


def download_zip(url: str, dest_dir: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    print(f"Downloading: {url}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    zpath = os.path.join(dest_dir, "archive.zip")
    with open(zpath, "wb") as f:
        f.write(r.content)
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(dest_dir)
    os.remove(zpath)
    # Return top folder
    names = [n for n in os.listdir(dest_dir) if os.path.isdir(os.path.join(dest_dir, n))]
    if not names:
        return dest_dir
    return os.path.join(dest_dir, names[0])


def download_github_zip_with_fallback(owner_repo: str, branch_candidates: List[str], dest_dir: str) -> str:
    """Try multiple branches for GitHub zip archives until one succeeds.

    owner_repo: e.g., 'godotengine/godot-docs'
    branch_candidates: e.g., ['master', 'latest', 'stable', '4.3', '4.2']
    """
    last_err = None
    for br in branch_candidates:
        try:
            url = f"https://github.com/{owner_repo}/archive/refs/heads/{br}.zip"
            return download_zip(url, os.path.join(dest_dir, f"{owner_repo.replace('/', '_')}-{br}"))
        except requests.HTTPError as e:
            last_err = e
            print(f"Download failed for branch '{br}', trying next… ({e})")
        except Exception as e:
            last_err = e
            print(f"Download error on branch '{br}': {e}")
    if last_err:
        raise last_err
    raise RuntimeError("No branches attempted")


def download_tar_gz(url: str, dest_dir: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    print(f"Downloading: {url}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    tpath = os.path.join(dest_dir, "archive.tar.gz")
    with open(tpath, "wb") as f:
        f.write(r.content)
    with tarfile.open(tpath, "r:gz") as tf:
        tf.extractall(dest_dir)
    os.remove(tpath)
    names = [n for n in os.listdir(dest_dir) if os.path.isdir(os.path.join(dest_dir, n))]
    if not names:
        return dest_dir
    return os.path.join(dest_dir, names[0])


def sanitize_text(s: str) -> str:
    # Strip control chars except whitespace
    return "".join(ch for ch in s if ord(ch) >= 32 or ch in "\n\r\t")


def rst_to_plaintext(rst_text: str) -> str:
    # Minimal, dependency-free conversion: drop directive lines and keep headings/code blocks.
    out_lines: List[str] = []
    for line in rst_text.splitlines():
        if line.strip().startswith(".. "):
            continue
        out_lines.append(line)
    return sanitize_text("\n".join(out_lines))


def chunk_by_chars(text: str, max_chars: int, overlap: int) -> List[Tuple[str, int, int]]:
    chunks: List[Tuple[str, int, int]] = []
    if not text:
        return chunks
    start = 0
    end = 0
    n = len(text)
    while start < n:
        end = min(n, start + max_chars)
        chunk = text[start:end]
        chunks.append((chunk, start, end))
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks


def enumerate_docs_files(docs_root: str) -> Iterable[Tuple[str, str]]:
    # Return (relative_key, absolute_path) pairs for .rst files
    for root, _dirs, files in os.walk(docs_root):
        for fn in files:
            if fn.endswith('.rst'):
                ap = os.path.join(root, fn)
                rel = os.path.relpath(ap, docs_root)
                yield (f"rst:{rel}", ap)


def enumerate_class_xml(engine_root: str) -> Iterable[Tuple[str, str]]:
    classes_dir = os.path.join(engine_root, 'doc', 'classes')
    if not os.path.isdir(classes_dir):
        return []
    for fn in os.listdir(classes_dir):
        if fn.endswith('.xml'):
            ap = os.path.join(classes_dir, fn)
            yield (f"xml:{fn}", ap)


def parse_xml_chunks(abs_path: str) -> List[str]:
    import xml.etree.ElementTree as ET
    try:
        tree = ET.parse(abs_path)
        root = tree.getroot()
        class_name = root.attrib.get('name') or os.path.splitext(os.path.basename(abs_path))[0]
        texts: List[str] = []
        # Class descriptions
        for tag in ('brief_description', 'description'):
            e = root.find(tag)
            if e is not None:
                t = ''.join(e.itertext()).strip()
                if t:
                    texts.append(f"Class: {class_name}\n\n{t}")
        # Methods
        for m in root.findall('methods/method'):
            mname = m.attrib.get('name', '')
            desc = ''.join((m.findtext('description') or '').splitlines(True)).strip()
            parts = [f"Method: {class_name}.{mname}"]
            if desc:
                parts.append(desc)
            texts.append("\n\n".join(parts))
        # Signals
        for s in root.findall('signals/signal'):
            sname = s.attrib.get('name', '')
            desc = ''.join((s.findtext('description') or '').splitlines(True)).strip()
            parts = [f"Signal: {class_name}.{sname}"]
            if desc:
                parts.append(desc)
            texts.append("\n\n".join(parts))
        # Properties
        for p in root.findall('members/member'):
            pname = p.attrib.get('name', '')
            ptype = p.attrib.get('type', '')
            desc = ''.join((p.findtext('description') or '').splitlines(True)).strip()
            texts.append(f"Property: {class_name}.{pname}: {ptype}\n\n{desc}")
        return [sanitize_text(t) for t in texts if t.strip()]
    except Exception as e:
        print(f"XML parse failed for {abs_path}: {e}")
        return []


def embed_texts(openai_key: str, texts: List[str], *, on_progress=None) -> List[List[float]]:
    if not texts:
        return []
    client = openai.OpenAI(api_key=openai_key)
    results: List[List[float]] = []
    # Simple sequential over batches with optional throttle
    total = len(texts)
    start_ts = time.time()
    for i in range(0, total, EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        # Truncate long texts
        batch = [t[:8000] if len(t) > 8000 else t for t in batch]
        attempt = 0
        while True:
            try:
                resp = client.embeddings.create(model="text-embedding-3-small", input=batch)
                for d in resp.data:
                    results.append(d.embedding)
                if on_progress:
                    done = min(i + EMBED_BATCH_SIZE, total)
                    elapsed = max(1e-6, time.time() - start_ts)
                    rate = done / elapsed
                    remaining = total - done
                    eta = remaining / max(1e-6, rate)
                    on_progress(done, total, rate, eta)
                break
            except Exception as e:
                attempt += 1
                if attempt >= 5:
                    print(f"Embedding failed after retries: {e}")
                    raise
                backoff = min(30.0, (2 ** attempt) + (0.25 * attempt))
                print(f"Embedding error, retrying in {backoff:.1f}s: {e}")
                time.sleep(backoff)
    return results


def insert_rows_with_backoff(bq: bigquery.Client, table_ref: str, rows: List[Dict]) -> None:
    table = bq.get_table(table_ref)
    attempt = 0
    while True:
        try:
            errors = bq.insert_rows_json(table, rows)
            if errors:
                raise RuntimeError(str(errors))
            return
        except Exception as e:
            attempt += 1
            if attempt >= 6:
                raise
            backoff = min(45.0, (2 ** attempt) + (0.5 * attempt))
            print(f"Insert retry {attempt} in {backoff:.1f}s due to: {e}")
            time.sleep(backoff)


def main() -> int:
    ap = argparse.ArgumentParser(description="Index latest Godot docs into BigQuery vector store")
    ap.add_argument("--gcp-project", required=True, help="GCP project id for BigQuery")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--table", default=DEFAULT_TABLE)
    ap.add_argument("--openai-key", default=os.getenv("OPENAI_API_KEY"))
    ap.add_argument("--branch", default=DEFAULT_BRANCH, help="godot-docs branch (e.g., latest, 4.3)")
    ap.add_argument("--engine-branch", default=DEFAULT_ENGINE_BRANCH, help="godot engine branch for XML class docs")
    ap.add_argument("--user-id", default=DEFAULT_USER_ID)
    ap.add_argument("--project-id", default=DEFAULT_PROJECT_ID)
    ap.add_argument("--force", action="store_true", help="Force reindex: clear prior docs corpus first")
    ap.add_argument("--max-files-rst", type=int, default=int(os.getenv("DOCS_MAX_FILES_RST", "0")), help="Limit number of rst files (for quick tests)")
    ap.add_argument("--max-files-xml", type=int, default=int(os.getenv("DOCS_MAX_FILES_XML", "0")), help="Limit number of xml files (for quick tests)")
    ap.add_argument("--limit-chunks", type=int, default=int(os.getenv("DOCS_LIMIT_CHUNKS", "0")), help="Cap total chunks embedded (for quick tests)")
    ap.add_argument("--out-jsonl", type=str, default=os.getenv("DOCS_OUT_JSONL", ""), help="Write rows to JSONL file(s) instead of streaming to BigQuery. If total rows exceed shard size, files are suffixed with .NNN.jsonl")
    ap.add_argument("--jsonl-shard-size", type=int, default=int(os.getenv("DOCS_JSONL_SHARD_SIZE", "10000")), help="Max rows per JSONL shard file")
    args = ap.parse_args()

    if not args.openai_key:
        print("OPENAI_API_KEY is required (env or --openai-key)")
        return 2

    use_jsonl = bool(args.out_jsonl)
    if not use_jsonl:
        bq = bigquery.Client(project=args.gcp_project)
        ensure_dataset_and_table(bq, args.gcp_project, args.dataset, args.table)
        table_ref = f"{args.gcp_project}.{args.dataset}.{args.table}"

    # Optional clear
    if args.force:
        try:
            q = f"""
            DELETE FROM `{table_ref}`
            WHERE user_id = @uid AND project_id = @pid
            """
            cfg = bigquery.QueryJobConfig(query_parameters=[
                bigquery.ScalarQueryParameter("uid", "STRING", args.user_id),
                bigquery.ScalarQueryParameter("pid", "STRING", args.project_id),
            ])
            bq.query(q, job_config=cfg).result()
            print("Cleared prior docs corpus")
        except Exception as e:
            print(f"WARN: clear failed: {e}")

    # Fetch docs
    cache_root = _d("build_temp/docs_cache")
    os.makedirs(cache_root, exist_ok=True)

    # Fetch docs repo (rst). 'latest' maps to 'master' in repo
    doc_branches = [args.branch] if args.branch else []
    if args.branch == 'latest':
        doc_branches = ['master', 'latest', 'stable', '4.3', '4.2']
    else:
        # Always have robust fallbacks
        doc_branches = [args.branch, 'master', 'stable', '4.3', '4.2']
    docs_root = download_github_zip_with_fallback(
        'godotengine/godot-docs', doc_branches, os.path.join(cache_root, 'godot-docs')
    )

    # Fetch engine repo (XML)
    eng_branches = [args.engine_branch, 'master', '4.3', '4.2']
    try:
        engine_root = download_tar_gz(
            f"https://github.com/godotengine/godot/archive/refs/heads/{eng_branches[0]}.tar.gz",
            os.path.join(cache_root, f"godot-engine-{eng_branches[0]}")
        )
    except requests.HTTPError:
        # try fallbacks
        engine_root = None
        for br in eng_branches[1:]:
            try:
                engine_root = download_tar_gz(
                    f"https://github.com/godotengine/godot/archive/refs/heads/{br}.tar.gz",
                    os.path.join(cache_root, f"godot-engine-{br}")
                )
                break
            except Exception as e:
                print(f"Engine download failed for {br}: {e}")
        if not engine_root:
            raise

    # Collect texts
    items: List[Tuple[str, str]] = []  # (file_key, text)
    # RST tutorials/guides
    rst_count = 0
    for relkey, ap in enumerate_docs_files(os.path.join(docs_root, "docs")):
        try:
            with open(ap, "r", encoding="utf-8", errors="ignore") as f:
                raw = f.read()
            txt = rst_to_plaintext(raw)
            items.append((relkey, txt))
            rst_count += 1
            if args.max_files_rst and rst_count >= args.max_files_rst:
                break
        except Exception as e:
            print(f"RST read failed {ap}: {e}")
    # XML class reference
    xml_count = 0
    for relkey, ap in enumerate_class_xml(engine_root):
        texts = parse_xml_chunks(ap)
        for idx, t in enumerate(texts):
            items.append((f"{relkey}#part{idx}", t))
        xml_count += 1
        if args.max_files_xml and xml_count >= args.max_files_xml:
            break

    # Chunk and embed
    rows: List[Dict] = []
    now_iso = dt.datetime.utcnow().isoformat()
    all_texts: List[str] = []
    meta: List[Tuple[str, int]] = []  # (file_key, chunk_index)
    for file_key, text in items:
        for content, start, end in chunk_by_chars(text, CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS):
            all_texts.append(content)
            meta.append((file_key, len([m for m in meta if m[0] == file_key])))
            if args.limit_chunks and len(all_texts) >= args.limit_chunks:
                break
        if args.limit_chunks and len(all_texts) >= args.limit_chunks:
            break

    print(f"Embedding {len(all_texts)} chunks from {len(items)} documents...")
    sys.stdout.flush()
    def _on_prog(done, total, rate, eta):
        # Update every ~5s
        if done == total or done % max(1, EMBED_BATCH_SIZE) == 0:
            print(f"  • Embedded {done}/{total} chunks | {rate:.1f} ch/s | ETA {eta/60:.1f} min")
    embeddings = embed_texts(args.openai_key, all_texts, on_progress=_on_prog)
    if len(embeddings) != len(all_texts):
        print("ERROR: embedding count mismatch")
        return 3

    # Build rows
    for (file_key, chunk_index), emb, content in zip(meta, embeddings, all_texts):
        file_hash = hashlib.md5((file_key + str(chunk_index) + content[:128]).encode("utf-8")).hexdigest()
        rows.append({
            "id": f"{args.user_id}:{args.project_id}:{file_key}:{chunk_index}",
            "user_id": args.user_id,
            "project_id": args.project_id,
            "file_path": file_key,
            "chunk_index": chunk_index,
            "content": content,
            "start_line": 0,
            "end_line": 0,
            "file_hash": file_hash,
            "indexed_at": now_iso,
            "embedding": emb,
        })

    if use_jsonl:
        # Write JSONL shards
        out_path = os.path.abspath(args.out_jsonl)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        shard_size = max(1, args.jsonl_shard_size)
        total_rows = len(rows)
        print(f"Writing {total_rows} rows to JSONL (shard_size={shard_size}) at {out_path}")
        if total_rows <= shard_size:
            with open(out_path, 'w', encoding='utf-8') as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
            print(f"  • Wrote {total_rows} rows -> {out_path}")
        else:
            base, ext = os.path.splitext(out_path)
            if ext.lower() != '.jsonl':
                base = out_path
                ext = '.jsonl'
            shard_idx = 0
            written = 0
            for i in range(0, total_rows, shard_size):
                shard = rows[i:i + shard_size]
                shard_file = f"{base}.{shard_idx:03d}{ext}"
                with open(shard_file, 'w', encoding='utf-8') as f:
                    for r in shard:
                        f.write(json.dumps(r, ensure_ascii=False) + '\n')
                written += len(shard)
                print(f"  • Wrote {written}/{total_rows} rows -> {shard_file}")
                sys.stdout.flush()
                shard_idx += 1
        print("Done.")
    else:
        print(f"Inserting {len(rows)} rows into {table_ref} ...")
        # Insert in batches to avoid payload limits
        batch_size = 5000
        total_rows = len(rows)
        inserted = 0
        for i in range(0, total_rows, batch_size):
            insert_rows_with_backoff(bq, table_ref, rows[i:i + batch_size])
            inserted = min(total_rows, i + batch_size)
            print(f"  • Inserted {inserted}/{total_rows} rows")
            sys.stdout.flush()
        print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())


