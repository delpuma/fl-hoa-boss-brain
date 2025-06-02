import os, json, datetime as dt, uuid, base64
from io import BytesIO
from typing import List, Dict

import openai, pinecone
from pypdf import PdfReader

# ---- 1. Keys & clients ----
openai.api_key = os.getenv("OPENAI_API_KEY")
PINECONE_KEY   = os.getenv("PINECONE_API_KEY")
PINECONE_ENV   = os.getenv("PINECONE_ENV", "us-east-1")
INDEX_NAME     = "kitt3n"
NAMESPACE      = "shared_memory"

pinecone.init(api_key=PINECONE_KEY, environment=PINECONE_ENV)
index = pinecone.Index(INDEX_NAME)


# ---- 2. Helpers ----
def chunk_text(text: str, size: int = 350, overlap: int = 50):
    words = text.split()
    for i in range(0, len(words), size - overlap):
        yield " ".join(words[i : i + size])

def embed(texts: List[str]) -> List[List[float]]:
    resp = openai.embeddings.create(model="text-embedding-3-large", input=texts)
    return [d.embedding for d in resp.data]

def extract_pdf_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


# ---- 3.  Vercel handler ----
def handler(request) -> Dict:
    """
    Expects JSON: { "openai_file_ids": ["file_abc", "file_xyz"] }
    Returns:      { "job_id": "...", "message": "Done" }
    """
    body = json.loads(request.body or "{}")
    file_ids: List[str] = body.get("openai_file_ids", [])
    if not file_ids:
        return response(400, {"error": "openai_file_ids array required"})

    #  A single Vercel invocation can last up to 10‑–30 s (plan‑dependent).
    #  So we process each file synchronously; keep PDFs < ~2 MB for safety.
    for fid in file_ids:
        pdf_bytes = openai.files.content(fid)                      # 1⃣ download
        text      = extract_pdf_text(pdf_bytes)                    # 2⃣ parse
        vectors   = []
        for order, chunk in enumerate(chunk_text(text)):
            emb = embed([chunk])[0]                                # 3⃣ embed
            meta = {
                "source_file_id": fid,
                "order": order,
                "ts": dt.datetime.utcnow().isoformat(timespec="seconds"),
            }
            vec_id = f"{fid}_{order}"
            vectors.append((vec_id, emb, meta))
        # 4⃣ upsert in one shot (fastest)
        index.upsert(vectors=vectors, namespace=NAMESPACE)

    job_id = uuid.uuid4().hex
    return response(202, {"job_id": job_id, "message": "Ingestion complete"})

def response(status: int, body: Dict):
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }
