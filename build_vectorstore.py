# build_vectorstore.py
import os
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pathlib import Path

DOCS_DIR = "kb_responses"
OUT_INDEX = "faiss_responses.index"
OUT_META = "faiss_responses_meta.json"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # small/faster; replace with biomedical embedding if desired

def gather_responses(data_dir="data"):
    # collect unique doctor responses from all json files
    resp_set = []
    for fname in ["english-train.json", "english-dev.json", "english-test.json"]:
        p = Path(data_dir) / fname
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        for entry in data:
            utts = entry.get("utterances", [])
            # try to extract doctor utterance
            doctor = None
            for u in utts:
                u_strip = u.strip()
                if u_strip.lower().startswith("doctor:"):
                    doctor = u_strip.split(":", 1)[1].strip()
                    break
            if not doctor and len(utts) >= 2:
                doctor = utts[1].split(":", 1)[-1].strip()
            if doctor:
                resp_set.append({"text": doctor, "source": fname})
    # dedupe while preserving order
    seen = set()
    unique = []
    for r in resp_set:
        t = r["text"]
        if t not in seen:
            unique.append(r)
            seen.add(t)
    return unique

def build_index(responses, emb_model=EMB_MODEL):
    texts = [r["text"] for r in responses]
    print("Loading embedding model:", emb_model)
    sbert = SentenceTransformer(emb_model)
    embs = sbert.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    dim = embs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embs))
    faiss.write_index(index, OUT_INDEX)
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)
    print("Saved FAISS index and metadata:", OUT_INDEX, OUT_META)

if __name__ == "__main__":
    responses = gather_responses("data")
    print(f"Found {len(responses)} unique doctor responses.")
    build_index(responses)
