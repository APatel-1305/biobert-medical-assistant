# assistant.py
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List
from dataclasses import dataclass

FAISS_INDEX = "faiss_responses.index"
FAISS_META = "faiss_responses_meta.json"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RANKER_MODEL_DIR = "./biobert_ranker"  # saved after training

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 5

@dataclass
class Candidate:
    text: str
    source: str
    score: float = 0.0

def load_faiss_and_meta():
    if not os.path.exists(FAISS_INDEX) or not os.path.exists(FAISS_META):
        raise FileNotFoundError("FAISS index or meta missing. Run build_vectorstore.py first.")
    idx = faiss.read_index(FAISS_INDEX)
    with open(FAISS_META, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return idx, meta

def retrieve_candidates(query: str, idx, meta, k=TOP_K):
    sbert = SentenceTransformer(EMB_MODEL)
    q_emb = sbert.encode([query], convert_to_numpy=True)
    D, I = idx.search(q_emb, k)
    candidates = []
    for i in I[0]:
        if i < 0 or i >= len(meta):
            continue
        entry = meta[i]
        candidates.append(Candidate(text=entry["text"], source=entry.get("source", "")))
    return candidates

class Ranker:
    def __init__(self, model_dir=RANKER_MODEL_DIR, device=DEVICE):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(device)
        self.device = device

    def score(self, patient: str, responses: List[Candidate], batch_size=8):
        texts_a = [patient] * len(responses)
        texts_b = [c.text for c in responses]
        scores = []
        for i in range(0, len(texts_a), batch_size):
            batch_a = texts_a[i:i+batch_size]
            batch_b = texts_b[i:i+batch_size]
            enc = self.tokenizer(batch_a, batch_b, truncation=True, padding=True, return_tensors="pt", max_length=256)
            enc = {k: v.to(self.device) for k,v in enc.items()}
            with torch.no_grad():
                out = self.model(**enc)
                logits = out.logits  # shape (batch, 2)
                # probability for label=1 (good reply)
                probs = torch.softmax(logits, dim=-1)[:,1].cpu().numpy()
                scores.extend(probs.tolist())
        # attach scores
        for c, s in zip(responses, scores):
            c.score = float(s)
        return responses

def reply(patient_query: str):
    # load index & meta
    idx, meta = load_faiss_and_meta()
    # retrieve candidates
    candidates = retrieve_candidates(patient_query, idx, meta, k=TOP_K)
    if not candidates:
        return "Sorry — I couldn't find candidate responses. Please consult a clinician. (No KB responses found.)"
    # load ranker
    ranker = Ranker()
    ranked = ranker.score(patient_query, candidates)
    # pick best
    ranked.sort(key=lambda x: x.score, reverse=True)
    best = ranked[0]
    # safety: if score low, be cautious
    if best.score < 0.4:
        # choose cautious fallback
        return (
            "I don't have a confident answer from my knowledge base. "
            "Please consult a healthcare professional. "
            f"\n\nClosest match (score={best.score:.2f}): {best.text}"
            "\n\nDISCLAIMER: This is informational only, not medical advice."
        )
    return f"{best.text}\n\n(SOURCE: {best.source} — confidence {best.score:.2f})\n\nDISCLAIMER: This is informational only, not medical advice. Consult a licensed clinician."

if __name__ == "__main__":
    print("Medical dialogue assistant (simple CLI). Type 'quit' to exit.\n")
    while True:
        q = input("Patient: ").strip()
        if q.lower() in ("quit", "exit"):
            break
        ans = reply(q)
        print("\nDoctor Assistant:\n")
        print(ans)
        print("\n" + "-"*40 + "\n")
