# preprocess_dialogues.py
import json
import random
import os
from pathlib import Path
import argparse
import pandas as pd

"""
Input JSON format expected (per your example):
[
  {
    "description": "...",
    "utterances": [
      "patient: ...",
      "doctor: ..."
    ]
  },
  ...
]
"""

def load_dialogues(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pairs = []
    for entry in data:
        utts = entry.get("utterances", [])
        if len(utts) < 2:
            continue
        # normalize: find patient and doctor utterance
        patient = None
        doctor = None
        for u in utts:
            u_strip = u.strip()
            if u_strip.lower().startswith("patient:"):
                patient = u_strip.split(":", 1)[1].strip()
            elif u_strip.lower().startswith("doctor:"):
                doctor = u_strip.split(":", 1)[1].strip()
        # fallback: sometimes order is known
        if not patient and len(utts) >= 1:
            patient = utts[0].split(":", 1)[-1].strip()
        if not doctor and len(utts) >= 2:
            doctor = utts[1].split(":", 1)[-1].strip()
        if patient and doctor:
            pairs.append({"patient": patient, "doctor": doctor})
    return pairs

def create_ranking_dataset(train_files, dev_files, out_dir="processed"):
    os.makedirs(out_dir, exist_ok=True)
    # load all dialogues to build negative sampling pool
    all_pairs = []
    for f in train_files + dev_files:
        all_pairs.extend(load_dialogues(f))

    # build dataset rows with negative samples
    rows = []
    doctors_pool = [p["doctor"] for p in all_pairs]
    for p in all_pairs:
        # positive example
        rows.append({"patient": p["patient"], "response": p["doctor"], "label": 1})
        # negative sampling: pick 1-3 wrong responses
        negs = random.sample(doctors_pool, k=min(3, max(1, len(doctors_pool)-1)))
        # ensure not picking the correct one
        negs = [n for n in negs if n != p["doctor"]][:2]
        for n in negs:
            rows.append({"patient": p["patient"], "response": n, "label": 0})
    df = pd.DataFrame(rows)
    train_df = df.sample(frac=0.9, random_state=42)
    dev_df = df.drop(train_df.index)
    train_path = Path(out_dir) / "ranker_train.csv"
    dev_path = Path(out_dir) / "ranker_dev.csv"
    train_df.to_csv(train_path, index=False)
    dev_df.to_csv(dev_path, index=False)
    print(f"Saved train -> {train_path}, dev -> {dev_path}, total rows: {len(df)}")
    return str(train_path), str(dev_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data", help="folder containing english-train.json etc.")
    parser.add_argument("--out-dir", type=str, default="processed", help="where to place processed csv files")
    args = parser.parse_args()

    train_files = [os.path.join(args.data_dir, "english-train.json")]
    dev_files   = [os.path.join(args.data_dir, "english-dev.json")]
    create_ranking_dataset(train_files, dev_files, out_dir=args.out_dir)
