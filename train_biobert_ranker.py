# train_biobert_ranker.py
import os
import argparse
import pandas as pd
from datasets import Dataset
import evaluate
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"

def prepare_dataset(csv_path, tokenizer, max_length=256):
    df = pd.read_csv(csv_path)
    df = df.fillna("")
    texts = list(zip(df["patient"].tolist(), df["response"].tolist()))
    labels = df["label"].astype(int).tolist()
    ds = Dataset.from_dict({"patient": [t[0] for t in texts], "response": [t[1] for t in texts], "label": labels})

    def tokenize_fn(example):
        # pair encode: patient as first, response as second
        return tokenizer(example["patient"], example["response"], truncation=True, max_length=max_length)
    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=["patient", "response"])
    return tokenized

def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return metric.compute(predictions=preds, references=labels)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=str, default="processed/ranker_train.csv")
    parser.add_argument("--dev-csv", type=str, default="processed/ranker_dev.csv")
    parser.add_argument("--output-dir", type=str, default="./biobert_ranker")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    print("Preparing datasets...")
    train_ds = prepare_dataset(args.train_csv, tokenizer)
    dev_ds = prepare_dataset(args.dev_csv, tokenizer)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=False,  # set True if your GPU supports it
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print("Saved ranker model to", args.output_dir)

if __name__ == "__main__":
    main()
