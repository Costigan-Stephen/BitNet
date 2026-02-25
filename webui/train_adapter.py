from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _fail(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(1)


def _import_training_stack() -> dict[str, Any]:
    try:
        import torch
        from datasets import Dataset, load_dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:
        _fail(
            "Missing training dependencies. Install with:\n"
            "python -m pip install -r webui/requirements-train.txt\n"
            f"Import error: {exc}"
        )
    return {
        "torch": torch,
        "Dataset": Dataset,
        "load_dataset": load_dataset,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "DataCollatorForLanguageModeling": DataCollatorForLanguageModeling,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
    }


def _build_text(example: dict[str, Any]) -> str:
    if "text" in example and str(example["text"]).strip():
        return str(example["text"])

    instruction = str(example.get("instruction", "")).strip()
    input_text = str(example.get("input", "")).strip()
    output = str(example.get("output", "")).strip()
    if instruction and output:
        if input_text:
            return (
                "### Instruction:\n"
                f"{instruction}\n\n"
                "### Input:\n"
                f"{input_text}\n\n"
                "### Response:\n"
                f"{output}"
            )
        return "### Instruction:\n" f"{instruction}\n\n" "### Response:\n" f"{output}"

    prompt = str(example.get("prompt", "")).strip()
    response = str(example.get("response", "")).strip()
    if prompt and response:
        return f"{prompt}\n{response}"

    question = str(example.get("question", "")).strip()
    answer = str(example.get("answer", "")).strip()
    if question and answer:
        return f"Q: {question}\nA: {answer}"

    return ""


def _load_training_data(load_dataset: Any, dataset_path: Path):
    suffix = dataset_path.suffix.lower()
    if suffix in {".jsonl", ".json"}:
        return load_dataset("json", data_files=str(dataset_path), split="train")
    if suffix == ".csv":
        return load_dataset("csv", data_files=str(dataset_path), split="train")
    _fail(f"Unsupported dataset format: {dataset_path.suffix}. Use JSON/JSONL/CSV.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LoRA adapter with Hugging Face + PEFT")
    parser.add_argument("--base-model", required=True, help="HF model ID or local model directory")
    parser.add_argument("--dataset-path", required=True, help="Path to JSON/JSONL/CSV dataset")
    parser.add_argument("--output-dir", required=True, help="Output directory for adapter")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names to target for LoRA",
    )
    args = parser.parse_args()

    deps = _import_training_stack()
    torch = deps["torch"]
    load_dataset = deps["load_dataset"]
    LoraConfig = deps["LoraConfig"]
    get_peft_model = deps["get_peft_model"]
    AutoModelForCausalLM = deps["AutoModelForCausalLM"]
    AutoTokenizer = deps["AutoTokenizer"]
    DataCollatorForLanguageModeling = deps["DataCollatorForLanguageModeling"]
    Trainer = deps["Trainer"]
    TrainingArguments = deps["TrainingArguments"]

    dataset_path = Path(args.dataset_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        _fail(f"Dataset path does not exist: {dataset_path}")

    print(f"Loading dataset from {dataset_path}")
    raw_dataset = _load_training_data(load_dataset, dataset_path)

    def format_row(example: dict[str, Any]) -> dict[str, Any]:
        text = _build_text(example)
        return {"text": text}

    formatted = raw_dataset.map(format_row)
    formatted = formatted.filter(lambda row: bool(str(row["text"]).strip()))
    if len(formatted) == 0:
        _fail("No valid training rows were found after formatting.")

    print(f"Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model: {args.base_model}")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    target_modules = [name.strip() for name in args.target_modules.split(",") if name.strip()]
    if not target_modules:
        _fail("No target modules provided for LoRA.")

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def tokenize_row(example: dict[str, Any]) -> dict[str, Any]:
        tokens = tokenizer(
            example["text"],
            truncation=True,
            max_length=args.max_seq_len,
            padding="max_length",
        )
        tokens["labels"] = list(tokens["input_ids"])
        return tokens

    tokenized = formatted.map(tokenize_row, remove_columns=formatted.column_names)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
    )

    print("Starting LoRA training job...")
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metadata = {
        "base_model": args.base_model,
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "learning_rate": args.learning_rate,
        "max_seq_len": args.max_seq_len,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "target_modules": target_modules,
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Training complete. Adapter saved to: {output_dir}")
    print(
        "If you need llama.cpp-compatible LoRA GGUF, run the llama.cpp conversion "
        "workflow on the produced adapter."
    )


if __name__ == "__main__":
    main()

