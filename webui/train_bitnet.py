from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def _fail(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)
    raise SystemExit(1)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _iter_exception_chain(exc: Exception):
    seen: set[int] = set()
    current: Exception | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        next_exc = current.__cause__ or current.__context__
        current = next_exc if isinstance(next_exc, Exception) else None


def _missing_module(exc: Exception, module_name: str) -> bool:
    for current in _iter_exception_chain(exc):
        if isinstance(current, ModuleNotFoundError):
            name = str(getattr(current, "name", "") or "").strip()
            if name == module_name or name.startswith(f"{module_name}."):
                return True
        text = str(current or "")
        if f"No module named '{module_name}'" in text or f'No module named "{module_name}"' in text:
            return True
    return False


def _format_dependency_error(exc: Exception) -> str:
    parts = [
        "Missing BitNet training dependencies. Install with:",
        "python -m pip install -r webui/requirements-train.txt",
    ]
    if _missing_module(exc, "triton"):
        if os.name == "nt":
            parts.extend(
                [
                    "",
                    "Triton is also required by onebitllms on Windows. Install:",
                    "python -m pip install triton-windows",
                ]
            )
        elif sys.platform == "darwin":
            parts.extend(
                [
                    "",
                    "Triton is required by onebitllms and is not generally available on macOS.",
                    "Use a Linux/WSL training environment, or a Windows environment with triton-windows.",
                ]
            )
        else:
            parts.extend(
                [
                    "",
                    "Triton is also required by onebitllms. Install:",
                    "python -m pip install triton",
                ]
            )

    chain = " <- ".join(f"{current.__class__.__name__}: {current}" for current in _iter_exception_chain(exc))
    parts.extend(["", f"Import error: {chain}"])
    return "\n".join(parts)


def _import_onebitllms_helpers():
    first_exc: Exception | None = None
    try:
        # onebitllms >= 0.0.3
        from onebitllms.utils import quantize_to_1bit, replace_linear_with_bitnet_linear

        return quantize_to_1bit, replace_linear_with_bitnet_linear
    except Exception as exc:
        first_exc = exc

    try:
        # compatibility path for older onebitllms releases
        from onebitllms.model import quantize_to_1bit, replace_linear_with_bitnet_linear

        return quantize_to_1bit, replace_linear_with_bitnet_linear
    except Exception as exc:
        raise exc from first_exc


def _import_training_stack() -> dict[str, Any]:
    try:
        import torch
        from datasets import load_dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:
        _fail(_format_dependency_error(exc))

    try:
        quantize_to_1bit, replace_linear_with_bitnet_linear = _import_onebitllms_helpers()
    except Exception as exc:
        _fail(_format_dependency_error(exc))

    return {
        "torch": torch,
        "load_dataset": load_dataset,
        "replace_linear_with_bitnet_linear": replace_linear_with_bitnet_linear,
        "quantize_to_1bit": quantize_to_1bit,
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


def _friendly_model_error(model_id: str, exc: Exception) -> str:
    raw = str(exc or "")
    lowered = raw.lower()

    if "gated repo" in lowered or "access to model" in lowered or "401 client error" in lowered:
        return (
            f"Cannot access Hugging Face model '{model_id}'.\n"
            "This model appears gated/private.\n"
            "Request access and authenticate first:\n"
            "1) huggingface-cli login\n"
            "2) Or set HF_TOKEN in your environment.\n"
            f"Original error: {raw}"
        )

    if "repository not found" in lowered or "404 client error" in lowered:
        return (
            f"Model repository not found: '{model_id}'.\n"
            "Check the model ID/revision or provide a valid local model directory.\n"
            f"Original error: {raw}"
        )

    return f"Failed to load model/tokenizer '{model_id}'. Original error: {raw}"


def _export_quantized_checkpoint(quantize_to_1bit: Any, output_dir: Path, quantized_dir: Path) -> None:
    try:
        quantize_to_1bit(
            input_checkpoint_path=str(output_dir),
            output_checkpoint_path=str(quantized_dir),
        )
        return
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc).lower():
            raise

    # compatibility for older onebitllms argument names
    quantize_to_1bit(model_name_or_path=str(output_dir), output_dir=str(quantized_dir))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BitNet-native continued training (full fine-tune, not PEFT LoRA)."
    )
    parser.add_argument("--base-model", required=True, help="HF model ID or local model directory")
    parser.add_argument(
        "--model-revision",
        default="prequantized",
        help="Model revision/branch (for example: prequantized)",
    )
    parser.add_argument(
        "--tokenizer-model",
        default="",
        help="Optional tokenizer model ID/path. Defaults to base model.",
    )
    parser.add_argument("--dataset-path", required=True, help="Path to JSON/JSONL/CSV dataset")
    parser.add_argument("--output-dir", required=True, help="Output directory for trained checkpoint")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument(
        "--quantize-output",
        action="store_true",
        help="Also export a post-training 1-bit checkpoint via onebitllms.quantize_to_1bit.",
    )
    parser.add_argument(
        "--quantized-output-dir",
        default="",
        help="Optional target directory for --quantize-output. Default: <output-dir>-1bit",
    )
    args = parser.parse_args()

    deps = _import_training_stack()
    torch = deps["torch"]
    load_dataset = deps["load_dataset"]
    replace_linear_with_bitnet_linear = deps["replace_linear_with_bitnet_linear"]
    quantize_to_1bit = deps["quantize_to_1bit"]
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

    _log(f"Loading dataset from {dataset_path}")
    raw_dataset = _load_training_data(load_dataset, dataset_path)

    def format_row(example: dict[str, Any]) -> dict[str, Any]:
        return {"text": _build_text(example)}

    formatted = raw_dataset.map(format_row)
    formatted = formatted.filter(lambda row: bool(str(row["text"]).strip()))
    if len(formatted) == 0:
        _fail("No valid training rows were found after formatting.")

    revision = (args.model_revision or "").strip() or "main"
    tokenizer_model = (args.tokenizer_model or "").strip() or args.base_model
    _log(f"Loading tokenizer: {tokenizer_model}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True)
    except Exception as exc:
        _fail(_friendly_model_error(tokenizer_model, exc))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _log(f"Loading base model: {args.base_model} (revision={revision})")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }
    if revision:
        model_load_kwargs["revision"] = revision
    try:
        model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_load_kwargs)
    except Exception as exc:
        _fail(_friendly_model_error(args.base_model, exc))

    _log("Converting Linear layers to BitNetLinear for continued training")
    try:
        model = replace_linear_with_bitnet_linear(model)
    except Exception as exc:
        _fail(
            "Failed to initialize BitNetLinear layers. Ensure this checkpoint is supported by onebitllms.\n"
            f"Original error: {exc}"
        )

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
        logging_steps=1,
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

    _log("Starting BitNet continued training job...")
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metadata = {
        "train_mode": "bitnet_continued",
        "base_model": args.base_model,
        "model_revision": revision,
        "tokenizer_model": tokenizer_model,
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "learning_rate": args.learning_rate,
        "max_seq_len": args.max_seq_len,
        "quantize_output": bool(args.quantize_output),
    }
    quantized_output = ""

    if args.quantize_output:
        quantized_dir = (
            Path(args.quantized_output_dir).expanduser().resolve()
            if (args.quantized_output_dir or "").strip()
            else Path(str(output_dir) + "-1bit")
        )
        _log(f"Quantizing trained checkpoint to 1-bit output: {quantized_dir}")
        try:
            _export_quantized_checkpoint(quantize_to_1bit, output_dir, quantized_dir)
        except Exception as exc:
            _fail(
                "Training completed, but 1-bit quantization export failed.\n"
                "You can retry quantization manually with onebitllms.quantize_to_1bit.\n"
                f"Original error: {exc}"
            )
        quantized_output = str(quantized_dir)
        metadata["quantized_output_dir"] = quantized_output

    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _log(f"Training complete. Checkpoint saved to: {output_dir}")
    if quantized_output:
        _log(f"1-bit quantized checkpoint saved to: {quantized_output}")
    _log("If needed, run your GGUF conversion flow before activating in llama.cpp/bitnet.cpp server.")


if __name__ == "__main__":
    main()
