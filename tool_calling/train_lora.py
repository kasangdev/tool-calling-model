from pathlib import Path
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
import math
import torch
from tool_calling.baseline import load_model_and_tokenizer
from tool_calling.data import load_jsonl
from tool_calling.training_data import build_training_dataset
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
from transformers import set_seed
import json
import argparse



def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--version",
        choices=["v1", "v2", "v3"],
        default="v1",
    )

    args = parser.parse_args()

    project_dir=Path(__file__).resolve().parents[1]
    train_path = project_dir / "data" / "processed" / args.version / "train.jsonl"
    validation_path = project_dir / "data" / "processed" / "v1" / "validation.jsonl"
    cache_dir=project_dir / "models" / "hf"
    output_dir = project_dir / "runs" / "lora" / args.version
    model_id="Qwen/Qwen2.5-0.5B-Instruct"
    max_sequence_tokens=1024

    set_seed(42)

    model, tokenizer = load_model_and_tokenizer(model_id,cache_dir=cache_dir)

    model.train()
    model.config.use_cache = False

    train_records = load_jsonl(train_path)
    validation_records = load_jsonl(validation_path)
    train_dataset=build_training_dataset(train_records,
                                            tokenizer,
                                            max_sequence_tokens=max_sequence_tokens)
    validation_dataset=build_training_dataset(validation_records,
                                              tokenizer,
                                              max_sequence_tokens=max_sequence_tokens)

    assert len(train_dataset) == 2000
    assert len(validation_dataset) == 200

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    training_args = SFTConfig(
        output_dir=str(output_dir),
        max_steps=-1,
        num_train_epochs=1.0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=1,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=25,
        bf16=True,
        fp16=False,
        optim="adamw_torch",
        logging_steps=10,
        activation_checkpointing=True,
        activation_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        dataloader_num_workers=0,
        seed=42,
        data_seed=42,
        max_length=1024,
        packing=False,
        dataset_kwargs={"skip_prepare_dataset": True}
    )
    collator = DataCollatorForLanguageModeling(
        pad_token_id=tokenizer.pad_token_id,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=collator,
        peft_config=lora_config
    )

    trainer.model.print_trainable_parameters()

    trainable_params, total_params = trainer.model.get_nb_trainable_parameters()

    assert 0 < trainable_params < total_params

    torch.cuda.reset_peak_memory_stats()

    train_output=trainer.train()
    eval_metrics=trainer.evaluate()

    global_step=train_output.global_step
    train_loss=train_output.metrics["train_loss"]
    eval_loss=eval_metrics["eval_loss"]

    assert global_step==500
    assert math.isfinite(train_loss)
    assert math.isfinite(eval_loss)

    peak_allocated_gib=torch.cuda.max_memory_allocated()/1024 ** 3
    peak_reserved_gib=torch.cuda.max_memory_reserved()/1024 ** 3

    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))

    assert (adapter_dir / "adapter_config.json").is_file()
    assert (adapter_dir / "adapter_model.safetensors").is_file()

    lora_summary = lora_config.to_dict()
    lora_summary["target_modules"] = sorted(lora_config.target_modules)

    summary = {
        "model_id": model_id,
        "data_version": args.version,
        "train_data_path": str(train_path),
        "validation_data_path": str(validation_path),
        "train_size": len(train_dataset),
        "validation_size": len(validation_dataset),
        "global_step": global_step,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "lora_config": lora_summary,
        "training_args": training_args.to_dict(),
        "train_metrics": train_output.metrics,
        "eval_metrics": eval_metrics,
        "peak_allocated_gib": peak_allocated_gib,
        "peak_reserved_gib": peak_reserved_gib,
        "best_model_checkpoint":trainer.state.best_model_checkpoint,
        "best_metric":trainer.state.best_metric

    }

    summary_path=output_dir / "run_summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

