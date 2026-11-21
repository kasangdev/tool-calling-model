from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

from tool_calling.data import load_jsonl
from tool_calling.model_config import BASE_MODEL_ID, BASE_MODEL_REVISION



def encode_training_record(
    record: dict,
    tokenizer,
    max_sequence_tokens: int = 1024,
) -> dict:
    sample_id = record["sample_id"]
    full_ids=tokenizer.apply_chat_template(record["messages"],
                                           tools=record["tools"],
                                           tokenize=True,
                                           add_generation_prompt=False,
                                           return_dict=False)

    prompt_ids=tokenizer.apply_chat_template(record["messages"][:-1],
                                           tools=record["tools"],
                                           tokenize=True,
                                           add_generation_prompt=True,
                                           return_dict=False)
    prompt_length=len(prompt_ids)

    if prompt_ids!=full_ids[:prompt_length]:
        raise ValueError(f"Sample {sample_id}: Prompt prefix mismatch")
    if len(full_ids) > max_sequence_tokens:
        raise ValueError(f"Sample {sample_id}: Sequence too long")
    if len(full_ids) <= prompt_length:
        raise ValueError(f"Sample {sample_id}: No answer tokens")

    labels=[-100]*prompt_length + full_ids[prompt_length:]


    return {"input_ids": full_ids,
            "labels": labels}


def build_training_dataset(
    records: list[dict],
    tokenizer,
    max_sequence_tokens: int = 1024,
):
    encoded_records=[encode_training_record(record,
                                            tokenizer,
                                            max_sequence_tokens=max_sequence_tokens) for record in records]

    dataset=Dataset.from_list(encoded_records)

    return dataset

def main() -> None:
    project_dir=Path(__file__).resolve().parents[1]
    data_dir=project_dir / "data" / "processed" / "v1"
    cache_dir=project_dir / "models" / "hf"
    max_sequence_tokens=1024

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        cache_dir=cache_dir,
        revision=BASE_MODEL_REVISION,
        local_files_only=True,
    )

    train_records=load_jsonl(data_dir / "train.jsonl")
    validation_records=load_jsonl(data_dir / "validation.jsonl")

    train_dataset = build_training_dataset(
        train_records,
        tokenizer,
        max_sequence_tokens=max_sequence_tokens,
    )

    validation_dataset = build_training_dataset(
        validation_records,
        tokenizer,
        max_sequence_tokens=max_sequence_tokens,
    )

    assert len(train_dataset)==2000
    assert len(validation_dataset)==200

    for dataset in (train_dataset, validation_dataset):
        for encoded_record in dataset:
            assert len(encoded_record["input_ids"]) == len(encoded_record["labels"])
            assert len(encoded_record["input_ids"]) <= max_sequence_tokens
            assert any(token_id != -100 for token_id in encoded_record["labels"])

    collator = DataCollatorForLanguageModeling(
        pad_token_id=tokenizer.pad_token_id,
    )

    batch = collator([train_dataset[0], train_dataset[1]])

    print(batch["input_ids"].shape)
    print(batch["labels"].shape)
    print(batch["attention_mask"].shape)

    batch_shape = batch["input_ids"].shape
    assert batch["labels"].shape == batch_shape
    assert batch["attention_mask"].shape == batch_shape
    assert len(batch_shape) == 2
    assert batch_shape[0] == 2

    padding_mask = batch["attention_mask"] == 0
    assert (batch["labels"][padding_mask] == -100).all().item()
    print("Padding tokens:", padding_mask.sum().item())

    supervised_ids = [
        token_id for token_id in train_dataset[0]["labels"]
        if token_id != -100
    ]

    decoded_answer = tokenizer.decode(
        supervised_ids,
        skip_special_tokens=False,
    )

    print(decoded_answer)

    assert "<|im_end|>" in decoded_answer

    print(f"训练集样本数量:{len(train_dataset)}")
    print(f"验证集样本数量:{len(validation_dataset)}")


if __name__ == "__main__":
    main()






