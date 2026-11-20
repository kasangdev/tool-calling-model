import copy
import json
from pathlib import Path

from transformers import AutoTokenizer
from tool_calling.data import load_jsonl
from tool_calling.prepare_data import save_jsonl
from tool_calling.training_data import encode_training_record


def remove_schema_defaults(record: dict) -> tuple[dict, int]:
    cleaned_record=copy.deepcopy(record)
    removed_count=0
    for tool in cleaned_record["tools"]:
        properties = tool["function"]["parameters"]["properties"]
        for parameter in properties.values():
            if "default" in parameter:
                del parameter["default"]
                removed_count+=1
    return cleaned_record, removed_count


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    source_train_path = project_dir / "data" / "processed" / "v1" / "train.jsonl"
    output_dir = project_dir / "data" / "processed" / "v2"
    cache_dir = project_dir / "models" / "hf"
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    max_sequence_tokens = 1024

    original_records = load_jsonl(source_train_path)
    cleaned_records = []
    changed_records = 0
    removed_default_fields = 0

    for original_record in original_records:
        cleaned_record,removed_count=remove_schema_defaults(original_record)
        assert cleaned_record["sample_id"] == original_record["sample_id"]
        assert cleaned_record["messages"] == original_record["messages"]

        if removed_count > 0:
            changed_records+=1
        removed_default_fields+=removed_count
        cleaned_records.append(cleaned_record)

    original_sample_ids=[record["sample_id"] for record in original_records]
    cleaned_sample_ids=[record["sample_id"] for record in cleaned_records]

    assert len(original_records) == 2000
    assert len(cleaned_records) == 2000
    assert original_sample_ids == cleaned_sample_ids

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        local_files_only=True,
    )
    max_observed_sequence_tokens = 0
    for record in cleaned_records:
        encoded_record=encode_training_record(record,
                                              tokenizer,
                                              max_sequence_tokens=max_sequence_tokens)
        sequence_length=len(encoded_record["input_ids"])
        max_observed_sequence_tokens = max(
            max_observed_sequence_tokens,
            sequence_length
        )

    report = {
        "source_train_path": str(source_train_path),
        "train_size": len(cleaned_records),
        "changed_records": changed_records,
        "removed_default_fields": removed_default_fields,
        "max_sequence_tokens": max_sequence_tokens,
        "max_observed_sequence_tokens": max_observed_sequence_tokens
    }

    train_path = output_dir / "train.jsonl"
    report_path = output_dir / "preparation_report.json"

    save_jsonl(cleaned_records, train_path)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
