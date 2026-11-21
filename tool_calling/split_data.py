import hashlib
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

from tool_calling.data import load_jsonl
from tool_calling.model_config import BASE_MODEL_ID, BASE_MODEL_REVISION
from tool_calling.prepare_data import save_jsonl




def input_fingerprint(record: dict) -> str:
    payload={"messages":record["messages"][:-1],
             "tools":record["tools"]}
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    data_bytes = text.encode("utf-8")
    fingerprint = hashlib.sha256(data_bytes).hexdigest()
    return fingerprint



def build_splits(
    records: list[dict],
    tokenizer,
    train_size: int = 2000,
    validation_size: int = 200,
    test_size: int = 200,
    max_sequence_tokens: int = 1024,
    max_calls: int = 3,
    seed: int = 42,
) -> tuple[dict[str, list[dict]], dict]:
    candidates=[]
    seen_fingerprints=set()
    filtered_by_calls=0
    duplicate_inputs=0
    for record in records:
        call_count=len(record["messages"][-1]["tool_calls"])
        if not (1<=call_count<=max_calls):
            filtered_by_calls+=1
            continue
        fingerprint=input_fingerprint(record)
        if fingerprint in seen_fingerprints:
            duplicate_inputs+=1
            continue
        seen_fingerprints.add(fingerprint)
        candidates.append(record)

    rng = random.Random(seed)
    rng.shuffle(candidates)
    total_needed=train_size+validation_size+test_size
    selected=[]
    length_checked=0
    filtered_by_length=0

    for record in candidates:
        length_checked+=1
        token_ids = tokenizer.apply_chat_template(
            record["messages"],
            tools=record["tools"],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
        )
        sequence_length=len(token_ids)
        if sequence_length>max_sequence_tokens:
            filtered_by_length+=1
            continue
        selected.append(record)
        if len(selected)==total_needed:
            break
    if len(selected)<total_needed:
        raise ValueError(
            f"Not enough samples: selected {len(selected)}, needed {total_needed}"
        )

    train_end=train_size
    validation_end=validation_size+train_size
    train=selected[:train_end]
    validation=selected[train_end:validation_end]
    test=selected[validation_end:]
    splits={"train":train, "validation":validation, "test":test}

    report = {
        "total_records": len(records),
        "filtered_by_calls": filtered_by_calls,
        "duplicate_inputs": duplicate_inputs,
        "candidate_records": len(candidates),
        "length_checked": length_checked,
        "filtered_by_length": filtered_by_length,
        "train_size": len(train),
        "validation_size": len(validation),
        "test_size": len(test),
        "seed": seed,
        "max_calls": max_calls,
        "max_sequence_tokens": max_sequence_tokens,
    }

    return splits,report


def main()->None:
    project_dir=Path(__file__).resolve().parents[1]
    data_path=project_dir / "data" / "processed" / "normalized.jsonl"
    output_dir=project_dir / "data" / "processed" / "v1"
    records=load_jsonl(data_path)
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        cache_dir=project_dir / "models" / "hf",
        revision=BASE_MODEL_REVISION,
        local_files_only=True,
    )
    splits,report=build_splits(records, tokenizer)

    train=splits["train"]
    validation=splits["validation"]
    test=splits["test"]
    assert len(train)==2000
    assert len(validation)==200
    assert len(test)==200
    train_ids={record["sample_id"] for record in train}
    validation_ids={record["sample_id"] for record in validation}
    test_ids={record["sample_id"] for record in test}
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)

    train_fingerprints={input_fingerprint(record) for record in train}
    validation_fingerprints={input_fingerprint(record) for record in validation}
    test_fingerprints={input_fingerprint(record) for record in test}

    assert train_fingerprints.isdisjoint(validation_fingerprints)
    assert train_fingerprints.isdisjoint(test_fingerprints)
    assert validation_fingerprints.isdisjoint(test_fingerprints)

    train_path=output_dir / "train.jsonl"
    validation_path=output_dir / "validation.jsonl"
    test_path=output_dir / "test.jsonl"
    report_path=output_dir / "split_report.json"

    save_jsonl(train,train_path)
    save_jsonl(validation,validation_path)
    save_jsonl(test,test_path)

    with open(report_path,"w",encoding="utf-8") as file:
        json.dump(report,file,ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()







