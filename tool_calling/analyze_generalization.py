import copy
import json
import hashlib
from pathlib import Path
from tool_calling.data import load_jsonl


def tool_fingerprint(tool: dict) -> str:
    normalized_tool=copy.deepcopy(tool)
    parameters = normalized_tool["function"]["parameters"]
    if "required" in parameters:
        parameters["required"] = sorted(parameters["required"])
    serialized=json.dumps(normalized_tool,sort_keys=True,ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()



def classify_tool_exposure(
    record: dict,
    training_fingerprints: set[str],
) -> str:
    tool_fingerprints = {}
    for tool in record["tools"]:
        tool_name = tool["function"]["name"]
        if tool_name in tool_fingerprints:
            raise ValueError(f"{record['sample_id']}: 重复工具名称 {tool_name}")
        tool_fingerprints[tool_name] = tool_fingerprint(tool)

    expected_calls = record["messages"][-1]["tool_calls"]
    if not expected_calls:
        raise ValueError(f"{record['sample_id']}: 标准调用为空")
    exposure_flags = []
    for call in expected_calls:
        tool_name=call["function"]["name"]
        if tool_name not in tool_fingerprints:
            raise ValueError(
                f"{record['sample_id']}: 工具不存在 {tool_name}"
            )
        fingerprint=tool_fingerprints[tool_name]

        exposure_flags.append(fingerprint in training_fingerprints)

    if all(exposure_flags):
        return "seen"
    elif not any(exposure_flags):
        return "unseen"
    else:
        return "mixed"


def main() -> None:


    project_dir = Path(__file__).resolve().parents[1]
    train_path = project_dir / "data" / "processed" / "v1" / "train.jsonl"
    test_path = project_dir / "data" / "processed" / "v1" / "test.jsonl"
    baseline_path = project_dir / "runs" / "baseline" / "test200" / "predictions.jsonl"
    lora_path = project_dir / "runs" / "lora" / "v1" / "test200" / "predictions.jsonl"
    output_dir = project_dir / "runs" / "analysis" / "v1" / "test200"

    train_records = load_jsonl(train_path)
    test_records = load_jsonl(test_path)
    baseline_results = load_jsonl(baseline_path)
    lora_results = load_jsonl(lora_path)

    assert len(train_records) == 2000
    assert len(test_records) == 200
    assert len(baseline_results) == 200
    assert len(lora_results) == 200

    test_ids = [record["sample_id"] for record in test_records]
    baseline_ids = [record["sample_id"] for record in baseline_results]
    lora_ids = [record["sample_id"] for record in lora_results]

    assert test_ids == baseline_ids == lora_ids


    training_fingerprints = set()
    for record in train_records:
        for tool in record["tools"]:
            training_fingerprints.add(tool_fingerprint(tool))

    groups = {
        "seen": {
            "sample_count": 0,
            "sample_ids": [],
            "baseline_correct_count": 0,
            "lora_correct_count": 0
        },
        "unseen": {
            "sample_count": 0,
            "sample_ids": [],
            "baseline_correct_count": 0,
            "lora_correct_count": 0
        },
        "mixed": {
            "sample_count": 0,
            "sample_ids": [],
            "baseline_correct_count": 0,
            "lora_correct_count": 0
        }
    }

    for record, baseline_result, lora_result in zip(
            test_records, baseline_results, lora_results
    ):
        category = classify_tool_exposure(record, training_fingerprints)
        group = groups[category]
        group["sample_count"] += 1
        group["sample_ids"].append(record["sample_id"])
        if baseline_result["exact_match"]:
            group["baseline_correct_count"] += 1
        if lora_result["exact_match"]:
            group["lora_correct_count"] += 1

    for group in groups.values():
        sample_count = group["sample_count"]
        if sample_count > 0:
            group["baseline_exact_match_rate"] = group["baseline_correct_count"] / sample_count
            group["lora_exact_match_rate"] = group["lora_correct_count"] / sample_count
        else:
            group["baseline_exact_match_rate"] = None
            group["lora_exact_match_rate"] = None

    total_sample_count = sum(
        group["sample_count"] for group in groups.values()
    )
    total_baseline_correct = sum(
        group["baseline_correct_count"] for group in groups.values()
    )
    total_lora_correct = sum(
        group["lora_correct_count"] for group in groups.values()
    )

    assert total_sample_count == 200
    assert total_baseline_correct == sum(
        result["exact_match"] for result in baseline_results
    )
    assert total_lora_correct == sum(
        result["exact_match"] for result in lora_results
    )

    seen_ids = set(groups["seen"]["sample_ids"])
    unseen_ids = set(groups["unseen"]["sample_ids"])
    mixed_ids = set(groups["mixed"]["sample_ids"])

    assert not (seen_ids & unseen_ids)
    assert not (seen_ids & mixed_ids)
    assert not (unseen_ids & mixed_ids)

    assert len(seen_ids | unseen_ids | mixed_ids) == 200

    report = {
        "train_size": len(train_records),
        "test_size": len(test_records),
        "unique_training_tool_count": len(training_fingerprints),
        "groups": groups
    }

    report_path = output_dir / "generalization_report.json"

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()