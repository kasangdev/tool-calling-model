import json
import argparse
from pathlib import Path

from tool_calling.data import load_jsonl
from tool_calling.prepare_data import save_jsonl



def diff_arguments(expected: dict, predicted: dict) -> dict:
    missing_keys=sorted([key for key in expected if key not in predicted])
    extra_keys=sorted([key for key in predicted if key not in expected])
    different_values={}
    for key in expected:
        if key in predicted and expected[key] != predicted[key]:
            different_values[key] = {
                "expected": expected[key],
                "predicted": predicted[key]
            }

    return {"missing_keys": missing_keys,
            "extra_keys": extra_keys,
            "different_values": different_values}


def analyze_result(result: dict) -> dict:
    expected_calls=result["expected_calls"]
    predicted_calls=result["predicted_calls"]

    if result["exact_match"]:
        category="correct"
    elif not result["parse_ok"]:
        category="parse_error"
    elif not predicted_calls:
        category="no_call"
    elif len(expected_calls) != len(predicted_calls):
        category="call_count_mismatch"
    elif not result["tool_names_match"]:
        category="tool_name_mismatch"
    else:
        category="argument_mismatch"

    argument_differences = []

    if category in ("correct", "argument_mismatch"):
        for call_index, (expected_call, predicted_call) in enumerate(
                zip(expected_calls, predicted_calls)
        ):
            differences=diff_arguments(expected_call["arguments"],predicted_call["arguments"])
            differences["call_index"]=call_index
            differences["tool_name"]=expected_call["name"]
            argument_differences.append(differences)

    return {"sample_id": result["sample_id"],
            "category": category,
            "argument_differences": argument_differences}

def summarize_errors(analyses: list[dict]) -> dict:
    category_counts = {
        "correct": 0,
        "parse_error": 0,
        "no_call": 0,
        "call_count_mismatch": 0,
        "tool_name_mismatch": 0,
        "argument_mismatch": 0
    }
    for analysis in analyses:
        category_counts[analysis["category"]]+=1

    sample_count=len(analyses)
    correct_count=category_counts["correct"]
    error_count=sample_count - correct_count

    assert sum(category_counts.values()) == sample_count

    missing_key_sample_count=0
    extra_key_sample_count=0
    different_value_sample_count=0
    for analysis in analyses:
        differences = analysis["argument_differences"]
        if any(diff["missing_keys"] for diff in differences):
            missing_key_sample_count += 1
        if any(diff["extra_keys"] for diff in differences):
            extra_key_sample_count += 1
        if any(diff["different_values"] for diff in differences):
            different_value_sample_count += 1

    return {
        "sample_count": sample_count,
        "correct_count": correct_count,
        "error_count": error_count,
        "category_counts": category_counts,
        "missing_key_sample_count": missing_key_sample_count,
        "extra_key_sample_count": extra_key_sample_count,
        "different_value_sample_count": different_value_sample_count
    }





def build_error_details(
    records: list[dict],
    results: list[dict],
    analyses: list[dict],
) -> list[dict]:
    error_details=[]
    for record, result, analysis in zip(records, results, analyses):
        if analysis["category"] == "correct":
            continue
        error_detail = {
            "sample_id": analysis["sample_id"],
            "category": analysis["category"],
            "argument_differences": analysis["argument_differences"],
            "query": record["messages"][0]["content"],
            "expected_calls": result["expected_calls"],
            "predicted_calls": result["predicted_calls"],
            "generated_text": result["generated_text"],
            "error": result["error"]
        }
        error_details.append(error_detail)
    return error_details




def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--version",
        choices=["v1", "v2"],
        default="v1",
    )
    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="validation",
    )
    args = parser.parse_args()

    split_prefix = {
        "validation": "dev",
        "test": "test"
    }
    subset_name = f"{split_prefix[args.split]}200"

    project_dir = Path(__file__).resolve().parents[1]

    data_path = project_dir / "data" / "processed" / "v1" / f"{args.split}.jsonl"
    baseline_path = project_dir / "runs" / "baseline" / subset_name / "predictions.jsonl"
    lora_path = project_dir / "runs" / "lora" /args.version / subset_name / "predictions.jsonl"
    output_dir = project_dir / "runs" / "analysis" / args.version / subset_name


    records = load_jsonl(data_path)
    baseline_results = load_jsonl(baseline_path)
    lora_results = load_jsonl(lora_path)


    assert len(records) == 200, f"records 长度为 {len(records)}，期望为 200"
    assert len(baseline_results) == 200, f"baseline_results 长度为 {len(baseline_results)}，期望为 200"
    assert len(lora_results) == 200, f"lora_results 长度为 {len(lora_results)}，期望为 200"


    sample_ids = [item["sample_id"] for item in records]
    baseline_sample_ids = [item["sample_id"] for item in baseline_results]
    lora_sample_ids = [item["sample_id"] for item in lora_results]


    assert sample_ids == baseline_sample_ids == lora_sample_ids


    baseline_analyses=[analyze_result(result) for result in baseline_results]
    lora_analyses=[analyze_result(result) for result in lora_results]
    baseline_summary=summarize_errors(baseline_analyses)
    lora_summary=summarize_errors(lora_analyses)

    baseline_errors=build_error_details(records, baseline_results, baseline_analyses)
    lora_errors=build_error_details(records, lora_results, lora_analyses)

    assert len(baseline_errors) == baseline_summary["error_count"]
    assert len(lora_errors) == lora_summary["error_count"]

    summary = {
        "baseline": baseline_summary,
        "lora": lora_summary,
        "split": args.split,
        "data_path": str(data_path)
    }


    baseline_errors_path = output_dir / "baseline_errors.jsonl"
    lora_errors_path = output_dir / "lora_errors.jsonl"
    summary_path = output_dir / "summary.json"

    save_jsonl(baseline_errors, baseline_errors_path)
    save_jsonl(lora_errors, lora_errors_path)
    with open(summary_path, "w",encoding="utf-8") as f:
        json.dump(summary, f,ensure_ascii=False,indent=2)
    print(json.dumps(summary,ensure_ascii=False,indent=2))



if __name__ == "__main__":
    main()