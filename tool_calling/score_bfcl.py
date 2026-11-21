import argparse
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from statistics import mean, median
from tool_calling.prepare_data import save_jsonl

from bfcl_eval.constants.enums import Language
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker

from tool_calling.data import load_jsonl
from tool_calling.bfcl_data import add_bfcl_data_dir_argument, load_bfcl_records
from tool_calling.evaluation import parse_tool_calls
from tool_calling.model_config import BASE_MODEL_ID



def register_model_config(model_id: str) -> None:
    source_config = MODEL_CONFIG_MAPPING["Qwen/Qwen3-4B-Instruct-2507"]

    new_config = replace(
        source_config,
        model_name=model_id,
        underscore_to_dot=False
    )

    MODEL_CONFIG_MAPPING[model_id] = new_config


def score_record(
    record: dict,
    generated_text: str,
    model_id: str
) -> dict:

    try:
        predicted_calls = parse_tool_calls(generated_text)
    except ValueError as e:
        return {
            "parse_ok": False,
            "correct": False,
            "predicted_calls": None,
            "error": str(e)
        }

    if record["category"] == "irrelevance":
        return {
            "parse_ok": True,
            "correct": predicted_calls == [],
            "predicted_calls": predicted_calls,
            "error": None
        }

    model_output = [
        {call["name"]: call["arguments"]}
        for call in predicted_calls
    ]

    check_result = ast_checker(
        func_description=record["bfcl_functions"],
        model_output=model_output,
        possible_answer=record["ground_truth"],
        language=Language.PYTHON,
        test_category="simple_python",
        model_name=model_id
    )

    return {
    "parse_ok": True,
    "correct": check_result["valid"],
    "predicted_calls": predicted_calls,
    "error": None,
    "checker_result": check_result
}


def run_score_checks(records: list[dict], model_id: str) -> None:
    record_map = {record["sample_id"]: record for record in records}
    triangle_record = record_map["simple_python_0"]
    generated_text = (
        '<tool_call>\n'
        '{"name": "calculate_triangle_area", '
        '"arguments": {"base": 10, "height": 5}}\n'
        '</tool_call>'
    )

    result = score_record(triangle_record, generated_text, model_id)

    assert result["parse_ok"] is True
    assert result["correct"] is True
    print(result)

    factorial_record = next(
        record for record in records
        if record["bfcl_functions"][0]["name"] == "math.factorial"
    )

    generated_text = (
        '<tool_call>\n'
        '{"name": "math.factorial", "arguments": {"number": 5}}\n'
        '</tool_call>'
    )
    result = score_record(factorial_record, generated_text, model_id)
    assert result["parse_ok"] is True
    assert result["correct"] is True
    print(result)

    generated_text = (
        '<tool_call>\n'
        '{"name": "calculate_triangle_area", '
        '"arguments": {"base": 10, "height": 5}}'
    )
    result = score_record(triangle_record, generated_text, model_id)
    assert result["parse_ok"] is False
    assert result["correct"] is False
    assert result["predicted_calls"] is None

    print(result)


def summarize_categories(results: list[dict]) -> dict:
    category_stats = {}
    for category in ("simple_python", "irrelevance"):
        category_results = [row for row in results if row["category"] == category]
        sample_count = len(category_results)
        correct_count = sum(row["correct"] for row in category_results)
        stats = {
            "sample_count": sample_count,
            "correct_count": correct_count,
            "accuracy": correct_count / sample_count if sample_count else 0,
            "parse_failure_count": sum(not row["parse_ok"] for row in category_results),
        }
        if category == "irrelevance":
            stats["false_call_count"] = sum(
                bool(row["parse_ok"] and row["predicted_calls"])
                for row in category_results
            )
        else:
            stats["missing_call_count"] = sum(
                row["parse_ok"] and not row["predicted_calls"]
                for row in category_results
            )
        category_stats[category] = stats
    return category_stats


def main():
    parser = argparse.ArgumentParser()
    add_bfcl_data_dir_argument(parser)
    parser.add_argument(
        "--model",
        choices=["baseline", "lora-v1", "lora-v3", "baseline-with-v3-veto-live"],
        required=True,
    )
    parser.add_argument("--limit-per-category", type=int, default=5)
    parser.add_argument("--offset-per-category", type=int, default=0)
    args = parser.parse_args()
    use_veto = args.model == "baseline-with-v3-veto-live"

    project_dir = Path(__file__).resolve().parents[1]
    model_id = BASE_MODEL_ID
    data_dir = args.bfcl_data_dir

    subset_name = f"per_category_{args.limit_per_category}"

    if args.offset_per_category != 0:
        subset_name += f"_offset_{args.offset_per_category}"

    output_dir = (
            project_dir / "runs" / "bfcl"
            / subset_name
            / args.model
    )

    records = load_bfcl_records(
        data_dir,
        limit_per_category=args.limit_per_category,
        offset_per_category=args.offset_per_category
    )

    predictions = load_jsonl(output_dir / "predictions.jsonl")

    assert len(records) == 2 * args.limit_per_category
    assert len(predictions) == len(records)
    assert [p["sample_id"] for p in predictions] == [
        r["sample_id"] for r in records
    ]

    register_model_config(model_id)

    check_records = load_bfcl_records(data_dir, limit_per_category=2)

    run_score_checks(check_records, model_id)
    print("三个评分测试全部通过")
    results = []
    baseline_results = []

    for record, prediction in zip(records, predictions):
        baseline_result = None
        if use_veto and prediction["vetoed"]:
            baseline_result = score_record(
                deepcopy(record), prediction["baseline_generated_text"], model_id
            )
        result = score_record(record, prediction["generated_text"], model_id)
        category = record["category"]
        results.append({**prediction, **result, "category": category})
        if use_veto:
            if baseline_result is None:
                baseline_result = result
            baseline_results.append({**baseline_result, "category": category})
    assert len(results) == len(records)

    summary = {
        "model_id": model_id,
        "model": args.model,
        "bfcl_commit": "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8",
        "limit_per_category": args.limit_per_category,
        "offset_per_category": args.offset_per_category,
        "sample_count": len(results),
        "categories": summarize_categories(results),
    }
    if use_veto:
        baseline_times = [row["baseline_latency_seconds"] for row in predictions]
        request_times = [row["latency_seconds"] for row in predictions]
        summary["baseline_categories"] = summarize_categories(baseline_results)
        summary["routing"] = {
            "gate_checks": sum(row["gate_checked"] for row in predictions),
            "vetoes": sum(row["vetoed"] for row in predictions),
            "correct_calls_vetoed": sum(
                prediction["vetoed"] and baseline_result["correct"]
                for prediction, baseline_result in zip(predictions, baseline_results)
            ),
        }
        summary["latency_seconds"] = {
            "baseline_mean": mean(baseline_times),
            "baseline_median": median(baseline_times),
            "end_to_end_mean": mean(request_times),
            "end_to_end_median": median(request_times),
            "end_to_end_total": sum(request_times),
        }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    save_jsonl(results, output_dir / "scores.jsonl")

if __name__ == "__main__":
    main()
