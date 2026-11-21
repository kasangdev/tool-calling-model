import json
import time
from pathlib import Path
import argparse

from tool_calling.data import load_jsonl
from tool_calling.baseline import load_model_and_tokenizer, generate_completion
from tool_calling.evaluation import evaluate_response
from tool_calling.model_config import BASE_MODEL_ID
from tool_calling.prepare_data import save_jsonl


def evaluate_records(
    records: list[dict],
    model,
    tokenizer,
    max_new_tokens: int = 1024,
) -> list[dict]:
    results=[]
    for index, record in enumerate(records,start=1):
        start_time = time.perf_counter()
        generated_text = generate_completion(model,
                                             tokenizer,
                                             record,
                                             max_new_tokens=max_new_tokens)
        end_time = time.perf_counter()
        elapsed_seconds = end_time - start_time
        raw_expected_calls=record["messages"][-1]["tool_calls"]
        expected_calls=[call["function"] for call in raw_expected_calls]
        metrics = evaluate_response(generated_text, expected_calls)
        result={"sample_id":record["sample_id"],
                "generated_text":generated_text,
                "expected_calls":expected_calls,
                "elapsed_seconds":elapsed_seconds}
        result.update(metrics)
        results.append(result)
        print(f"[{index}/{len(records)}] {record['sample_id']}")


    return results


def summarize_results(results: list[dict]) -> dict:
    sample_count = len(results)
    if sample_count == 0:
        raise ValueError("No evaluation results")
    parse_ok_rate=sum(result["parse_ok"] for result in results)/sample_count
    tool_names_match_rate=sum(result["tool_names_match"] for result in results)/sample_count
    arguments_match_rate=sum(result["arguments_match"] for result in results)/sample_count
    exact_match_rate=sum(result["exact_match"] for result in results)/sample_count
    mean_elapsed_seconds=sum(result["elapsed_seconds"] for result in results)/sample_count

    summary={"sample_count":sample_count,
             "parse_ok_rate":parse_ok_rate,
             "tool_names_match_rate":tool_names_match_rate,
             "arguments_match_rate":arguments_match_rate,
             "exact_match_rate":exact_match_rate,
             "mean_elapsed_seconds":mean_elapsed_seconds}

    return summary


def main()->None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="validation",
    )
    args = parser.parse_args()
    sample_limit = args.limit

    split_prefix = {
        "validation": "dev",
        "test": "test"
    }
    subset_name = f"{split_prefix[args.split]}{sample_limit}"
    project_dir=Path(__file__).resolve().parents[1]
    data_path=project_dir / "data" / "processed" / "v1" / f"{args.split}.jsonl"
    output_dir=project_dir / "runs" / "baseline" / subset_name
    cache_dir=project_dir / "models" / "hf"

    model_id = BASE_MODEL_ID
    max_new_tokens = 1024

    all_records = load_jsonl(data_path)
    if not 1 <= sample_limit <= len(all_records):
        raise ValueError(f"Sample limit {sample_limit} is out of range")
    records = all_records[:sample_limit]

    assert len(records)==sample_limit

    model,tokenizer=load_model_and_tokenizer(model_id,cache_dir)
    results = evaluate_records(records,model,tokenizer,max_new_tokens=max_new_tokens)
    summary = summarize_results(results)
    summary["model_id"]=model_id
    summary["max_new_tokens"]=max_new_tokens
    summary["do_sample"]=False
    summary["split"] = args.split
    summary["data_path"] = str(data_path)

    assert len(results) == sample_limit
    assert summary["sample_count"] == sample_limit
    sample_ids=[record["sample_id"] for record in records]
    result_sample_ids=[record["sample_id"] for record in results]
    assert sample_ids == result_sample_ids
    assert 0 <= summary["parse_ok_rate"] <= 1
    assert 0 <= summary["tool_names_match_rate"] <= 1
    assert 0 <= summary["arguments_match_rate"] <= 1
    assert 0 <= summary["exact_match_rate"] <= 1

    predictions_path=output_dir / "predictions.jsonl"
    summary_path=output_dir / "summary.json"
    save_jsonl(results,predictions_path)
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary,file,ensure_ascii=False, indent=2)
    print(json.dumps(summary,ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()


