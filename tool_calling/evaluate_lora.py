from pathlib import Path
from peft import PeftModel
import json
import argparse


from tool_calling.data import load_jsonl
from tool_calling.batch_evaluate import evaluate_records, summarize_results
from tool_calling.prepare_data import save_jsonl
from tool_calling.baseline import load_model_and_tokenizer


def load_lora_model_and_tokenizer(
    model_id: str,
    cache_dir: str | Path,
    adapter_dir: str | Path,
):

    base_model,tokenizer = load_model_and_tokenizer(model_id, cache_dir)
    model=PeftModel.from_pretrained(base_model,
                                    str(adapter_dir),
                                    is_trainable=False,
                                    local_files_only=True)
    model.eval()


    return model,tokenizer



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=30)
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
    sample_limit = args.limit
    split_prefix = {
        "validation": "dev",
        "test": "test"
    }
    subset_name = f"{split_prefix[args.split]}{sample_limit}"

    project_dir = Path(__file__).resolve().parents[1]
    data_path = project_dir / "data" / "processed" / "v1" / f"{args.split}.jsonl"
    cache_dir = project_dir / "models" / "hf"
    adapter_dir = project_dir / "runs" / "lora" / args.version / "adapter"
    baseline_path = project_dir / "runs" / "baseline" / subset_name / "predictions.jsonl"
    output_dir = project_dir / "runs" / "lora" / args.version / subset_name
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    max_new_tokens = 1024

    all_records = load_jsonl(data_path)
    if not 1 <= sample_limit <= len(all_records):
        raise ValueError(f"Sample limit {sample_limit} is out of range")
    records = all_records[:sample_limit]

    baseline_results=load_jsonl(baseline_path)

    assert len(records) == sample_limit
    assert len(baseline_results) == sample_limit

    sample_ids=[record["sample_id"] for record in records]
    baseline_sample_ids=[record["sample_id"] for record in baseline_results]

    assert sample_ids == baseline_sample_ids

    model, tokenizer = load_lora_model_and_tokenizer(model_id, cache_dir,adapter_dir)
    print(model.active_adapters)
    results=evaluate_records(records, model, tokenizer, max_new_tokens=max_new_tokens)
    summary=summarize_results(results)


    assert len(results) == sample_limit
    assert summary["sample_count"] == sample_limit

    result_sample_ids = [result["sample_id"] for result in results]
    assert result_sample_ids == baseline_sample_ids

    improved_sample_ids=[]
    regressed_sample_ids=[]


    for baseline_result, result in zip(baseline_results, results):
        if not baseline_result["exact_match"] and result["exact_match"]:
            improved_sample_ids.append(result["sample_id"])
        elif baseline_result["exact_match"] and not result["exact_match"]:
            regressed_sample_ids.append(result["sample_id"])

    baseline_correct_count=sum(result["exact_match"] for result in baseline_results)
    lora_correct_count=sum(result["exact_match"] for result in results)
    baseline_rate=baseline_correct_count/len(baseline_sample_ids)
    lora_rate=summary["exact_match_rate"]
    delta_pp=(lora_rate - baseline_rate) * 100

    assert lora_correct_count - baseline_correct_count == (
            len(improved_sample_ids) - len(regressed_sample_ids)
    )

    summary["model_id"] = model_id
    summary["adapter_dir"] = str(adapter_dir)
    summary["max_new_tokens"] = max_new_tokens
    summary["do_sample"] = False
    summary["baseline_exact_match_rate"] = baseline_rate
    summary["exact_match_delta_pp"] = delta_pp
    summary["improved_sample_ids"] = improved_sample_ids
    summary["regressed_sample_ids"] = regressed_sample_ids
    summary["split"] = args.split
    summary["data_path"] = str(data_path)

    predictions_path=output_dir / "predictions.jsonl"
    summary_path=output_dir / "summary.json"
    save_jsonl(results, predictions_path)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
if __name__=="__main__":
    main()

