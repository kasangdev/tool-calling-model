import argparse
from pathlib import Path

from tool_calling.baseline import (
    load_model_and_tokenizer,
    generate_from_messages
)
from tool_calling.evaluate_lora import load_lora_model_and_tokenizer
from tool_calling.bfcl_data import load_bfcl_records
from tool_calling.prepare_data import save_jsonl
from tool_calling.v3_veto import generate_with_v3_veto


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["baseline", "lora-v1", "lora-v3", "baseline-with-v3-veto-live"],
        required=True,
    )
    parser.add_argument("--limit-per-category", type=int, default=5)
    parser.add_argument("--offset-per-category", type=int, default=0)
    args = parser.parse_args()
    use_veto = args.model == "baseline-with-v3-veto-live"

    data_dir = Path(
        r"E:\Codex\references\gorilla"
        r"\berkeley-function-call-leaderboard\bfcl_eval\data"
    )

    project_dir = Path(__file__).resolve().parents[1]
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    cache_dir = project_dir / "models" / "hf"
    max_new_tokens = 1024

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

    assert len(records) == 2 * args.limit_per_category

    if args.model == "baseline":
        model, tokenizer = load_model_and_tokenizer(
            model_id,
            cache_dir
        )
    else:
        version = "v1" if args.model == "lora-v1" else "v3"
        adapter_dir = project_dir / "runs" / "lora" / version / "checkpoint-500"
        model, tokenizer = load_lora_model_and_tokenizer(
            model_id,
            cache_dir,
            adapter_dir
        )

    predictions = []
    for index, record in enumerate(records, start=1):
        generation_args = {
            "messages": record["prompt_messages"],
            "tools": record["tools"],
            "max_new_tokens": max_new_tokens,
        }
        if use_veto:
            output = generate_with_v3_veto(model, tokenizer, **generation_args)
        else:
            output = {
                "generated_text": generate_from_messages(
                    model, tokenizer, **generation_args
                )
            }

        prediction = {
            "sample_id": record["sample_id"],
            "category": record["category"],
            **output,
        }
        predictions.append(prediction)
        print(f"[{index}/{len(records)}] {record['sample_id']}")

    assert len(predictions) == len(records)
    assert [p["sample_id"] for p in predictions] == [
        r["sample_id"] for r in records
    ]

    predictions_path = output_dir / "predictions.jsonl"
    save_jsonl(predictions,predictions_path)
    print(f"预测结果已保存: {predictions_path}")


if __name__ == "__main__":
    main()
