from copy import deepcopy
from pathlib import Path
from tool_calling.data import load_jsonl
from transformers import AutoTokenizer
from tool_calling.training_data import encode_training_record
import random
from tool_calling.split_data import input_fingerprint
from tool_calling.prepare_data import save_jsonl
from tool_calling.model_config import BASE_MODEL_ID, BASE_MODEL_REVISION


def make_no_call_record(source: dict, spec: dict) -> dict:
    source_tool_names = [
        tool["function"]["name"]
        for tool in source["tools"]
    ]

    if len(source_tool_names) != len(set(source_tool_names)):
        raise ValueError("Duplicate tool names in source record")

    expected_calls=source["messages"][-1]["tool_calls"]
    if len(expected_calls) != 1:
        raise ValueError("Expected exactly one tool call")

    called_name=expected_calls[0]["function"]["name"]
    keep_tool_names=spec["keep_tool_names"]
    if not keep_tool_names:
        raise ValueError("No tools to keep")
    elif called_name in keep_tool_names:
        raise ValueError("Called tool must be removed")

    if len(keep_tool_names) != len(set(keep_tool_names)):
        raise ValueError("Duplicate tool names")

    response = spec["response"]

    if not isinstance(response, str):
        raise ValueError("Response must be a string")

    if not response.strip():
        raise ValueError("Response must not be empty")



    kept_tools = []

    for tool in source["tools"]:
        tool_name = tool["function"]["name"]

        if tool_name in keep_tool_names:
            kept_tools.append(deepcopy(tool))

    kept_names = {tool["function"]["name"] for tool in kept_tools}

    if kept_names != set(keep_tool_names):
        raise ValueError("Tool names do not match")

    messages = deepcopy(source["messages"][:-1])
    messages.append({
        "role": "assistant",
        "content": response,
        "tool_calls": []
    })

    return {
        "sample_id": source["sample_id"] + "-no-call",
        "tools": kept_tools,
        "messages": messages
    }






def validate_no_call_record(record: dict, tokenizer) -> None:
    encoded = encode_training_record(record, tokenizer)
    supervised_ids = [
        label for label in encoded["labels"] if label != -100
    ]
    decoded = tokenizer.decode(
        supervised_ids,
        skip_special_tokens=False
    )

    response = record["messages"][-1]["content"]
    if response not in decoded:
        raise ValueError(
            f"原始回答内容未完整出现在 decoded 中。\n"
            f"原始回答: {repr(response)}\n"
            f"解码结果: {repr(decoded)}"
        )

    if "<tool_call>" in decoded:
        raise ValueError("解码结果中不应存在 <tool_call> 标签，但检测到了。")

    if "</tool_call>" in decoded:
        raise ValueError("解码结果中不应存在 </tool_call> 标签，但检测到了。")

    if not decoded.rstrip().endswith("<|im_end|>"):
        raise ValueError(
            f"解码结果末尾必须包含 <|im_end|> 标签。\n"
            f"解码结果末尾部分: {repr(decoded[-50:])}"
        )


def select_positive_records(records: list[dict], source_ids: list[str]) -> list[dict]:
    source_ids_set = set(source_ids)
    removal_candidates = []

    for record in records:
        sample_id = record["sample_id"]
        if sample_id not in source_ids_set and sample_id != "xlam-4480":
            tool_calls = record["messages"][-1]["tool_calls"]
            if len(tool_calls) == 1:
                removal_candidates.append(record)

    removed_records = random.Random(42).sample(removal_candidates, 199)

    removed_ids = {r["sample_id"] for r in removed_records}
    removed_ids.add("xlam-4480")

    selected_records = [r for r in records if r["sample_id"] not in removed_ids]

    if len(selected_records) != 1800:
        raise ValueError(f"筛选后的正样本数量必须为 1800 条，实际为 {len(selected_records)} 条")

    return selected_records


def validate_v3_dataset(
    positive_records: list[dict],
    no_call_records: list[dict],
    source_ids: list[str],
    validation_records: list[dict],
    test_records: list[dict]
) -> None:
    v3_records = positive_records + no_call_records

    if len(positive_records) != 1800:
        raise ValueError(f"正样本数量必须为 1800，实际为 {len(positive_records)}")

    if len(no_call_records) != 200:
        raise ValueError(f"负样本数量必须为 200，实际为 {len(no_call_records)}")

    if len(v3_records) != 2000:
        raise ValueError(f"合并后总数量必须为 2000，实际为 {len(v3_records)}")

    positive_ids = {r["sample_id"] for r in positive_records}
    if not set(source_ids).issubset(positive_ids):
        raise ValueError("必须保留的 200 个配对来源 ID 未全部存在于正样本中")

    unique_ids = {r["sample_id"] for r in v3_records}
    if len(unique_ids) != 2000:
        raise ValueError(f"唯一样本 ID 数量必须为 2000，实际为 {len(unique_ids)}")

    v3_fingerprints = {input_fingerprint(record) for record in v3_records}
    if len(v3_fingerprints) != 2000:
        raise ValueError(f"唯一输入指纹数量必须为 2000，实际为 {len(v3_fingerprints)}")

    val_fingerprints = {input_fingerprint(record) for record in validation_records}
    val_intersection = v3_fingerprints & val_fingerprints
    if val_intersection:
        raise ValueError(f"与验证集的指纹交集必须为 0，实际交集大小为 {len(val_intersection)}")

    test_fingerprints = {input_fingerprint(record) for record in test_records}
    test_intersection = v3_fingerprints & test_fingerprints
    if test_intersection:
        raise ValueError(f"与测试集的指纹交集必须为 0，实际交集大小为 {len(test_intersection)}")
    if any(not r["messages"][-1]["tool_calls"] for r in positive_records):
        raise ValueError("Positive records must contain tool calls")

    if any(r["messages"][-1]["tool_calls"] for r in no_call_records):
        raise ValueError("No-call records must not contain tool calls")
def main():
    project_dir = Path(__file__).resolve().parents[1]
    train_path = project_dir / "data" / "processed" / "v1" / "train.jsonl"
    specs_path = project_dir / "data" / "examples" / "v3_no_call_specs.jsonl"

    records = load_jsonl(train_path)
    specs = load_jsonl(specs_path)

    source_ids = [spec["source_id"] for spec in specs]

    if len(specs) != 200:
        raise ValueError(
            f"specs 数量必须恰好为 200 条，实际为 {len(specs)} 条"
        )

    seen_source_ids = set()
    duplicated_source_ids = set()
    for source_id in source_ids:
        if source_id in seen_source_ids:
            duplicated_source_ids.add(source_id)
        seen_source_ids.add(source_id)

    if duplicated_source_ids:
        raise ValueError(
            f"source_id 存在重复: {sorted(duplicated_source_ids)}"
        )

    record_map = {
        record["sample_id"]: record
        for record in records
    }

    missing_source_ids = [
        source_id for source_id in source_ids
        if source_id not in record_map
    ]

    if missing_source_ids:
        raise ValueError(
            f"以下 source_id 不存在于 record_map: {missing_source_ids}"
        )


    cache_dir = project_dir / "models" / "hf"

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        cache_dir=cache_dir,
        revision=BASE_MODEL_REVISION,
        local_files_only=True
    )

    no_call_records = []
    failed_records = []

    for spec in specs:
        source_id = spec["source_id"]
        try:
            source = record_map[source_id]
            record = make_no_call_record(source, spec)

            validate_no_call_record(record, tokenizer)
            no_call_records.append(record)

        except ValueError as e:
            failed_records.append({
                "source_id": source_id,
                "error": str(e)
            })

    total = len(specs)
    passed = len(no_call_records)
    failed = len(failed_records)

    print(f"Total samples: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed > 0:
        for item in failed_records:
            print(f"ID: {item['source_id']} | Error: {item['error']}")

        raise ValueError("No-call validation failed")

    positive_records = select_positive_records(records, source_ids)

    validation_records = load_jsonl(
        project_dir / "data" / "processed" / "v1" / "validation.jsonl"
    )

    test_records = load_jsonl(
        project_dir / "data" / "processed" / "v1" / "test.jsonl"
    )

    validate_v3_dataset(
        positive_records,
        no_call_records,
        source_ids,
        validation_records,
        test_records
    )

    v3_records = positive_records + no_call_records

    output_path = project_dir / "data" / "processed" / "v3" / "train.jsonl"

    save_jsonl(v3_records, output_path)

    positive_ids = {r["sample_id"] for r in positive_records}

    print("\nV3 dataset summary:")
    print(f"Total records: {len(v3_records)}")
    print(f"Positive records: {len(positive_records)}")
    print(f"No-call records: {len(no_call_records)}")
    print(f"Retained paired sources: {len(set(source_ids) & positive_ids)}")
    print("Validation fingerprint overlap: 0")
    print("Test fingerprint overlap: 0")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()
