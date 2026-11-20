import copy
import json
from pathlib import Path

from tool_calling.data import load_jsonl


def normalize_bfcl_schema(schema: dict) -> dict:
    normalized_schema = copy.deepcopy(schema)
    type_mapping = {
        "dict": "object",
        "float": "number",
        "tuple": "array"
    }
    if "type" in normalized_schema:
        schema_type = normalized_schema["type"]
        normalized_schema["type"] = type_mapping.get(
            schema_type, schema_type
        )

    if "properties" in normalized_schema:
        for name, parameter in normalized_schema["properties"].items():
            normalized_schema["properties"][name] = normalize_bfcl_schema(parameter)

    if "items" in normalized_schema:
        normalized_schema["items"] = normalize_bfcl_schema(
            normalized_schema["items"]
        )

    return normalized_schema


def load_bfcl_records(
    data_dir: Path,
    limit_per_category: int = 100,
    offset_per_category: int = 0
) -> list[dict]:
    simple_records=load_jsonl(data_dir /"BFCL_v4_simple_python.json")
    irrelevance_records=load_jsonl(data_dir /"BFCL_v4_irrelevance.json")
    answers_records=load_jsonl(data_dir / "possible_answer" / "BFCL_v4_simple_python.json")

    if limit_per_category < 1:
        raise ValueError("样本数量必须大于 0")

    if offset_per_category < 0:
        raise ValueError("偏移量不能为负数")

    end_index = offset_per_category + limit_per_category

    if end_index > min(len(simple_records), len(irrelevance_records)):
        raise ValueError("样本取样范围超出有效范围")
    
    answer_map={record["id"]  : record["ground_truth"] for record in answers_records}
    records = []

    category_records = {
        "simple_python": simple_records,
        "irrelevance": irrelevance_records
    }
    for category, raw_records in category_records.items():
        for record in raw_records[offset_per_category:end_index]:
            tools=[]
            for function in record["function"]:
                normalized_function = copy.deepcopy(function)
                normalized_function["parameters"] = normalize_bfcl_schema(
                    function["parameters"]
                )
                tools.append({
                    "type": "function",
                    "function": normalized_function
                })

            ground_truth = answer_map[record["id"]] if category == "simple_python" else []


            normalized_record = {
                "sample_id": record["id"],
                "category": category,
                "prompt_messages": record["question"][0],
                "tools": tools,
                "bfcl_functions": record["function"],
                "ground_truth": ground_truth
            }

            records.append(normalized_record)

    return records



def main() -> None:
    data_dir = Path(
        r"E:\Codex\references\gorilla"
        r"\berkeley-function-call-leaderboard\bfcl_eval\data"
    )
    records = load_bfcl_records(data_dir)
    simple_records = [record for record in records if record["category"] == "simple_python"]
    irrelevance_records = [record for record in records if record["category"] == "irrelevance"]
    sample_ids={record["sample_id"] for record in records}

    assert len(simple_records) == 100
    assert len(irrelevance_records) == 100
    assert len(records) == 200
    assert len(sample_ids) == 200

    print(f"simple_python: {len(simple_records)}")
    print(f"irrelevance: {len(irrelevance_records)}")
    print(f"唯一ID数量: {len(sample_ids)}")

    record_map = {r["sample_id"]: r for r in records}

    coordinate_record = record_map["simple_python_83"]

    normalized_coord = coordinate_record["tools"][0]["function"]["parameters"]["properties"]["coord1"]
    original_coord = coordinate_record["bfcl_functions"][0]["parameters"]["properties"]["coord1"]

    assert normalized_coord["type"] == "array"
    assert normalized_coord["items"]["type"] == "number"
    assert original_coord["type"] == "tuple"

    print(json.dumps(normalized_coord))
    print(json.dumps(original_coord))

    irrelevance_record = record_map["irrelevance_0"]


    assert irrelevance_record["ground_truth"] == []
    assert len(irrelevance_record["tools"]) == 1

    print(json.dumps(irrelevance_record, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()




