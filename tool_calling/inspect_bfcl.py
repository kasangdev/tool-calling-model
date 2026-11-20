import json
from pathlib import Path

from tool_calling.data import load_jsonl


def inspect_records(name: str, records: list[dict]) -> None:
    print(name)
    print(f"记录总数: {len(records)}")
    unique_ids = {record["id"] for record in records}
    print(f"唯一ID数量: {len(unique_ids)}")
    print(json.dumps(records[0], ensure_ascii=False, indent=2))



def main() -> None:
    data_dir = Path(
        r"E:\Codex\references\gorilla"
        r"\berkeley-function-call-leaderboard\bfcl_eval\data"
    )

    simple_path=data_dir / "BFCL_v4_simple_python.json"
    irrelevance_path=data_dir / "BFCL_v4_irrelevance.json"
    answer_path=data_dir / "possible_answer" / "BFCL_v4_simple_python.json"

    simple_records = load_jsonl(simple_path)
    irrelevance_records = load_jsonl(irrelevance_path)
    answer_records = load_jsonl(answer_path)

    inspect_records("simple_python", simple_records)
    inspect_records("irrelevance", irrelevance_records)
    inspect_records("possible_answer", answer_records)

    simple_ids = {record["id"] for record in simple_records}
    answer_ids = {record["id"] for record in answer_records}

    assert simple_ids == answer_ids
    assert len(simple_ids) == len(simple_records)
    assert len(answer_ids) == len(answer_records)

    print("simple_python 题目与答案 ID 检查通过")

if __name__ == "__main__":
    main()