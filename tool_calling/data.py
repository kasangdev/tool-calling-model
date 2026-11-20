import json
from pathlib import Path




def load_jsonl(path:str | Path) -> list[dict]:
    records = []
    file_path = Path(path)
    with file_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f,start=1):
            if not line.strip():
                continue
            try:
                record=json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid json in line {i}")

            is_dict =isinstance(record,dict)

            if not is_dict:
                raise ValueError(f"Expected a dict in line {i}")
            records.append(record)


    return records