import json
from pathlib import Path
from collections import Counter




def main():

    project_dir=Path(__file__).resolve().parents[1]
    data_path=project_dir/"data"/"raw"/"xlam_function_calling_60k.json"
    with data_path.open(encoding="utf-8") as f:
        records=json.load(f)
    default_stats = Counter()

    for record in records:
        tools=json.loads(record["tools"])
        for tool in tools:
            parameters=tool["parameters"]
            for parameter in parameters.values():
                if "default" not in parameter:
                    default_kind="missing"
                elif parameter["default"] is None:
                    default_kind="null"
                else:
                    default_kind="value"

                is_optional=", optional" in parameter["type"]

                label=f"optional={is_optional}, default={default_kind}"

                default_stats[label]+=1

    report={}
    report["default_stats"]=dict(default_stats)
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()

