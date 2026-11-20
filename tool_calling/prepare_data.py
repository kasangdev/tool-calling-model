import re
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from pydantic import TypeAdapter


TYPE_ADAPTERS = {
    "str": TypeAdapter(str),
    "int": TypeAdapter(int),
    "float": TypeAdapter(float),
    "bool": TypeAdapter(bool),
    "list": TypeAdapter(list),
    "List": TypeAdapter(list),
    "Dict": TypeAdapter(dict),
    "set": TypeAdapter(set),
    "List[int]": TypeAdapter(list[int]),
    "List[float]": TypeAdapter(list[float]),
    "List[str]": TypeAdapter(list[str]),
    "List[List[int]]": TypeAdapter(list[list[int]]),
    "List[List[float]]": TypeAdapter(list[list[float]]),
    "List[List[str]]": TypeAdapter(list[list[str]]),
    "List[Union[int,float]]": TypeAdapter(list[int | float]),
    "Tuple[float,float]": TypeAdapter(tuple[float, float]),
    "List[Tuple[float,float]]": TypeAdapter(list[tuple[float, float]]),
    "List[Tuple[int,int]]": TypeAdapter(list[tuple[int, int]]),
}



def type_to_json_schema(raw_type: str) -> dict:
    pattern=r",\s*(?:optional\b|default\b)"
    base_type=re.split( pattern,raw_type,maxsplit=1)[0]
    type_key=re.sub(r"\s+", "", base_type )
    adapter=TYPE_ADAPTERS.get(type_key)
    if  adapter is None:
        raise ValueError(f"Unsupported type:{raw_type}")

    schema=adapter.json_schema()
    return schema

def convert_tool_schema(raw_tool: dict) -> dict:
    properties={}
    required=[]
    for parameter_name,parameter in raw_tool["parameters"].items():
        raw_type=parameter["type"]
        property_schema=type_to_json_schema(raw_type)
        property_schema["description"]=parameter["description"]
        if "default" in parameter:
            property_schema["default"]=parameter["default"]
        is_optional=re.search( r",\s*(?:optional|default)\b", raw_type ) is not None
        has_default="default" in parameter
        if not is_optional and not has_default:
            required.append(parameter_name)
        properties[parameter_name]=property_schema

    parameters_schema={"type":"object",
                       "properties":properties,
                       "required":required}
    function_schema={"name":raw_tool["name"],
                     "description":raw_tool["description"],
                     "parameters":parameters_schema}

    return {"type":"function",
            "function":function_schema}

def normalize_record(raw_record: dict) -> dict:
    record_id=raw_record.get("id")
    try:
        if record_id is None:
            raise ValueError("Invalid record id")
        raw_tools=json.loads(raw_record["tools"])
        raw_answers=json.loads(raw_record["answers"])
        if not isinstance(raw_tools, list):
            raise ValueError("Invalid tools format")
        if not isinstance(raw_answers, list):
            raise ValueError("Invalid answers format")
        tools=[convert_tool_schema(tool) for tool in raw_tools]
        tool_names={tool["function"]["name"] for tool in tools}

        tool_calls=[]
        for answer in raw_answers:
            name=answer["name"]
            arguments=answer["arguments"]
            if name not in tool_names:
                raise ValueError("Unknown tool name")
            if not isinstance(arguments,dict):
                raise ValueError("Invalid arguments format")

            function_call={"name":name,
                           "arguments":arguments}
            call={"type":"function",
                  "function":function_call}
            tool_calls.append(call)
        query = raw_record["query"]
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Invalid query format")

        user_message={"role":"user","content":query}
        assistant_message={"role":"assistant","tool_calls":tool_calls}
        messages=[user_message,assistant_message]

        sample_id=f"xlam-{record_id}"

        return {"sample_id":sample_id,
                "tools":tools,
                "messages":messages}
    except (ValueError,KeyError,TypeError,AttributeError) as error:
        raise ValueError(f"Sample {record_id}: {error}") from error




def prepare_records(
    raw_records: list[dict],
) -> tuple[list[dict], list[dict]]:
    normalized_records=[]
    rejected_records=[]

    for index, raw_record in enumerate(raw_records):
        try:
            record=normalize_record(raw_record)
            normalized_records.append(record)
        except ValueError as error:
            rejected_record={"row_index":index,
                             "source_id":raw_record.get("id"),
                             "error":str(error)}
            rejected_records.append(rejected_record)


    return normalized_records,rejected_records



def save_jsonl(records: list[dict], path: str | Path) -> None:
    file_path=Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path=None

    try:
        with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=file_path.parent,
                delete=False,
        ) as file:
            temporary_path=Path(file.name)
            for record in records:
                line = json.dumps(record, ensure_ascii=False) + "\n"
                file.write(line)
        temporary_path.replace(file_path)


    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)





def main():
    project_dir=Path(__file__).resolve().parents[1]
    data_path=project_dir/ "data" / "raw" / "xlam_function_calling_60k.json"
    output_dir=project_dir/ "data" / "processed"
    with data_path.open("r",encoding="utf-8") as file:
        raw_records=json.load(file)
    normalized_records, rejected_records = prepare_records(raw_records)

    assert len(normalized_records)+len(rejected_records)==len(raw_records)

    normalized_path=output_dir / "normalized.jsonl"
    rejected_path=output_dir / "rejected.jsonl"
    report_path=output_dir / "preparation_report.json"

    save_jsonl(normalized_records, normalized_path)
    save_jsonl(rejected_records, rejected_path)

    sample_ids={record["sample_id"] for record in normalized_records}
    report={"total_records":len(raw_records),
            "accepted_records":len(normalized_records),
            "rejected_records":len(rejected_records),
            "unique_sample_ids":len(sample_ids),
            "rejected_examples":rejected_records[:3]}

    with report_path.open("w",encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()









