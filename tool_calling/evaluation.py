import re
import json

def parse_tool_calls(text: str) -> list[dict]:
    calls = []
    pattern=r"<tool_call>\s*(.*?)\s*</tool_call>"
    blocks=re.findall(pattern,text,flags=re.DOTALL)
    start_count=text.count("<tool_call>")
    end_count=text.count("</tool_call>")
    if not start_count==end_count ==len(blocks):
        raise ValueError("Incomplete tool call tags")

    for block in blocks:
        try:
            call = json.loads(block)
        except json.JSONDecodeError:
            raise ValueError("Invalid json")

        if not isinstance(call, dict):
            raise ValueError("Expected a dict")
        name=call.get("name")
        arguments=call.get("arguments")

        if not isinstance(name, str) or not name.strip() :
            raise ValueError("Invalid tool name")

        if not isinstance(arguments, dict):
            raise ValueError("Arguments must be a dict")

        calls.append({"name":name,"arguments":arguments})


    return calls


def evaluate_tool_calls(
    predicted_calls: list[dict],
    expected_calls: list[dict],
) -> dict:

    predicted_names=[predicted_call["name"]for predicted_call in predicted_calls]
    expected_names=[expected_call["name"]for expected_call in expected_calls]

    predicted_arguments=[predicted_call["arguments"] for predicted_call in predicted_calls]
    expected_arguments=[expected_call["arguments"] for expected_call in expected_calls]

    tool_names_match=predicted_names==expected_names
    arguments_match=predicted_arguments==expected_arguments

    exact_match = tool_names_match and arguments_match

    return {"tool_names_match":tool_names_match,
            "arguments_match":arguments_match,
            "exact_match":exact_match}


def evaluate_response(
    text: str,
    expected_calls: list[dict],
) -> dict:

    try:
        predicted_calls=parse_tool_calls(text)
    except ValueError as error:
        return {"parse_ok":False,
                "predicted_calls":None,
                "tool_names_match":False,
                "arguments_match":False,
                "exact_match":False,
                "error":str(error)}

    metrics=evaluate_tool_calls(predicted_calls,expected_calls)
    result={"parse_ok":True,
            "predicted_calls":predicted_calls,
            "error":None}

    result.update(metrics)

    return result