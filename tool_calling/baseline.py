from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tool_calling.data import load_jsonl
from tool_calling.evaluation import evaluate_response


def load_model_and_tokenizer(model_id: str, cache_dir: str | Path):
    tokenizer = AutoTokenizer.from_pretrained(model_id,
                                              cache_dir=cache_dir,
                                              local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(model_id,
                                               cache_dir=cache_dir,
                                               dtype=torch.bfloat16,
                                               device_map="cuda:0",
                                               attn_implementation="sdpa",
                                               local_files_only=True)

    model.eval()
    return model,tokenizer




def generate_from_messages(
    model,
    tokenizer,
    messages: list[dict],
    tools: list[dict],
    max_new_tokens: int = 128
) -> str:


    inputs=tokenizer.apply_chat_template(messages,
                                         tools=tools,
                                         add_generation_prompt=True,
                                         tokenize=True,
                                         return_dict=True,
                                         return_tensors="pt")
    inputs=inputs.to(model.device)
    prompt_length=inputs["input_ids"].shape[1]

    with torch.inference_mode():
        output_ids=model.generate(**inputs,
                                  max_new_tokens=max_new_tokens,
                                  do_sample=False)

    new_token_ids=output_ids[0,prompt_length:]

    generated_text=tokenizer.decode(new_token_ids,skip_special_tokens=True)
    return generated_text

def generate_completion(
    model,
    tokenizer,
    record: dict,
    max_new_tokens: int = 128
) -> str:
    return generate_from_messages(
        model,
        tokenizer,
        messages=record["messages"][:-1],
        tools=record["tools"],
        max_new_tokens=max_new_tokens
    )

def main():
    project_dir = Path(__file__).resolve().parents[1]
    model_id="Qwen/Qwen2.5-0.5B-Instruct"
    cache_dir=project_dir / "models"/"hf"

    records=load_jsonl(project_dir/"data"/"examples"/"tool_calls_demo.jsonl")

    record=records[0]
    model,tokenizer=load_model_and_tokenizer(model_id, cache_dir)

    generated_text=generate_completion(model, tokenizer, record)



    raw_expected_calls = record["messages"][-1].get("tool_calls", [])

    expected_calls=[call["function"] for call in raw_expected_calls]

    result=evaluate_response(generated_text,expected_calls)

    print(f"模型运行位置:{model.device}")
    print(f"模型使用的数字精度:{model.dtype}")
    print(f"模型实际生成回答:{generated_text}")
    print(f"标准答案:{expected_calls}")
    print(f"评测结果：{result}")


if __name__ == "__main__":
    main()
