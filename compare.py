import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "./models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
LORA_MODEL = "./models/deepseek_r1_1.5b_lora/best_model"

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_base():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16).to(device)
    return tokenizer, model


def load_lora():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, LORA_MODEL)
    model = model.to(device)
    return tokenizer, model


def generate_answer(tokenizer, model, query):
    prompt = f"用户：{query}\n"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def check_reasoning(text):
    return "<reasoning>" in text.lower() or "推理" in text


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compare.py \"问题内容\"")
        sys.exit(1)

    query = sys.argv[1]

    print("\n=== Loading Models... ===")
    base_tok, base_model = load_base()
    lora_tok, lora_model = load_lora()

    print("\n=== Base Model Output ===")
    base_out = generate_answer(base_tok, base_model, query)
    print(base_out)
    print("带推理链?", "✅" if check_reasoning(base_out) else "❌")

    print("\n=== LoRA Distilled Model Output ===")
    lora_out = generate_answer(lora_tok, lora_model, query)
    print(lora_out)
    print("带推理链?", "✅" if check_reasoning(lora_out) else "❌")

    print("\n=== Comparison Summary ===")
    print(f"Base length: {len(base_out)} chars")
    print(f"LoRA length: {len(lora_out)} chars")
    print("LoRA 是否更像老师输出:", "✅是" if len(lora_out) > len(base_out) else "⚠️暂时看不出来")
