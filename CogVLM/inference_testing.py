"""
CLI demo for CogAgent / CogVLM (official version)
Supports 4-bit quantization + multi-GPU auto-sharding.
Best for image captioning and visual question answering.
"""

import argparse
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer

# -----------------------------
# CLI arguments
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--quant", choices=[4], type=int, default=4, help="Quantization bits (4-bit recommended)")
parser.add_argument("--from_pretrained", type=str, default="THUDM/cogvlm-chat-hf", help="Model path")
parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help="Tokenizer path")
parser.add_argument("--fp16", action="store_true", help="Use FP16")
parser.add_argument("--bf16", action="store_true", help="Use BF16 if available")
args = parser.parse_args()

# -----------------------------
# Setup device and dtype
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if args.bf16:
    torch_type = torch.bfloat16
else:
    torch_type = torch.float16

print(f"======== Using torch type: {torch_type} on device(s): {DEVICE} ========\n")

# -----------------------------
# Load tokenizer
# -----------------------------
tokenizer = LlamaTokenizer.from_pretrained(args.local_tokenizer)

# -----------------------------
# Load model (4-bit + multi-GPU)
# -----------------------------
print("\n Loading CogVLM model (4-bit quantized + auto GPU mapping)\n")
if args.quant:
    model = AutoModelForCausalLM.from_pretrained( args.from_pretrained, torch_dtype=torch_type, low_cpu_mem_usage=True, load_in_4bit=True, device_map="auto", trust_remote_code=True
    ).eval()
else:
    model = AutoModelForCausalLM.from_pretrained(
        args.from_pretrained,
        torch_dtype=torch_type, low_cpu_mem_usage=True, device_map="auto", trust_remote_code=True
    ).eval()

text_only_template = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions. "
    "USER: {} ASSISTANT:"
)

# -----------------------------
# Interactive loop
# -----------------------------
while True:
    image_path = input("  Enter image path (or press Enter for text-only chat): ").strip()
    if image_path == "":
        print("No image provided — running text-only conversation mode.")
        image = None
        text_only_first_query = True
    else:
        image = Image.open(image_path).convert("RGB")

    history = []

    while True:
        query = input("\n Human: ").strip()
        if query.lower() in ["exit", "quit", "clear"]:
            print("Conversation cleared / exited.\n")
            break

        # build query text
        if image is None:
            if text_only_first_query:
                query = text_only_template.format(query)
                text_only_first_query = False
            else:
                old_prompt = ""
                for old_query, response in history:
                    old_prompt += old_query + " " + response + "\n"
                query = old_prompt + f"USER: {query} ASSISTANT:"

        # build model input
        if image is None:
            input_by_model = model.build_conversation_input_ids(
                tokenizer, query=query, history=history, template_version="base"
            )
        else:
            input_by_model = model.build_conversation_input_ids(
                tokenizer, query=query, history=history, images=[image]
            )

        inputs = {
            "input_ids": input_by_model["input_ids"].unsqueeze(0).to(model.device),
            "token_type_ids": input_by_model["token_type_ids"].unsqueeze(0).to(model.device),
            "attention_mask": input_by_model["attention_mask"].unsqueeze(0).to(model.device),
            "images": [[input_by_model["images"][0].to(model.device).to(torch_type)]] if image is not None else None,
        }

        if "cross_images" in input_by_model and input_by_model["cross_images"]:
            inputs["cross_images"] = [[input_by_model["cross_images"][0].to(model.device).to(torch_type)]]

        # generation parameters
        gen_kwargs = {
            "max_new_tokens": 256,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9
        }

        print("\nGenerating response...\n")
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split("</s>")[0]

        print(f"Cog: {response}\n")
        history.append((query, response))
