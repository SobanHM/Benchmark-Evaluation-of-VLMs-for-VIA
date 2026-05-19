"""
Batch Inference for CogVLM (4-bit, Multi-GPU)
Runs captioning/VQA on ALL images in a given folder.
Saves results automatically to results.json
"""

import os
import argparse
import torch
import pandas as pd
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer


# -----------------------------
# CLI arguments
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--images_dir", type=str, required=True,
                    help="Path to folder where your dataset images exist")
parser.add_argument("--question", type=str, default="Describe this image in detail.",
                    help="Question to ask model for every image")
parser.add_argument("--quant", type=int, default=4)
parser.add_argument("--from_pretrained", type=str, default="THUDM/cogvlm-chat-hf")
parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5")
args = parser.parse_args()


# -----------------------------
# Setup dtype
# -----------------------------
torch_type = torch.float16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n======== Using dtype {torch_type} on {DEVICE} ========\n")


# -----------------------------
# Load tokenizer
# -----------------------------
tokenizer = LlamaTokenizer.from_pretrained(args.local_tokenizer)


# -----------------------------
# Load model (4-bit + multi-GPU)
# -----------------------------
print("\nLoading CogVLM model in 4-bit with multi-GPU...\n")

model = AutoModelForCausalLM.from_pretrained(
    args.from_pretrained,
    torch_dtype=torch_type,
    low_cpu_mem_usage=True,
    load_in_4bit=True,
    device_map="auto",
    trust_remote_code=True
).eval()


# -----------------------------
# Get all images from directory
# -----------------------------
valid_ext = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
image_files = [f for f in os.listdir(args.images_dir)
               if f.lower().endswith(tuple(valid_ext))]

if len(image_files) == 0:
    print("No images found in the directory.")
    exit()

print(f"Found {len(image_files)} images. Starting batch inference...\n")


# output storage
results = []


# process each image
for idx, img_name in enumerate(image_files):
    img_path = os.path.join(args.images_dir, img_name)

    try:
        image = Image.open(img_path).convert("RGB")
    except:
        print(f"Skipping unreadable file: {img_name}")
        continue

    print(f"\n[{idx+1}/{len(image_files)}] Processing: {img_name}")

    # Build input for multimodal model
    input_pack = model.build_conversation_input_ids(
        tokenizer,
        query=args.question,
        history=[],
        images=[image],
    )

    inputs = {
        "input_ids": input_pack["input_ids"].unsqueeze(0).to(model.device),
        "token_type_ids": input_pack["token_type_ids"].unsqueeze(0).to(model.device),
        "attention_mask": input_pack["attention_mask"].unsqueeze(0).to(model.device),
        "images": [[input_pack["images"][0].to(model.device).to(torch_type)]],
    }

    if "cross_images" in input_pack and input_pack["cross_images"]:
        inputs["cross_images"] = [[input_pack["cross_images"][0].to(model.device).to(torch_type)]]

    # Generation settings
    gen_kwargs = {
        "max_new_tokens": 300,
        "temperature": 0.2,
        "top_p": 0.9,
    }

    with torch.no_grad():
        output_tokens = model.generate(**inputs, **gen_kwargs)
        output_tokens = output_tokens[:, inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        response = response.split("</s>")[0]

    print(f"Output: {response}")

    # Save result
    results.append({
        "image": img_name,
        "question": args.question,
        "answer": response
    })


# saving results to csv
df = pd.DataFrame(results)
save_path = "data/cogvlm_results_supermarket.json"
df.to_csv(save_path, index=False)

print(f"\n Batch inference completed!")
print(f"Results saved to: {save_path}\n")
