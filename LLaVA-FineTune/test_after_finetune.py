import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor
from peft import PeftModel
from PIL import Image
import os

# config lora adapther path
model_id = "llava-hf/llava-1.5-7b-hf"
adapter_path = "./llava-via-final-acc4"
image_folder = r"C:\Users\soban\OneDrive\VLM_Evaluation_Benchmark_Dataset(FYP)\VIA_Finetune_Dataset\finetuneEVALUATION"
output_file = "test_results.txt"

print(f"Loading base model: {model_id}..")
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto"
)

print(f"Loading Fine-Tuned Adapter from {adapter_path}..")
model = PeftModel.from_pretrained(model, adapter_path)
processor = AutoProcessor.from_pretrained(model_id)


def generate_guidance(image_path):
    image = Image.open(image_path).convert("RGB")

    # Use the exact same prompt format that was used during training
    # prompt = "USER: <image>\nI am visually impaired and need your assistance to complete the task of 'Approaching the target object'. Provide tactile cues to guide me.\nASSISTANT:"
    prompt = "USER: <image>\n I am visually impaired person and I can not see anything. The given image shows the environment in front of me. Please, describe the scene completely and provide necessary navigation guidance."
    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda", torch.float16)

    output = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    return processor.decode(output[0], skip_special_tokens=True).split("ASSISTANT:")[-1].strip()


# 2 Run testing on the folder
with open(output_file, "w") as f:
    for img_name in os.listdir(image_folder):
        if img_name.endswith((".jpg", ".png", ".jpeg")):
            print(f"Processing {img_name}...")
            path = os.path.join(image_folder, img_name)
            response = generate_guidance(path)

            f.write(f"Image: {img_name}\nResponse: {response}\n")
            f.write("-" * 50 + "\n")

print(f"Testing complete! Results saved to {output_file}")
