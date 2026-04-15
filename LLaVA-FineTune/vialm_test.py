# img_input = r"C:\Users\soban\OneDrive\VLM_Evaluation_Benchmark_Dataset(FYP)\VIA_Finetune_Dataset\finetune_evaluation_vialm\vialm_ph_163.jpg"
# query_input = "I am visually impaired person and I need your assistance to complete the task of 'Approaching the target object'. Here is where I stand, and the scene depicted in the image is the view in front of me.The target object I want to approach is chair. Could you please guide me to it using tactile cues based on the scene shown in the image?"
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor
from peft import PeftModel
from PIL import Image
import json
import os

# --- Configuration ---
model_id = "llava-hf/llava-1.5-7b-hf"
adapter_path = "./llava-via-final-acc4"
query_json_path = "vialm_query.json"
image_folder = r"C:\Users\soban\OneDrive\VLM_Evaluation_Benchmark_Dataset(FYP)\VIA_Finetune_Dataset\finetune_evaluation_vialm"
output_results_path = "vialm_images_test_results.json"

print("Loading model and adapter..")
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, adapter_path)
processor = AutoProcessor.from_pretrained(model_id)

# Load the queries from your JSON file
with open(query_json_path, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

final_results = []

print(f"Starting batch processing of {len(test_data)} images..")

for entry in test_data:
    image_filename = entry['image']
    query_text = entry['query']
    image_path = os.path.join(image_folder, image_filename)

    if not os.path.exists(image_path):
        print(f"Skipping: {image_filename} (not found in folder)")
        continue

    print(f"Processing ID {entry['id']}: {image_filename}...")

    try:
        # Prepare Image
        image = Image.open(image_path).convert("RGB")

        # Format prompt exactly as used in training
        prompt = f"USER: <image>\n{query_text}\nASSISTANT:"

        inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda", torch.float16)

        # Generate Response
        output = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

        full_response = processor.decode(output[0], skip_special_tokens=True)
        clean_response = full_response.split("ASSISTANT:")[-1].strip()

        # Save result
        final_results.append({
            "image": image_filename,
            "query": query_text,
            "response": clean_response
        })

    except Exception as e:
        print(f"Error processing {image_filename}: {e}")

# Save all results to the new JSON file
with open(output_results_path, "w", encoding='utf-8') as f:
    json.dump(final_results, f, indent=4, ensure_ascii=False)

print(f"\nDone! Results saved to {output_results_path}")
