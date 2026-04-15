import os
import json
import torch
from PIL import Image
from datasets import Dataset
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.nn.utils.rnn import pad_sequence

model_id = "llava-hf/llava-1.5-7b-hf"
data_path = r"C:\Users\soban\OneDrive\VLM_Evaluation_Benchmark_Dataset(FYP)\VIA_Finetune_Dataset\finetuning_annotation_schema\LLaVA_VIA_finetune_schema_650_samples.json"
image_folder = r"C:\Users\soban\OneDrive\VLM_Evaluation_Benchmark_Dataset(FYP)\VIA_Finetune_Dataset\LLava_VIA_Finetune_Dataset"

print("loading model..")
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16
)
model.config.use_cache = False
processor = AutoProcessor.from_pretrained(model_id)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

print("loading dataset..")
with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)


def preprocess(example):
    image_path = os.path.join(image_folder, example["image"])

    # We strip the existing <image> tag from the value and
    # force it to the very front of the USER prompt.
    user_value = example['conversation'][0]['value'].replace("<image>", "").strip()
    gpt_value = example['conversation'][1]['value']

    # Standardized LLaVA 1.5 format: <image> at the start
    full_prompt = f"USER: <image>\n{user_value}\nASSISTANT: {gpt_value}</s>"

    return {
        "full_prompt": full_prompt,
        "image_path": image_path
    }


dataset = dataset.map(preprocess, remove_columns=dataset.column_names)


def data_collator(batch):
    images = [Image.open(item["image_path"]).convert("RGB") for item in batch]
    texts = [item["full_prompt"] for item in batch]

    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1280  # increased to handle long tactile instructions
    )

    # Create labels
    labels = inputs["input_ids"].clone()

    # Mask padding tokens so they don't affect loss
    if processor.tokenizer.pad_token_id is not None:
        labels[labels == processor.tokenizer.pad_token_id] = -100

    # Important: Ensure pixel_values are in the right dtype for 4-bit training
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
    inputs["labels"] = labels

    return inputs


training_args = TrainingArguments(
    output_dir="./llava-via-experiment-acc4",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    remove_unused_columns=False,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

print("Starting training..")
trainer.train()

model.save_pretrained("./llava-via-final-acc4")
processor.save_pretrained("./llava-via-final-acc4")
# model.save_pretrained("./llava-via-final")
# processor.save_pretrained("./llava-via-final")
print("Training complete!")
