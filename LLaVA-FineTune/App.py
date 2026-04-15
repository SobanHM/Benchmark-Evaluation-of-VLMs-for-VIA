import torch
import gradio as gr
from transformers import LlavaForConditionalGeneration, AutoProcessor
from peft import PeftModel
from PIL import Image
from datetime import datetime

# MODEL CONFIGURATION
model_id = "llava-hf/llava-1.5-7b-hf"
adapter_path = "./llava-via-final-acc4" # lora finetuned model adapter

print("Loading base model...")
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto"
)

print("Loading fine-tuned adapter...")
model = PeftModel.from_pretrained(model, adapter_path)

processor = AutoProcessor.from_pretrained(model_id)

print("Model loaded successfully.")


# INFERENCE FUNCTION
def generate_guidance(image):
    try:
        if image is None:
            return "Please upload an image first.", "No image provided"

        prompt = (
            "USER: <image>\n"
            "I am visually impaired person and I can not see anything. The given image shows the environment in front of me. Please, describe the scene completely and provide necessary navigation guidance.\n"
            "ASSISTANT:"
        )

        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to("cuda", torch.float16)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.6,
                top_p=0.9
            )

        response = processor.decode(
            output[0],
            skip_special_tokens=True
        )

        final_response = response.split("ASSISTANT:")[-1].strip()

        status = f"Generated successfully at {datetime.now().strftime('%H:%M:%S')}"

        return final_response, status

    except Exception as e:
        return f"Error: {str(e)}", "Generation failed"


# =====================================
# CUSTOM CSS
# =====================================
custom_css = """
body {
    font-family: 'Inter', sans-serif;
}

.main-header {
    text-align: center;
    font-size: 34px;
    font-weight: 700;
    margin-bottom: 8px;
}

.sub-header {
    text-align: center;
    font-size: 16px;
    color: #666;
    margin-bottom: 25px;
}

.card {
    border-radius: 22px !important;
    padding: 20px !important;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    background: white;
}

.status-box {
    font-size: 14px;
    color: #444;
}
"""


# =====================================
# UI DESIGN
# =====================================
with gr.Blocks(
    theme=gr.themes.Soft(),
    css=custom_css,
    title="Scene Narrator Assistant"
) as demo:

    gr.HTML("""
        <div class="main-header">
            AI Scene Narrator & Navigation Guidance Assistant
        </div>
        <div class="sub-header">
            Human-Centered Vision-Language Interface for Assistive Navigation
        </div>
    """)

    with gr.Row():
        # LEFT PANEL
        with gr.Column(scale=1):
            with gr.Group(elem_classes="card"):
                gr.Markdown("## 📷 Upload Test Image")

                image_input = gr.Image(
                    type="pil",
                    label="Environment Scene",
                    height=350
                )

                with gr.Row():
                    generate_btn = gr.Button(
                        "Generate Scene Narration + Navigation Guidance",
                        variant="primary",
                        size="lg"
                    )

                    clear_btn = gr.Button(
                        "🗑 Clear",
                        size="lg"
                    )

        # RIGHT PANEL
        with gr.Column(scale=1):
            with gr.Group(elem_classes="card"):
                gr.Markdown("## Model Response")

                output_text = gr.Textbox(
                    label="Generated Navigation Guidance",
                    lines=16,
                    show_copy_button=True
                )

                status_text = gr.Textbox(
                    label="System Status",
                    interactive=False
                )

    # FOOTER INFO CARD
    with gr.Row():
        with gr.Group(elem_classes="card"):
            gr.Markdown("""
            ### ℹ Research Demo Information
            - Model: Fine-Tuned LLaVA 1.5 7B
            - Task: Scene Understanding + Navigation Guidance
            - Domain: Assistive AI for Visually Impaired Users
            """)

    # BUTTON ACTIONS
    generate_btn.click(
        fn=generate_guidance,
        inputs=image_input,
        outputs=[output_text, status_text]
    )

    clear_btn.click(
        fn=lambda: (None, "", ""),
        outputs=[image_input, output_text, status_text]
    )

demo.queue().launch(
    server_name="127.0.0.1",
    server_port=7860,
    debug=True
)
