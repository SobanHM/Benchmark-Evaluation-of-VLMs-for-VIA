# Benchmarking VLMs for Visual Impairment Assistance in Pakistan

### Fine-tuning LLaVA for Scene Narration & Navigation in Local Environments

<p align="center">
  <img src="https://github.com/user-attachments/assets/53b5a86c-48e6-4ec6-9750-26e0dcf98124" width="850" alt="Pipeline">
</p>

---

## 📌 Overview

Assistive AI systems for visually impaired individuals are primarily evaluated on **Western-centric datasets**, making them less reliable in developing regions such as Pakistan where environments, objects, store layouts, language usage, and navigation cues differ significantly.

This project introduces a **localized Pakistani benchmark dataset** and evaluates state-of-the-art **Vision-Language Models (VLMs)** for **Visual Impairment Assistance (VIA)** tasks including:

* Scene Narration
* Navigation Guidance
* Spatial Awareness
* Accessibility-Centered Descriptions
* Safety-Oriented Environmental Understanding

We further fine-tune **LLaVA v1.5** using **LoRA (Low-Rank Adaptation)** to improve contextual narration quality and reduce hallucinations in local environments.

---

# 🎯 Objectives

* Build a localized VIA benchmark dataset from Pakistani environments
* Evaluate multiple open-source VLMs under zero-shot settings
* Design human-centered assistive narration prompts
* Measure semantic quality, hallucination, and navigational specificity
* Fine-tune the best-performing model using LoRA
* Compare zero-shot vs fine-tuned performance

---

# 🧠 Evaluated Vision-Language Models

| Model      | Type            | Setting                |
| ---------- | --------------- | ---------------------- |
| LLaVA v1.5 | Open-source VLM | Zero-shot              |
| CogVLM     | Open-source VLM | Zero-shot              |
| BLIP-2     | Open-source VLM | Zero-shot              |
| LLaVA v1.5 | Open-source VLM | Fine-tune              |
---

# 🏠 Localized Pakistani Dataset

## Dataset Characteristics

* **400 real-world images**
* Captured from:

  * Pakistani homes
  * Grocery stores
  * Supermarkets
  * Indoor daily-life environments
* Human-written assistive narration annotations
* Focused on:

  * Navigation
  * Obstacle awareness
  * Spatial relationships
  * Safety cues
  * Object accessibility

---

# 🔄 Project Pipeline

```text
Data Collection
      ↓
Human Assistive Annotation
      ↓
Zero-shot Evaluation of VLMs
      ↓
Metric-based Benchmarking
      ↓
Best Model Selection
      ↓
LoRA Fine-tuning
      ↓
Re-evaluation & Comparison
```

---

# 📊 Evaluation Metrics

We performed multi-level evaluation using lexical, semantic, grounding, and hallucination-based metrics.

## Lexical Metrics

* BLEU-1
* ROUGE-L
* METEOR

## Semantic Metrics

* SBERT Similarity
* BERTScore

## Visual Grounding Metrics

* CLIPScore
* PickScore

## Hallucination & Reliability

* CHAIR
* POPE

## Accessibility-Oriented Metric

* VIA-SPECS (proposed specificity metric)

---

# 🧪 Zero-Shot Benchmark Results

| Model          | BLEU-1    | BERTScore | VIA-SPECS |
| -------------- | --------- | --------- | --------- |
| **LLaVA v1.5** | **0.268** | 0.863     | 1.62      |
| CogVLM         | 0.245     | **0.865** | **2.06**  |
| BLIP-2         | 0.011     | 0.848     | 0.37      |

### Key Observation

LLaVA v1.5 achieved the best balance among semantic alignment, descriptive quality, and consistency in assistive narration, making it the selected model for fine-tuning.

---

# ⚙️ LoRA Fine-Tuning

## Fine-tuning Configuration

| Parameter          | Value         |
| ------------------ | ------------- |
| Base Model         | LLaVA v1.5-7B |
| Fine-tuning Method | LoRA / PEFT   |
| Training Pairs     | 650           |
| Epochs             | 3             |
| Learning Rate      | 2e-4          |
| Quantization       | 4-bit         |
| Framework          | PyTorch       |

---

# 📈 Fine-Tuned LLaVA Results

| Metric           | Zero-Shot | Fine-Tuned        |
| ---------------- | --------- | ----------------- |
| SBERT Similarity | 0.608     | **0.712**         |
| SPECS-VIA        | 1.32      | **3.28**          |
| BLEU-1           | 0.268     | **0.31**          |
| METEOR           | 0.245     | **0.317**         |
| ROUGE-L          | 0.214     | **0.276**         |
| CHAIR-i          | Higher    | **41% Reduction** |

---

# 🔍 Key Contributions

## 1. Localized Pakistani VIA Dataset

A real-world assistive benchmark reflecting local indoor environments and navigation challenges.

## 2. Human-Centered Assistive Narration

Narration format designed specifically for visually impaired users with emphasis on:

* Safety
* Navigation
* Spatial guidance
* Accessibility

## 3. VIA-SPECS Metric

A custom specificity metric to evaluate navigational detail density and assistive usefulness.

## 4. Multi-Level Benchmarking

Comprehensive evaluation spanning:

* Lexical quality
* Semantic similarity
* Visual grounding
* Hallucination auditing
* Accessibility relevance

## 5. Efficient Domain Adaptation

LoRA fine-tuning significantly improved local scene understanding while remaining computationally efficient.

---

# 🛠️ Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* PEFT / LoRA
* LLaVA
* CogVLM
* BLIP-2
* SBERT
* BERTScore
* spaCy
* JSONL
* 4-bit Quantization

---

# 📂 Repository Structure

```bash
├── dataset/
├── annotations/
├── evaluation/
├── metrics/
├── fine_tuning/
├── inference/
├── results/
├── notebooks/
└── README.md
```

---

# 🚀 Future Work

* Real-time mobile deployment
* Urdu multilingual narration
* Outdoor navigation scenes
* OCR integration for signs & labels
* Audio-guided navigation assistant
* Lightweight on-device VLM inference

---

# 👨‍💻 Team

| Name              | ID          |
| ----------------- | ----------- |
| Soban Hussain     | 023-22-0116 |
| Praih Alias Faiza | 023-22-0055 |
| Tasmia            | 023-22-0051 |

---

# 👨‍🏫 Supervisor

**Dr. Mohammad Asif Khan**
Department of Computer Science
Sukkur IBA University

---

# 🎓 Academic Information

* Program: BS Computer Science
* Specialization: Software Engineering
* Final Year Project (FYP)
* Sukkur IBA University
* 2026

---

# 📜 Citation

```bibtex
@project{SceneNarrator2026,
  title={Benchmarking Vision-Language Models for Visual Impairment Assistance in Pakistan},
  author={Soban Hussain and Praih Alias Faiza and Tasmia},
  year={2026},
  institution={Sukkur IBA University}
}
```

---

# ⭐ Acknowledgment

Special thanks to our supervisor, faculty members, and participants who contributed to dataset creation, annotation, and evaluation.

---

# 📬 Contact

For collaboration, research discussion, or academic queries:

* Soban Hussain
* Sukkur IBA University
* Department of Computer Science
