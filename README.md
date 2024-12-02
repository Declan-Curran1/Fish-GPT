# Fish Species Classification with LLM and Image Encoder

This project trains a large language model (LLM) with an image encoder to classify fish species based on input images. The model is fine-tuned to not only identify the fish species but also provide a detailed explanation for its classification decision. You may ask why this was done instead of a normal CNN which is much faster and easier to train??? The answer is that the model also provides an explanation for why a given fish was chosen and suggests potential other species. Otherwise, a CNN would still be much faster and is probably the better option regardless. 

The fine-tuned model is available on [Hugging Face](https://huggingface.co/Declan1/llava-v1.6-mistral-7b-sydneyfish-a100).

---

## 📖 Overview

### Objectives
- Train an LLM with an image encoder to classify fish species.
- Fine-tune the model to generate explanations for its predictions.
- Use image data scraped from the internet and caption data curated from the Australian Museum and an LLM.

### Features
- **Fish Species Identification**: Predicts the fish species based on a given image.
- **Explainability**: Provides a detailed textual explanation for its predictions.
- **Fine-Tuning**: Optimized for 4 hours on an A100 GPU.

---

## 🛠️ Model Details

### Base Model
- **Architecture**: LLaVA (Large Language and Vision Assistant) v1.6
- **Model Backbone**: Mistral-7B
- **Fine-Tuned Dataset**: Sydney Fish Dataset
  - Image Data: Scraped from the internet.
  - Caption Data: Curated from the Australian Museum and enhanced using an LLM.

### Training Details
- **Training Duration**: 4 hours
- **Hardware**: A100 GPU
- **Optimizer**: AdamW
- **Learning Rate Scheduler**: Cosine Annealing

### Model Access
The trained model is hosted on Hugging Face:  
[https://huggingface.co/Declan1/llava-v1.6-mistral-7b-sydneyfish-a100](https://huggingface.co/Declan1/llava-v1.6-mistral-7b-sydneyfish-a100)

---

## 📂 Dataset

- **Image Data**: 
  - Source: Scraped from various online sources.
  - Format: JPEG/PNG images.
- **Caption Data**:
  - Source: Australian Museum's fish species database and an LLM for data augmentation.
  - Format: JSON with mappings between image IDs and captions.

---


![image](https://github.com/user-attachments/assets/342a158d-d9d9-46f8-9b51-4e7a9d3651ba)

