# Sentence-prediction
🚀 GPT-2 LoRA Fine-Tuning on WikiText-2
This project demonstrates parameter-efficient fine-tuning (LoRA) of a GPT-2 model using the Hugging Face ecosystem. We train on the WikiText-2 dataset and compare text generation results before and after fine-tuning.

By leveraging LoRA, we reduce trainable parameters and speed up training while achieving competitive results for text generation.

📂 Project Overview
Base Model: distilgpt2 (lightweight GPT-2 variant)

Fine-tuning Method: LoRA (peft library)

Dataset: WikiText-2

Frameworks: PyTorch, Hugging Face transformers, datasets

Goal: Show the benefits of LoRA fine-tuning vs. vanilla GPT-2

⚙️ Installation
Clone the repo and install dependencies:

git clone https://github.com/DabestCode/gpt2-lora-finetune.git
cd gpt2-lora-finetune

pip install -r requirements.txt
Requirements (requirements.txt):

torch
transformers
datasets
peft
🏃‍♂️ Usage
1️⃣ Run the notebook
Open the GPT_Training.ipynb notebook in Jupyter or Colab and execute all cells.

Or run as a Python script:

python GPT_Training.py
📊 Workflow
Load GPT-2 model and tokenizer

Apply LoRA configuration for parameter-efficient fine-tuning

Load & preprocess WikiText-2 dataset

Train with Hugging Face Trainer

Generate text before and after fine-tuning to compare improvements

🔍 Example Results
Before Fine-tuning:

Prompt: Artificial intelligence will
Output: Artificial intelligence will be able to do things.
After Fine-tuning:

Prompt: Artificial intelligence will
Output: Artificial intelligence will continue to evolve, shaping industries and daily life in unexpected ways.
📁 Project Structure
bash
Copy
Edit
├── GPT_Training.ipynb     # Main training notebook
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
🚀 Future Improvements
Train on a larger dataset for better generalization

Experiment with different LoRA configurations

Add evaluation metrics (perplexity, BLEU score, etc.)

Save & load LoRA adapters for deployment

