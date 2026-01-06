# IndicTrans2-FineTune-LoRA
Fine-tuning IndicTrans2 (1B) on Hindi to Gujarati, Marathi, Kashmiri, and Telugu using LoRA and 4-bit quantization for domain-specific translation.
🇮🇳 IndicTrans2 Fine-Tuning (Hindi → Guj/Mar/Kas/Tel)



This repository contains the code, data processing scripts, and fine-tuning logs for adapting the IndicTrans2-1B model for specific Indic language pairs using LoRA (Low-Rank Adaptation).

📌 Project Overview
The goal of this project is to improve translation quality for domain-specific Hindi text into four target languages:

Gujarati (guj_Gujr)
Marathi (mar_Marq)
Kashmiri (kas_Arab)
Telugu (tel_Telu)
We utilize 4-bit quantization (BitsAndBytes) and PEFT/LoRA to fine-tune the 1B parameter model efficiently on a single NVIDIA A100 GPU.

📊 Results & Performance
Language Pair	Status	Dataset Size	Best BLEU	Best chrF
Hindi → Gujarati	✅ Complete	~38k	53.01	74.97
Hindi → Marathi	✅ Complete	~32k	47.64	74.41
Hindi → Kashmiri	✅ Complete	~34k	18.74	47.55
Hindi → Telugu	⏳ Training	~44k	Pending	Pending


📂 Repository Structure
├── fine_tune_hin_guj/       # Output & Scripts for Gujarati
├── fine_tune_hin_mar/       # Output & Scripts for Marathi
├── fine_tune_hin_kas/       # Output & Scripts for Kashmiri
├── fine_tune_hin_tel/       # Output & Scripts for Telugu
├── fetch_parallel_results.sh # Utility to download models
└── process_data.py           # Data preprocessing logic
🚀 Setup & Installation
Clone the repository:

git clone https://github.com/bidyut2611/IndicTrans2-FineTuning.git
cd IndicTrans2-FineTuning
Install Dependencies:

pip install torch transformers peft bitsandbytes scipy
# Install IndicTransToolkit for robust preprocessing
git clone https://github.com/AI4Bharat/IndicTransToolkit.git
cd IndicTransToolkit
pip install -e .
cd ..
💡 How to Run Inference
Use the following Python script to translate text using your fine-tuned models.

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
from IndicTransToolkit import IndicProcessor


# 1. Configuration
# Choose language code: "guj_Gujr", "mar_Marq", "kas_Arab", "tel_Telu"
target_lang = "guj_Gujr" 
adapter_path = "fine_tune_hin_guj/results/output" # Path to your fine-tuned adapter
# 2. Load Models
model_name = "ai4bharat/indictrans2-indic-indic-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")


# Load LoRA Adapter
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()


# 3. Translate
ip = IndicProcessor(inference=True)
input_text = ["सरकार ने किसानों के लिए नई योजना शुरू की है।"] # Example Hindi Sentence
batch = ip.preprocess_batch(input_text, src_lang="hin_Deva", tgt_lang=target_lang)
inputs = tokenizer(batch, return_tensors="pt", padding=True).to("cuda")
with torch.no_grad():
    generated = model.generate(**inputs, max_length=128, num_beams=5)
decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
translations = ip.postprocess_batch(decoded, lang=target_lang)
print("Translation:", translations[0])
📜 Dataset Information
The dataset consists of domain-specific sentence pairs (Government, Education, Agriculture) cleaned and split into 90% Training and 10% Validation sets.

🙌 Acknowledgements
AI4Bharat for the IndicTrans2 model.
Hugging Face for the PEFT and Transformers libraries.
