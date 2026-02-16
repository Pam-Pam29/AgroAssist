 import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# -------------------------
# CONFIG
# -------------------------
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "./agroassist_exp4_model"

SYSTEM_MESSAGE = "You are AgroAssist, an expert agricultural advisor providing practical farming guidance."

# -------------------------
# DEVICE
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# LOAD MODEL
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32
)

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.to(device)
model.eval()

# -------------------------
# CHAT FUNCTION
# -------------------------
def agroassist_chat(message, history):

    prompt = f"<|system|>\n{SYSTEM_MESSAGE}</s>\n<|user|>\n{message}</s>\n<|assistant|>\n"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("<|assistant|>")[-1].strip()

    return response

# -------------------------
# INTERFACE
# -------------------------
demo = gr.ChatInterface(
    fn=agroassist_chat,
    title="🌾 AgroAssist: Intelligent Agricultural Advisory System",
    description="AI-powered agricultural advisory system fine-tuned for farming guidance."
)

demo.launch()
