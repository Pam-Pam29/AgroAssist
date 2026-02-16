import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import re

print("="*80)
print("LOADING AGROASSIST MODEL")
print("="*80)

# System message for agricultural expertise
SYSTEM_MESSAGE = """You are AgroAssist, an expert agricultural advisor. Provide accurate, practical advice on farming, crops, soil management, pest control, and agricultural best practices. Keep responses concise, actionable, and based on scientific principles."""

# Load model and tokenizer
print("\nLoading base model and tokenizer...")
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

print("Loading LoRA adapter from local directory...")
adapter_path = "./agroassist_model"

if os.path.exists(adapter_path):
    print(f"✓ Found adapter at {adapter_path}")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        device_map="auto"
    )
    
    print("Merging adapter with base model...")
    model = model.merge_and_unload()
    model.eval()
    print("✓ Model loaded successfully!")
else:
    print(f"⚠️ Adapter not found at {adapter_path}")
    print("Available files:", os.listdir("."))
    print("Using base model without fine-tuning...")
    model = base_model
    model.eval()

print("="*80)

def clean_response(text):
    """
    Clean up model output to remove repetitions and artifacts
    """
    # Remove special tokens
    text = re.sub(r'<\|.*?\|>', '', text)
    
    # Remove repetitive dots (more than 2 in a row)
    text = re.sub(r'\.{3,}', '.', text)
    
    # Remove repetitive spaces
    text = re.sub(r' {2,}', ' ', text)
    
    # Remove repetitive punctuation
    text = re.sub(r'([!?.])\1+', r'\1', text)
    
    # Stop at common hallucination phrases
    stop_phrases = [
        "The information provided",
        "I would love",
        "Do you have",
        "Is there anything",
        "<|user|>",
        "<|system|>",
        "Hope this helps",
        "Let me know if",
        "Feel free to"
    ]
    
    for phrase in stop_phrases:
        if phrase in text:
            text = text.split(phrase)[0].strip()
            break
    
    # If text ends with incomplete sentence, remove it
    if text and text[-1] not in '.!?':
        sentences = text.split('.')
        if len(sentences) > 1:
            text = '.'.join(sentences[:-1]) + '.'
    
    # Remove trailing whitespace and ensure single space after periods
    text = re.sub(r'\.\s+', '. ', text).strip()
    
    return text

def agroassist_chat(message, history):
    """
    Generate agricultural advice response.
    
    Args:
        message: User's agricultural question
        history: Conversation history (managed by Gradio)
    
    Returns:
        Generated expert agricultural advice
    """
    # Format prompt for TinyLlama
    prompt = f"<|system|>\n{SYSTEM_MESSAGE}</s>\n<|user|>\n{message}</s>\n<|assistant|>\n"
    
    # Tokenize
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        padding=False, 
        truncation=True, 
        max_length=512
    ).to(model.device)
    
    # Generate response with stronger repetition penalty
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,  # Reduced from 200 to prevent rambling
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.3,  # Increased from 1.15
            no_repeat_ngram_size=3,  # Prevent repeating 3-word sequences
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant response
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()
    
    # Clean the response
    response = clean_response(response)
    
    # Ensure response ends properly
    if response and response[-1] not in '.!?':
        response += '.'
    
    return response

# Create Gradio interface
demo = gr.ChatInterface(
    fn=agroassist_chat,
    title="🌾 AgroAssist: AI Agricultural Advisory System",
    description="""
    **AgroAssist** is an AI-powered agricultural advisor fine-tuned on expert Q&A pairs. 
    Ask questions about crops, soil management, pest control, irrigation, or any farming topic. 
    
    ---
    
    ** Disclaimer:** This AI provides educational information. For critical agricultural decisions, 
    Consult certified agricultural extension officers or local experts.
    """,
    examples=[
        "What are the best practices for growing tomatoes?",
        "How can I control pests naturally without chemicals?",
        "What is crop rotation and why is it important?",
        "How do I test my soil pH at home?",
        "What are the signs of nitrogen deficiency in plants?",
        "When is the best time to plant corn?",
        "How much water do cucumber plants need?",
        "What causes blossom end rot in tomatoes?",
        "How do I prepare soil for planting vegetables?",
        "What are the benefits of composting?"
    ],
    theme=gr.themes.Soft(
        primary_hue="green",
        secondary_hue="emerald"
    ),
    cache_examples=False
)

if __name__ == "__main__":
    demo.launch()