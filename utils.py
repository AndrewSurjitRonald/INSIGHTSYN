# src/utils.py
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import streamlit as st

MODEL_REPO_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
STATE_FILE = Path("state.json")

def load_state(default_state={"history": []}):
    """Loads the application state."""
    if not STATE_FILE.exists():
        return default_state
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default_state

def save_state(state):
    """Saves the application state."""
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")

@st.cache_resource
def load_main_model():
    """Loads and caches the main LLM from the Hugging Face Hub."""
    with st.spinner(f"Loading model '{MODEL_REPO_ID}'... This is a one-time download and may take a while."):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO_ID, use_fast=True)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_REPO_ID,
            device_map="auto",
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        )
        model.eval()
    return tokenizer, model

# Do not load the model at import time; load lazily via the cached function

def run_idea_generator(user_text: str, context: str = "") -> str:
    """Uses the main LLM to generate ideas or responses."""
    tokenizer, model = load_main_model()
    prompt = f"Based on the following context: '{context}'. Please brainstorm and expand on this idea: '{user_text}'"
    messages = [{"role": "user", "content": prompt}]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=250)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)