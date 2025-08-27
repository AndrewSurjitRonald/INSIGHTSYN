# src/syntone_text.py
from transformers import pipeline

emotion_classifier = pipeline(
    "text-classification", 
    model="SamLowe/roberta-base-go_emotions",
    top_k=1
)

def analyze_text_emotion(text_input: str) -> dict:
    """
    Analyzes the emotional tone of a given text string.
    """
    if not isinstance(text_input, str) or not text_input.strip():
        return {"primary": "unknown", "score": 0.0}
    
    try:
        predictions = emotion_classifier(text_input)
        top_prediction = predictions[0][0]
        return {
            "primary": top_prediction['label'],
            "score": round(top_prediction['score'], 4)
        }
    except Exception:
        return {"primary": "error", "score": 0.0}