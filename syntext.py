# src/syntext.py
from transformers import pipeline

# Use a dictionary to cache pipelines so they are only loaded once
pipelines_cache = {}

def get_pipeline(task, model):
    if model not in pipelines_cache:
        print(f"Loading pipeline: {task} with model {model}")
        pipelines_cache[model] = pipeline(task, model=model)
    return pipelines_cache[model]

def analyze_text(text_input: str) -> dict:
    """
    Performs summarization, sentiment, and NER on a text input.
    """
    # Get pipelines
    summarizer = get_pipeline("summarization", "t5-small")
    sentiment_analyzer = get_pipeline("sentiment-analysis", "distilbert-base-uncased-finetuned-sst-2-english")
    ner_pipeline = get_pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", grouped_entities=True)

    # Run models
    summary = summarizer(text_input, max_length=50, min_length=10, do_sample=False)[0]['summary_text']
    sentiment = sentiment_analyzer(text_input)[0]
    entities = ner_pipeline(text_input)

    # Structure and return output
    structured_output = {
        "summary": summary,
        "sentiment": {"label": sentiment['label'], "score": round(sentiment['score'], 4)},
        "entities": [entity['word'] for entity in entities if 'word' in entity]
    }
    return structured_output