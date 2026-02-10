import os
import numpy as np
import onnxruntime as ort
from transformers import DistilBertTokenizerFast

# -------- PATH SETUP (IMPORTANT) --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
 
MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "results",
    "onnx_quantized_original",
    "model_quantized.onnx"
)

TOKENIZER_PATH = os.path.join(
    BASE_DIR,
    "..",
    "results",
    "onnx_quantized_original"
)

MAX_LEN = 128
# --------------------------------------

# Load tokenizer (LOCAL, not HF Hub)
tokenizer = DistilBertTokenizerFast.from_pretrained(
    TOKENIZER_PATH,
    local_files_only=True
)

# Load ONNX model
session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

output_name = session.get_outputs()[0].name


def predict_sentiment(text: str):
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="np"
    )

    inputs = {
        "input_ids": encoded["input_ids"].astype(np.int64),
        "attention_mask": encoded["attention_mask"].astype(np.int64),
    }

    logits = session.run([output_name], inputs)[0]

    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    label = int(np.argmax(probs))
    confidence = float(np.max(probs))

    sentiment = "POSITIVE" if label == 1 else "NEGATIVE"

    return sentiment, confidence


if __name__ == "__main__":
    while True:
        text = input("\nEnter a comment (or 'exit'): ")
        if text.lower() == "exit":
            break

        sentiment, conf = predict_sentiment(text)
        print(f"Sentiment: {sentiment} | Confidence: {conf:.4f}")
