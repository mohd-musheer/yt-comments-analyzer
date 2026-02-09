import torch
import numpy
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
import os

def load_model(model_path):
    model = ORTModelForSequenceClassification.from_pretrained(model_path, file_name="model_quantized.onnx")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def predict(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors = 'pt', truncation = True, max_length = 128)

    with torch.no_grad():
        output = model(**inputs)

    logits = output.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)

    pred_index = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][pred_index].item()

    labels = {0: "Negative", 1: "Positive"}

    return labels[pred_index], confidence

if __name__ == '__main__':
    model_path = 'results\onnx_quantized_original'
    model, tokenizer = load_model(model_path)
    print('Enter the comment -')
    text = input()
    sentiment, score = predict(text, model, tokenizer)
    print(f"Result: {sentiment} ({score:.1%})")