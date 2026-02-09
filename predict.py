import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

class NativeQuantizedModel:
    def __init__(self, model_folder):   
        self.tokenizer = AutoTokenizer.from_pretrained(model_folder)
        config = AutoConfig.from_pretrained(model_folder)
        model_fp32 = AutoModelForSequenceClassification.from_config(config)
        
        self.model = torch.quantization.quantize_dynamic(
            model_fp32,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        
        weights_path = f"{model_folder}/quantized_weights.pt"
        state_dict = torch.load(weights_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.eval() 
        print("Model loaded!")

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_index = torch.argmax(probs, dim=-1).item()
        
        label_map = self.model.config.id2label if self.model.config.id2label else {0: "Negative", 1: "Positive"}
        
        return label_map[pred_index], probs[0][pred_index].item()

if __name__ == '__main__':
    model_path = 'results\minilm_pytorch_quantized'
    analyzer = NativeQuantizedModel(model_path)
    text = input('Enter your comment: ')
    sentiment, score = analyzer.predict(text)
    print(f"{sentiment} ({score:.1%})")