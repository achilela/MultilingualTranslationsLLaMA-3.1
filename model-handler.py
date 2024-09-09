import torch
import time

class ModelHandler:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_response(self, conversation):
        inputs = self.tokenizer(conversation, return_tensors="pt", truncation=True, max_length=1024)
        
        start_time = time.time()
        output = ""
        
        with torch.no_grad():
            for _ in range(150):  # Increased range for potentially longer responses
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95
                )
                
                new_token = generated[0, -1].item()
                new_word = self.tokenizer.decode([new_token])
                output += new_word
                
                inputs = self.tokenizer(conversation + output, return_tensors="pt", truncation=True, max_length=1024)
                
                if time.time() - start_time >= 0.01:
                    yield output
                    start_time = time.time()
                
                if new_token == self.tokenizer.eos_token_id:
                    break
        
        return output.strip()
