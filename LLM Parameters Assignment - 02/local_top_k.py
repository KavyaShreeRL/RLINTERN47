

try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer, set_seed
    import torch
    from torch.nn.functional import softmax
    import numpy as np
except ImportError:
    print("Missing transformers library")
    print("Install with: pip install transformers torch")
    exit(1)

class LocalTopKDemo:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def generate_with_top_k(self, prompt, top_k=50, max_length=50, temperature=1.0):
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=len(inputs[0]) + max_length,
                temperature=temperature,
                do_sample=True,
                top_k=top_k,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text[len(prompt):].strip()

def main():
    demo = LocalTopKDemo()
    prompt = "The weather today is"
    
    top_k_values = [1, 50]
    
    for k in top_k_values:
        print(f"Top-k = {k}:")
        
        for i in range(3):
            set_seed(42 + i)
            result = demo.generate_with_top_k(prompt, top_k=k, max_length=20)
            print(f"  Sample {i+1}: {result}")
        print()

if __name__ == "__main__":
    main()
