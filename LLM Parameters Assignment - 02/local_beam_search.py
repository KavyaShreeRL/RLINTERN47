

try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer, set_seed
    import torch
except ImportError:
    print("Missing transformers library")
    print("Install with: pip install transformers torch")
    exit(1)

class LocalBeamSearchDemo:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def generate_with_beam_search(self, prompt, num_beams=1, length_penalty=1.0, max_length=50):
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            if num_beams == 1:
                outputs = self.model.generate(
                    inputs,
                    max_length=len(inputs[0]) + max_length,
                    do_sample=True,
                    temperature=0.7,
                    no_repeat_ngram_size=2,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            else:
                outputs = self.model.generate(
                    inputs,
                    max_length=len(inputs[0]) + max_length,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    no_repeat_ngram_size=2,
                    repetition_penalty=1.2,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text[len(prompt):].strip()

def main():
    demo = LocalBeamSearchDemo()
    prompt = "Artificial Intelligence will transform industries"
    
    configs = [
        (1, 1.0, "Beam Search (num_beams=1)"),
        (5, 0.8, "Beam Search (num_beams=5, length_penalty=0.8)"),
        (5, 1.2, "Beam Search (num_beams=5, length_penalty=1.2)")
    ]
    
    for num_beams, length_penalty, description in configs:
        print(f"{description}:")
        
        set_seed(42)
        result = demo.generate_with_beam_search(prompt, num_beams=num_beams, length_penalty=length_penalty, max_length=40)
        
        print(f"  Result: {result}")
        print()

if __name__ == "__main__":
    main()
