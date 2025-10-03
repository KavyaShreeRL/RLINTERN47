import argparse
import json
import os
import sys
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

@dataclass
class GenerationConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 150
    top_p: float = 1.0
    top_k: Optional[int] = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    json_mode: bool = False
    logit_bias: Optional[Dict[str, float]] = None
    seed: Optional[int] = None

class LLMGenerator:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def generate(self, prompt: str, config: GenerationConfig) -> Dict[str, Any]:
        enhanced_prompt = f"{prompt} Please respond in JSON format." if config.json_mode and "json" not in prompt.lower() else prompt
        
        api_params = {
            "model": config.model,
            "messages": [{"role": "user", "content": enhanced_prompt}],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
            "frequency_penalty": config.frequency_penalty,
            "presence_penalty": config.presence_penalty,
        }
        
        if config.stop: api_params["stop"] = config.stop
        if config.json_mode: api_params["response_format"] = {"type": "json_object"}
        if config.logit_bias: api_params["logit_bias"] = config.logit_bias
        if config.seed is not None: api_params["seed"] = config.seed
        
        try:
            response = self.client.chat.completions.create(**api_params)
            return {
                "success": True,
                "text": response.choices[0].message.content,
                "usage": {"prompt_tokens": response.usage.prompt_tokens, "completion_tokens": response.usage.completion_tokens, "total_tokens": response.usage.total_tokens},
                "model": response.model,
                "config": asdict(config),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"success": False, "error": str(e), "config": asdict(config), "timestamp": datetime.now().isoformat()}

def parse_logit_bias(bias_str: str) -> Dict[str, float]:
    if not bias_str: return {}
    bias_dict = {}
    for pair in bias_str.split(','):
        token_id, bias = pair.split(':')
        bias_dict[token_id.strip()] = float(bias.strip())
    return bias_dict

def parse_stop_sequences(stop_str: str) -> List[str]:
    return [s.strip() for s in stop_str.split(',')] if stop_str else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["temperature", "top_p_top_k", "penalties", "stop_max_tokens", "beam_search", "logit_bias"])
    parser.add_argument("--prompt")
    parser.add_argument("legacy_prompt", nargs="?", help=argparse.SUPPRESS)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=150)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int)
    parser.add_argument("--frequency_penalty", type=float, default=0.0)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--stop", type=str)
    parser.add_argument("--json_mode", action="store_true")
    parser.add_argument("--logit_bias", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--output", "-o", type=str)
    parser.add_argument("--raw", action="store_true")
    
    args = parser.parse_args()
    prompt = args.prompt or args.legacy_prompt
    if not prompt: parser.error("Prompt required")
    if args.repetition_penalty: args.frequency_penalty = args.repetition_penalty
    
    config = GenerationConfig(
        model=args.model, temperature=args.temperature, max_tokens=args.max_tokens,
        top_p=args.top_p, top_k=args.top_k, frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty, stop=parse_stop_sequences(args.stop),
        json_mode=args.json_mode, logit_bias=parse_logit_bias(args.logit_bias), seed=args.seed
    )
    
    modified_prompt = prompt
    
    try:
        result = LLMGenerator().generate(modified_prompt, config)
        if result["success"]:
            if args.raw:
                print(result["text"])
            elif args.verbose:
                print(f"Model: {result['model']}\nPrompt: {prompt}\nTemperature: {config.temperature}")
                print(f"Generated Text:\n{result['text']}")
                print(f"Usage: {result['usage']['total_tokens']} tokens")
            else:
                print(f"Generated Text (Temperature: {config.temperature}):")
                print("-" * 40)
                print(result["text"])
            if args.output:
                with open(args.output, 'w') as f: json.dump(result, f, indent=2)
        else:
            print(f"Generation failed: {result['error']}")
            sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
