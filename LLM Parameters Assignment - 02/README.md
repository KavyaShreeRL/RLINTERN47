# LLM Parameters Assignment - 02

##  Objective
This repository contains comprehensive experiments exploring how different parameters control the behavior of Large Language Models (LLMs). Through systematic testing, we examine the effects on determinism, creativity, and coherence across various parameter configurations.

##  Repository Structure
```
llm-parameters-assignment-02/
│
├── .venv                    
├── notebooks/
│   ├── 01_temperature.ipynb      
│   ├── 02_top_p_top_k.ipynb        
│   ├── 03_penalties.ipynb        
│   ├── 04_stop_max_tokens.ipynb  
│   ├── 05_beam_search.ipynb      
│   └── 06_open_assignment.ipynb  
└── outputs/
    ├── temperature_examples.md    
    ├── top_p_top_k_examples.md   
    ├── penalties_examples.md     
    └── stop_max_tokens_examples.md
    ├── beam_search_examples.md   
    ├── open_assignment_examples.md     
├── .env
├── generate.py
├── local_beam_search.py
├── local_top_k.py
├── README.md
├── requirements.txt   

```

##  Setup Instructions

### Prerequisites
- Python 3.10 or higher
- OpenAI API key
- Optional: transformers library for local top-k demonstration

### Installation
1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

4. Optional - For top-k demonstration:
   ```bash
   pip install transformers torch
   ```

### Quick Start
```bash

python generate.py "Tell me a fun fact" --temperature 0.7


python generate.py "Create a user profile" --json_mode --temperature 0.2


python generate.py "Write about AI" --frequency_penalty 0.8 --presence_penalty 0.6


python local_top_k.py
python local_beam_search.py
```
```

##  Experiments Summary

| Experiment | Prompt | Parameters Tested | Key Observations |
|------------|--------|------------------|------------------|
| **Temperature** | "Tell me a fun fact about Paris." | temp: 0.0, 0.5, 1.0, 1.5 | Lower temp = more deterministic; Higher temp = more creative but potentially less coherent |
| **Top-p/Top-k** | "List 5 synonyms for happy." | top_p: 0.5 vs 1.0; top_k: 1 vs 50 (local GPT-2) | Top-p controls diversity; Top-k limits vocabulary scope |
| **Penalties** | "Write 4 lines starting with 'AI'." | frequency_penalty: 0.8; presence_penalty: 0.8 | Frequency penalty reduces repetition; Presence penalty encourages novel topics |
| **Stop/Tokens** | "Generate dialogue between User and Assistant." | stop: ["User:"]; max_tokens: 50 vs 200 | Stop sequences provide structure; Max tokens control length |
| **Beam Search** | "Summarize: AI will change industries..." | num_beams: 1 vs 5; length_penalty: 0.8 vs 1.2 (local GPT-2) | More beams = more coherent; Length penalty affects verbosity |
| **JSON/Logit Bias** | "Generate a book review for '1984'." | JSON schema + logit bias variations | JSON mode ensures structure; Logit bias influences content direction |

##  Key Findings

### Parameter Effects on Creativity vs Control

**Most Creative Configuration:**
- High temperature (1.0-1.5)
- High top_p (0.9-1.0)
- Low penalties (0.0-0.2)
- High max_tokens

**Most Controlled Configuration:**
- Low temperature (0.0-0.2)
- Low top_p (0.3-0.5)
- Moderate penalties (0.5-0.8)
- Stop sequences
- JSON mode

### Parameter Recommendations by Use Case

#### Factual Q&A
```python
{
    "temperature": 0.0,
    "top_p": 0.3,
    "max_tokens": 150,
    "frequency_penalty": 0.2,
    "response_format": {"type": "json_object"}
}
```

#### Creative Writing
```python
{
    "temperature": 0.9,
    "top_p": 0.95,
    "max_tokens": 500,
    "presence_penalty": 0.6,
    "frequency_penalty": 0.3
}
```

#### Structured JSON Output
```python
{
    "temperature": 0.2,
    "top_p": 0.5,
    "response_format": {"type": "json_object"},
    "max_tokens": 300
}
```

##  Insights & Reflections

### Which Parameters Most Affect Creativity?
1. **Temperature** - Primary driver of randomness and creativity
2. **Top-p** - Controls diversity of token selection
3. **Presence Penalty** - Encourages exploration of new topics

### Which Parameters Help Enforce Structure?
1. **JSON Mode** - Guarantees valid JSON structure
2. **Stop Sequences** - Provides clear boundaries
3. **Low Temperature** - Ensures predictable patterns
4. **Logit Bias** - Guides specific token selection

### Parameter Interactions
- **Temperature + Top-p**: Combined effect on randomness (multiplicative)
- **Penalties + Temperature**: Penalties can counteract high temperature chaos
- **JSON Mode + Low Temperature**: Optimal for structured, predictable outputs
- **Beam Search + Length Penalty**: Controls both quality and verbosity
- **Top-k vs Top-p**: Different sampling strategies - top-k limits vocabulary, top-p limits probability mass

### Production Recommendations
1. **Always validate outputs** when using higher creativity settings
2. **Use JSON mode** for any structured data requirements
3. **Test parameter combinations** as effects are not always additive
4. **Monitor token usage** especially with high max_tokens settings
5. **Implement fallback strategies** for when parameters produce unexpected results

##  Complete Experiment Commands

### Run All Experiments
```bash
# 1. Temperature Experiments
python generate.py --experiment temperature --prompt "Tell me a fun fact about Paris." --temperature 0.0
python generate.py --experiment temperature --prompt "Tell me a fun fact about Paris." --temperature 0.5
python generate.py --experiment temperature --prompt "Tell me a fun fact about Paris." --temperature 1.0
python generate.py --experiment temperature --prompt "Tell me a fun fact about Paris." --temperature 1.5

# 2. Top-p Experiments (OpenAI API)
python generate.py --experiment top_p_top_k --prompt "List 5 synonyms for happy." --top_p 0.5
python generate.py --experiment top_p_top_k --prompt "List 5 synonyms for happy." --top_p 1.0

# 3. Top-k Experiments (Local GPT-2)
python local_top_k.py

# 4. Penalties Experiments
python generate.py --experiment penalties --prompt "Write 4 lines starting with the word AI." --frequency_penalty 0.8
python generate.py --experiment penalties --prompt "Write 4 lines starting with the word AI." --presence_penalty 0.8

# 5. Stop Sequences & Max Tokens
python generate.py --experiment stop_max_tokens --prompt "Generate a dialogue between User and Assistant." --stop "User:" --max_tokens 50
python generate.py --experiment stop_max_tokens --prompt "Generate a dialogue between User and Assistant." --stop "User:" --max_tokens 200

# 6. Beam Search (Local GPT-2)
python local_beam_search.py

# 7. JSON Schema & Logit Bias
python generate.py --experiment logit_bias --prompt "Generate a book review for '1984' by George Orwell in JSON format." --json_mode --temperature 0.2
python generate.py --experiment logit_bias --prompt "Write a short review of 1984 by George Orwell." --logit_bias "12481:2.0" --temperature 0.8
python generate.py --experiment logit_bias --prompt "Write a short review of 1984 by George Orwell." --logit_bias "14935:-5.0" --temperature 0.8
```

##  CLI Tool Usage

The included `generate.py` CLI tool allows easy experimentation:

```bash

python generate.py "Your prompt here" --temperature 0.7


python generate.py "Write a story" \
    --temperature 0.8 \
    --top_p 0.9 \
    --frequency_penalty 0.6 \
    --presence_penalty 0.4 \
    --max_tokens 300 \
    --stop "THE END" \
    --verbose


python generate.py "Create user data" \
    --json_mode \
    --temperature 0.2 \
    --output results.json
```

##  Experiment Results
Detailed experimental outputs are available in the `outputs/` directory:
- [Temperature Examples](outputs/temperature_examples.md)
- [Top-p/Top-k Examples](outputs/top_p_top_k_examples.md) - includes local GPT-2 top-k demonstration
- [Penalties Examples](outputs/penalties_examples.md)
- [Stop/Max Tokens Examples](outputs/stop_max_tokens_examples.md)
- [Beam Search Examples](outputs/beam_search_examples.md)
- [JSON Schema & Logit Bias Examples](outputs/open_assignment_examples.md)

##  Learning Outcomes
This assignment demonstrated:
- How to systematically test LLM parameters
- The trade-offs between creativity and control
- Practical parameter configurations for different use cases
- The importance of validation and testing in production systems
- How parameter combinations can produce unexpected emergent behaviors
- API limitations and creative workarounds (local models for top-k sampling)
- Different sampling strategies and their practical applications

---

