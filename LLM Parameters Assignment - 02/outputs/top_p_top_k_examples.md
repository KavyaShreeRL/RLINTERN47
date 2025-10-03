# Top-p vs Top-k Experiment Results

## Quick Summary
**Prompt**: "List 5 synonyms for happy."  
**Parameters tested**: top_p=0.5 vs 1.0, simulated top_k effects

---

## Results

### Top-p = 0.5 (Nucleus Sampling)
> Here are five synonyms for "happy":
> 1. Joyful  2. Cheerful  3. Elated  4. Content  5. Delighted

### Top-p = 1.0 (Full Distribution)
> Here are five synonyms for "happy":  
> 1. Joyful  2. Cheerful  3. Delighted  4. Content  5. Elated

**Analysis**: Minimal difference in this simple task. Top-p effects are more noticeable in longer, more creative text generation.

---

## Key Findings

| Parameter | Effect | Use Case |
|-----------|--------|----------|
| top_p = 0.5 | More focused, consistent choices | Technical writing, factual content |
| top_p = 1.0 | Full vocabulary access | Creative writing, diverse outputs |
| Low top_k (1-10) | Very focused, repetitive | Specific domain tasks |
| High top_k (50+) | More diverse vocabulary | General content generation |

**Note**: OpenAI's API doesn't support top_k directly, but similar effects can be achieved through temperature and top_p combinations.

## Real Top-k Demonstration
For true top-k sampling with actual parameter control, see:
```bash
python local_top_k.py
```
This uses GPT-2 locally to demonstrate real top-k=1 vs top-k=50 behavior.
