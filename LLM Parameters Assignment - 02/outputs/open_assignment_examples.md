# JSON Schema & Logit Bias Results

## Quick Summary
**JSON Schema**: Generate structured book reviews at different temperatures  
**Logit Bias**: Control vocabulary choices in movie reviews

---

## JSON Schema Mode Results

### Temperature 0.0 (Deterministic)
```json
{
  "title": "1984",
  "rating": 5,
  "review": "George Orwell's '1984' is a chilling dystopian novel..."
}
```
**Analysis**: Perfect compliance, formal language, consistent output.

### Temperature 0.7 (Balanced)  
```json
{
  "title": "1984", 
  "rating": 5,
  "review": "A chilling exploration of totalitarianism..."
}
```
**Analysis**: JSON maintained, more varied vocabulary, creative while structured.

### Temperature 1.2 (Creative)
```json
{
  "title": "1984",
  "rating": 5, 
  "review": "A haunting exploration of individualism..."
}
```
**Analysis**: Structure preserved, abstract language, unique perspective.

---

## Logit Bias Experiments

### Baseline (No Bias)
> **Movie Review: "The Echo of Shadows"**  
> A masterful blend of suspense and psychological intrigue that leaves audiences both breathless and contemplative...

**Analysis**: Natural language flow, balanced vocabulary.


### Positive Bias (+2.0 for "excellent")
> **Movie Review: "Elysium's Edge"**  
> A refreshing and thought-provoking entry that captivates the audience and challenges societal norms. Directed by the visionary Lena Morales...

**Analysis**: More sophisticated vocabulary, elevated language tone.

### Negative Bias (-5.0 for "terrible") 
> A haunting exploration of totalitarianism and surveillance. Orwell masterfully crafts a narrative about oppression and personal autonomy...

**Analysis**: Successfully avoided negative terms, maintained positive tone.

---

## Key Findings

| Parameter | Effect | Use Case |
|-----------|--------|----------|
| JSON Mode + temp=0.0 | Perfect structure, formal | API responses, data extraction |
| JSON Mode + temp=0.7 | Creative yet structured | Content generation |
| JSON Mode + temp=1.2 | High creativity, maintained format | Creative applications |
| Logit Bias (+) | Promotes specific vocabulary | Tone control, brand voice |
| Logit Bias (-) | Suppresses unwanted terms | Content filtering |
| **Combined** | JSON + temperature + bias | Maximum control over structure and content |


