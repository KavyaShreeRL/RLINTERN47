# Temperature Experiment Results

## Quick Summary
**Prompt**: "Tell me a fun fact about Paris."  
**Temperatures tested**: 0.0, 0.5, 1.0, 1.5

---

## Results

### Temperature 0.0 (Deterministic)
> A fun fact about Paris is that it has a hidden vineyard! Located in the Montmartre district, the Clos Montmartre is a small vineyard that produces a limited amount of wine each year. This charming vineyard, which dates back to the 1930s, is a reminder of the city's agricultural past and is often overlooked by tourists.

**Analysis**: Consistent, factual, predictable output.

### Temperature 0.5 (Low Creativity) 
> A fun fact about Paris is that it has a "secret" vineyard located in the Montmartre district. The vineyard, known as the Clos Montmartre, is a small, charming vineyard that produces a limited amount of wine each year. It was established in 1933 and is one of the last remaining vineyards within the city limits.

**Analysis**: Similar content with slight variation in phrasing.

### Temperature 1.0 (Balanced)
> A fun fact about Paris is that it has a "secret" train station, known as the Saint-Martin tunnel. Originally opened in 1859 as a part of the Parisian railway network, it was closed in 1930 and remained largely forgotten for decades. However, the station is part of the popular urban exploration scene.

**Analysis**: Different topic choice, more creative angle.

### Temperature 1.5 (High Creativity)
> A fun fact about Paris is that the city has its own "spite fort." The Fort de l'Eau was built in response to a property dispute in the early 19th century: an engineer wanted to become seen or take claim of the area along a fortification by becoming securely fenced into the space... [text becomes increasingly incoherent]

**Analysis**: Creative but loses coherence, potential hallucinations.

---

## Key Findings

| Temperature | Consistency | Creativity | Coherence | Use Case |
|-------------|-------------|------------|-----------|----------|
| 0.0 | Perfect | Very Low | Perfect | Technical docs, APIs |
| 0.5 | High | Low | High | Educational content |
| 1.0 | Medium | Medium | Good | General content, marketing |
| 1.5 | Low | Very High | Variable | Creative writing, brainstorming |


