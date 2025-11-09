# TTS Synthesis Parameter Recommendations

> **TODO:** Copy and fill in this document after running fine-tuning tests and evaluating voice quality.
>
> Run: `uv run python scripts/finetune_tts_params.py` from the ai-server directory
>
> Then listen to audio samples in `output/tts_finetuning/` and document your findings here.

## Fine-Tuning Results Summary

<!-- TODO: Fill in after running tests -->

**Performance Results:**
- Average RTF: _[TODO]_
- Performance rating: _[TODO]_
- Synthesis time range: _[TODO]_

### Performance Ranking

<!-- TODO: Fill in with actual test results -->

| Rank | Configuration | RTF | Performance Rating | Notes |
|------|--------------|-----|-------------------|-------|
| 1 | _[TODO]_ | _[TODO]_ | _[TODO]_ | _[TODO]_ |
| 2 | _[TODO]_ | _[TODO]_ | _[TODO]_ | _[TODO]_ |
| 3 | _[TODO]_ | _[TODO]_ | _[TODO]_ | _[TODO]_ |

## Recommended Configurations by Use Case

### 1. Default Configuration (Current)

```python
# Current settings in config/tts.py
LENGTH_SCALE = 1.0   # [TODO: Update if changed]
NOISE_SCALE = 0.667  # [TODO: Update if changed]
NOISE_W = 0.8        # [TODO: Update if changed]
```

**Characteristics:** _[TODO: Describe voice quality]_

**Use when:** _[TODO: Describe use cases]_

---

### 2. Natural Conversation

<!-- TODO: Fill in after testing -->

```python
LENGTH_SCALE = _[TODO]_
NOISE_SCALE = _[TODO]_
NOISE_W = _[TODO]_
```

**Characteristics:**
- _[TODO: naturalness]_
- _[TODO: clarity]_
- _[TODO: expressiveness]_

**Use when:**
- _[TODO: use case 1]_
- _[TODO: use case 2]_

---

### 3. Robot Assistant (Professional)

<!-- TODO: Fill in after testing -->

```python
LENGTH_SCALE = _[TODO]_
NOISE_SCALE = _[TODO]_
NOISE_W = _[TODO]_
```

**Characteristics:**
- _[TODO]_

**Use when:**
- _[TODO]_

---

### 4. Expressive Responses

<!-- TODO: Fill in after testing -->

```python
LENGTH_SCALE = _[TODO]_
NOISE_SCALE = _[TODO]_
NOISE_W = _[TODO]_
```

**Characteristics:**
- _[TODO]_

**Use when:**
- _[TODO]_

---

### 5. Custom Configuration

<!-- TODO: Add your own configurations -->

```python
LENGTH_SCALE = _[TODO]_
NOISE_SCALE = _[TODO]_
NOISE_W = _[TODO]_
```

**Characteristics:**
- _[TODO]_

**Use when:**
- _[TODO]_

---

## Parameter Explanations

### LENGTH_SCALE (Speaking Rate)
- **0.5-0.8:** Very fast (rushing, urgent)
- **0.85-0.95:** Fast (energetic, quick)
- **1.0:** Normal (standard speaking rate) ✅
- **1.1-1.3:** Slow (deliberate, clear)
- **1.4+:** Very slow (teaching, careful)

### NOISE_SCALE (Pitch Variation/Expressiveness)
- **0.0-0.3:** Monotone (robot-like, stable)
- **0.4-0.6:** Moderate (natural, balanced) ✅
- **0.7-0.9:** High (expressive, dynamic)
- **1.0:** Maximum (very expressive)

### NOISE_W (Energy Variation/Dynamics)
- **0.0-0.4:** Flat (minimal dynamics)
- **0.5-0.7:** Moderate (natural dynamics) ✅
- **0.8-0.9:** High (dynamic, energetic)
- **1.0:** Maximum (very dynamic)

## Implementation Guide

### Quick Setup (Environment Variables)

Add to your `.env` file:

```bash
# TODO: Update with your preferred settings
TTS_LENGTH_SCALE=1.0
TTS_NOISE_SCALE=0.5
TTS_NOISE_W=0.6
```

### Dynamic Parameter Adjustment

You can adjust parameters per synthesis call:

```python
# Example: Use different parameters for different contexts
audio_data = await tts_service.synthesize(
    text="Your robot has completed the task.",
    length_scale=1.1,    # Slower for important message
    noise_scale=0.3,     # Clear and professional
    noise_w=0.4
)
```

### Context-Aware Synthesis

```python
# Example: Adjust based on message type
if message_type == "error":
    # TODO: Define parameters for errors
    params = {"length_scale": 1.1, "noise_scale": 0.3, "noise_w": 0.4}
elif message_type == "success":
    # TODO: Define parameters for successes
    params = {"length_scale": 0.9, "noise_scale": 0.8, "noise_w": 0.9}
else:
    # TODO: Define parameters for normal conversation
    params = {"length_scale": 1.0, "noise_scale": 0.5, "noise_w": 0.6}

audio_data = await tts_service.synthesize(text, **params)
```

## Testing Methodology

<!-- TODO: Document your testing process -->

**Test Phrases Used:**
1. _[TODO]_
2. _[TODO]_
3. _[TODO]_

**Evaluation Criteria:**
- Naturalness: _[TODO: 1-10 rating]_
- Clarity: _[TODO: 1-10 rating]_
- Expressiveness: _[TODO: 1-10 rating]_
- Suitability: _[TODO: description]_

## Audio Samples

Generated audio samples are available in:
```
ai-server/output/tts_finetuning/
```

Files are named: `{config}_{phrase_preview}.wav`

<!-- TODO: Document your listening test results -->

**Listening Test Results:**
- Best for naturalness: _[TODO]_
- Best for clarity: _[TODO]_
- Best for expressiveness: _[TODO]_
- Best overall: _[TODO]_

## Final Recommendation

<!-- TODO: Fill in after evaluation -->

**For NAILA robot assistant, we recommend:**

```python
# config/tts.py
LENGTH_SCALE = _[TODO]_
NOISE_SCALE = _[TODO]_
NOISE_W = _[TODO]_
```

**Rationale:**
- _[TODO: Why these settings?]_
- _[TODO: What trade-offs?]_
- _[TODO: What use cases?]_

## Performance Notes

<!-- TODO: Fill in performance observations -->

**Observed Performance:**
- RTF range: _[TODO]_
- Average synthesis time: _[TODO]_
- Performance bottlenecks: _[TODO: if any]_

**Conclusion:** _[TODO: Summary of findings]_

---

## Research Notes

<!-- TODO: Add any additional notes from your research -->

**Findings:**
- _[TODO]_

**Observations:**
- _[TODO]_

**Future Testing:**
- _[TODO: What to test next?]_
