# LLM Service Implementation Plan

**Status**: In Progress
**Priority**: P0 - Critical
**Owner**: AI-Server Team
**Created**: 2025-11-04
**Model**: Llama 3.1 8B Instruct (Q4_K_M quantized)

---

## Overview

Implement the LLM (Large Language Model) service to load and run the Llama 3.1 8B Instruct model for intelligent response generation in the NAILA AI-Server. This service will replace the current hardcoded response patterns with actual AI-powered text generation.

---

## Current State Analysis

### What Exists
- ✅ Model downloaded: `models/llm/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` (4.6GB)
- ✅ Hardware detection: `ai-server/utils/hardware_detection.py`
- ✅ Response generator framework: `ai-server/orchestration/response_generator.py`
- ✅ LangGraph orchestration pipeline: `ai-server/agents/orchestrator.py`
- ✅ Conversation memory system: `ai-server/memory/conversation_memory.py`
- ✅ MQTT communication infrastructure

### What's Missing
- ❌ LLM service implementation
- ❌ Model loading and initialization code
- ❌ Inference interface
- ❌ Integration with response generator
- ❌ Integration with server startup
- ❌ Error handling for model failures
- ❌ Performance monitoring for inference

### Dependencies Required
```
llama-cpp-python>=0.2.0  # Primary inference library
```

---

## Implementation Plan

### Phase 1: Service Foundation (Create the LLM Service Class)

**Goal**: Create the basic LLM service structure with model loading

**Files to Create**:
- `ai-server/services/llm_service.py`
- `ai-server/services/__init__.py` (if not exists)

**Implementation Details**:

1. **Create `LLMService` class** with:
   - `__init__()` - Initialize configuration
   - `async load_model()` - Load the GGUF model
   - `async generate()` - Generate text from prompts
   - `async generate_chat()` - Generate chat-style responses
   - `unload_model()` - Clean up resources
   - `is_loaded()` - Check if model is ready

2. **Configuration Parameters**:
   - Model path from `.env`
   - Context window size (default: 4096 tokens)
   - Max tokens per generation (default: 512)
   - Temperature (default: 0.7)
   - Top-p sampling (default: 0.9)
   - Hardware acceleration (CPU/CUDA/Metal)
   - Thread count (based on hardware detection)

3. **Hardware Optimization**:
   - Use `hardware_detection.py` to determine optimal settings
   - Enable GPU acceleration if available (CUDA/Metal)
   - Set appropriate thread count for CPU inference
   - Configure memory mapping for large model

**Key Considerations**:
- Model loading is slow (~5-10 seconds), do it during server startup
- GGUF format is optimized for CPU inference
- Q4_K_M quantization balances quality and speed
- Context window management for long conversations

**Success Criteria**:
- [ ] LLMService class created with all methods
- [ ] Model loads successfully without errors
- [ ] Can generate simple text completions
- [ ] Hardware acceleration configured correctly

---

### Phase 2: Inference Interface (Chat Completion API)

**Goal**: Create a robust chat interface compatible with OpenAI-style messages

**Implementation Details**:

1. **Message Format Support**:
   ```python
   messages = [
       {"role": "system", "content": "You are NAILA, a helpful robot assistant."},
       {"role": "user", "content": "What's the weather like?"},
       {"role": "assistant", "content": "Previous response..."},
       {"role": "user", "content": "And tomorrow?"}
   ]
   ```

2. **Generate Methods**:
   - `generate_chat(messages, max_tokens, temperature, stop_sequences)`
   - `generate_completion(prompt, max_tokens, temperature)`
   - `generate_streaming(messages)` - For future real-time responses

3. **Prompt Formatting**:
   - Convert message list to Llama 3.1 chat format
   - Include system prompts for personality
   - Handle multi-turn conversations
   - Apply proper special tokens (`<|begin_of_text|>`, `<|start_header_id|>`, etc.)

4. **Response Processing**:
   - Extract generated text
   - Strip special tokens
   - Handle partial responses
   - Detect completion vs truncation

**Success Criteria**:
- [ ] Chat-style message format working
- [ ] System prompts properly applied
- [ ] Multi-turn conversations supported
- [ ] Generated responses are coherent and relevant

---

### Phase 3: Integration with Response Generator

**Goal**: Replace hardcoded responses with LLM-generated responses

**Files to Modify**:
- `ai-server/orchestration/response_generator.py`

**Current Code Location**: Line 61-83 (hardcoded response patterns)

**Implementation Details**:

1. **Inject LLM Service**:
   - Pass LLM service instance to ResponseGenerator
   - Store as instance variable
   - Add fallback for when LLM unavailable

2. **Replace `_generate_response()` Method**:
   ```python
   # OLD: Pattern-based responses
   if intent == "greeting":
       return random.choice(["Hello!", "Hi there!"])

   # NEW: LLM-generated responses
   messages = self._build_chat_messages(intent, query, context)
   response = await self.llm_service.generate_chat(messages)
   return response
   ```

3. **Context Integration**:
   - Include conversation history from memory
   - Add retrieved context from input processing
   - Include user preferences if available
   - Add system prompt with personality traits

4. **Prompt Engineering**:
   - System prompt: Define NAILA's personality and capabilities
   - Context injection: Add relevant conversation history
   - Intent guidance: Guide LLM based on detected intent
   - Constraints: Keep responses concise and natural

**Example System Prompt**:
```
You are NAILA, a friendly and helpful robot assistant. You communicate naturally and conversationally.
You have access to sensors, can move around, and interact with the physical world.
Keep your responses concise (1-3 sentences) unless more detail is specifically requested.
Be helpful, curious, and show personality in your responses.
```

**Success Criteria**:
- [ ] LLM integrated into response generator
- [ ] Hardcoded responses replaced
- [ ] Context properly injected into prompts
- [ ] Responses are natural and personality-appropriate
- [ ] Fallback to basic responses if LLM fails

---

### Phase 4: Server Lifecycle Integration

**Goal**: Load LLM during server startup, handle gracefully during shutdown

**Files to Modify**:
- `ai-server/server/lifecycle.py`
- `ai-server/main.py`

**Implementation Details**:

1. **Add Model Loading Phase**:
   - Create new startup phase: "Loading AI Models"
   - Load LLM before starting MQTT service
   - Show progress/status during loading
   - Handle loading failures gracefully

2. **Initialization Sequence** (Update `_initialize_services()`):
   ```
   Phase 1: Initialize configuration
   Phase 2: Load AI models (NEW)
       - Initialize LLM service
       - Load Llama model
       - Verify model loaded
   Phase 3: Register protocol handlers
   Phase 4: Start MQTT service
   Phase 5: Start health monitoring
   ```

3. **Pass LLM to Components**:
   - Store LLM service in `NailaAIServer`
   - Pass to `ResponseGenerator` during init
   - Make available to orchestrator if needed

4. **Shutdown Handling**:
   - Unload model during graceful shutdown
   - Cancel any in-flight inference
   - Free GPU/memory resources

5. **Error Handling**:
   - If model fails to load, log error but continue
   - Operate in "degraded mode" with fallback responses
   - Attempt to reload on next restart

**Success Criteria**:
- [ ] Model loads during server startup
- [ ] Loading status visible in logs
- [ ] LLM service accessible to response generator
- [ ] Graceful shutdown unloads model properly
- [ ] Server handles model load failures without crashing

---

### Phase 5: Performance Optimization & Monitoring

**Goal**: Ensure LLM performs efficiently and track key metrics

**Implementation Details**:

1. **Performance Monitoring**:
   - Track inference time per request
   - Monitor tokens per second
   - Log context length usage
   - Track model memory usage

2. **Add to Health Monitoring**:
   - Include LLM status in health checks
   - Report inference metrics in `naila/ai/metrics` topic
   - Alert if inference times exceed thresholds

3. **Optimization Techniques**:
   - Batch requests if possible (future enhancement)
   - Implement token budget management
   - Cache common responses (future enhancement)
   - Consider prompt compression for long contexts

4. **Logging & Debugging**:
   - Log prompts sent to LLM (debug mode only)
   - Log generated responses with metadata
   - Track generation parameters used
   - Log any inference errors or warnings

**Metrics to Track**:
```python
{
    "llm_status": "loaded" | "unloaded" | "error",
    "model_name": "Meta-Llama-3.1-8B-Instruct-Q4_K_M",
    "inference_time_ms": 1250,
    "tokens_generated": 45,
    "tokens_per_second": 36.0,
    "context_length": 512,
    "temperature": 0.7,
    "hardware_acceleration": "cpu" | "cuda" | "metal"
}
```

**Success Criteria**:
- [ ] Inference metrics tracked and logged
- [ ] Performance data included in health metrics
- [ ] Inference time reasonable (<5s for typical responses)
- [ ] No memory leaks during extended operation

---

### Phase 6: Testing & Validation

**Goal**: Ensure LLM service works correctly in all scenarios

**Test Cases**:

1. **Unit Tests** (`tests/services/test_llm_service.py`):
   - [ ] Model loading succeeds
   - [ ] Generate simple completion
   - [ ] Generate chat response
   - [ ] Handle invalid prompts gracefully
   - [ ] Unload model properly
   - [ ] Handle concurrent requests

2. **Integration Tests** (`tests/integration/test_llm_integration.py`):
   - [ ] LLM loads during server startup
   - [ ] Response generator uses LLM
   - [ ] MQTT messages trigger LLM responses
   - [ ] Conversation history included in context
   - [ ] Fallback works when LLM unavailable

3. **End-to-End Tests**:
   - [ ] Send MQTT message, receive LLM-generated response
   - [ ] Multi-turn conversation maintains context
   - [ ] Different intents produce appropriate responses
   - [ ] Server handles model errors without crashing

4. **Manual Testing**:
   - [ ] Test various question types
   - [ ] Verify response quality and personality
   - [ ] Check response times are acceptable
   - [ ] Test with long conversation histories
   - [ ] Test edge cases (empty input, very long input, etc.)

**Success Criteria**:
- [ ] All unit tests passing
- [ ] All integration tests passing
- [ ] End-to-end flow working correctly
- [ ] Response quality meets expectations

---

## Technical Architecture

### Component Diagram
```
┌─────────────────────────────────────────────────────────────┐
│                      NAILA AI Server                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐         ┌─────────────────┐               │
│  │  MQTT Layer  │────────▶│  Orchestrator   │               │
│  └──────────────┘         └────────┬────────┘               │
│                                     │                        │
│                                     ▼                        │
│                          ┌──────────────────┐               │
│                          │ Response         │               │
│                          │ Generator        │               │
│                          └────────┬─────────┘               │
│                                   │                          │
│                                   ▼                          │
│                          ┌──────────────────┐               │
│                          │  LLM Service     │◀─┐            │
│                          │  (NEW)           │  │            │
│                          └────────┬─────────┘  │            │
│                                   │            │            │
│                                   ▼            │            │
│                          ┌──────────────────┐  │            │
│                          │ llama-cpp-python │  │            │
│                          │ (Inference)      │  │            │
│                          └────────┬─────────┘  │            │
│                                   │            │            │
│                                   ▼            │            │
│                          ┌──────────────────┐  │            │
│                          │ Llama 3.1 8B     │  │            │
│                          │ Model (GGUF)     │  │            │
│                          └──────────────────┘  │            │
│                                                 │            │
│  ┌──────────────┐                              │            │
│  │ Conversation │──────────────────────────────┘            │
│  │ Memory       │ (Context for prompts)                     │
│  └──────────────┘                                           │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow
```
1. MQTT Message arrives
   ↓
2. Orchestrator processes through graph nodes
   ↓
3. Response Generator receives intent + context
   ↓
4. Build chat messages with:
   - System prompt (personality)
   - Conversation history (from memory)
   - Current query + context
   ↓
5. LLM Service generates response
   ↓
6. Response sent back via MQTT
   ↓
7. Conversation memory updated
```

---

## Configuration

### Environment Variables (`.env`)
```bash
# Model Configuration
LLM_MODEL_PATH=models/llm/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
LLM_CONTEXT_SIZE=4096
LLM_MAX_TOKENS=512
LLM_TEMPERATURE=0.7
LLM_TOP_P=0.9
LLM_THREADS=4  # Auto-detected if not set

# Performance
LLM_GPU_LAYERS=0  # Number of layers to offload to GPU (0=CPU only)
LLM_BATCH_SIZE=512
```

### System Prompt Configuration
Location: `ai-server/config/prompts.py` (to be created)

```python
NAILA_SYSTEM_PROMPT = """
You are NAILA, a friendly and helpful robot assistant. You are:
- Curious and eager to learn
- Helpful and supportive
- Natural and conversational
- Concise but thorough when needed

You have physical capabilities:
- You can move around using wheels
- You have sensors to perceive the environment
- You can interact with objects
- You are learning and growing

Communication style:
- Keep responses brief (1-3 sentences) unless detail is requested
- Use natural, friendly language
- Show personality and emotion
- Be honest about limitations
"""
```

---

## Risks & Mitigations

### Risk 1: Model Loading Time
- **Risk**: 5-10 second startup delay
- **Impact**: Server takes longer to start
- **Mitigation**: Load model in background phase, show progress

### Risk 2: Inference Speed
- **Risk**: 2-5 seconds per response on CPU
- **Impact**: Slower conversational experience
- **Mitigation**:
  - Use Q4 quantization (already done)
  - Optimize context size
  - Consider GPU acceleration
  - Implement response streaming (future)

### Risk 3: Memory Usage
- **Risk**: Model uses 5-6GB RAM
- **Impact**: Requires significant memory
- **Mitigation**:
  - Use memory mapping
  - Ensure sufficient system RAM
  - Document minimum requirements

### Risk 4: Model Failure
- **Risk**: Model fails to load or crashes
- **Impact**: No intelligent responses
- **Mitigation**:
  - Implement fallback to pattern-based responses
  - Graceful degradation
  - Clear error logging

### Risk 5: Response Quality
- **Risk**: Generated responses may be off-topic or inappropriate
- **Impact**: Poor user experience
- **Mitigation**:
  - Careful system prompt engineering
  - Add response validation
  - Implement content filtering if needed
  - Log problematic responses for review

---

## Success Metrics

### Functional Metrics
- [ ] Model loads successfully on server start
- [ ] LLM generates responses for all intent types
- [ ] Response quality subjectively "good" (natural, relevant, personality-appropriate)
- [ ] Conversation context properly maintained
- [ ] Server stable during extended operation

### Performance Metrics
- Model load time: < 15 seconds
- Average inference time: < 5 seconds (CPU), < 2 seconds (GPU)
- Tokens per second: > 20 (CPU), > 50 (GPU)
- Memory usage: < 7GB
- No memory leaks over 24 hour operation

### Quality Metrics
- Response relevance: > 90% (manual evaluation)
- Natural language quality: Subjectively good
- Personality consistency: Maintained across conversations
- Context awareness: Demonstrates memory of previous messages

---

## Timeline & Milestones

### Milestone 1: Basic LLM Service (2-3 hours)
- Create LLMService class
- Implement model loading
- Basic text generation working

### Milestone 2: Chat Interface (1-2 hours)
- Message format support
- Prompt formatting for Llama 3.1
- Response processing

### Milestone 3: Integration (2-3 hours)
- Integrate with response generator
- Replace hardcoded responses
- Add to server lifecycle

### Milestone 4: Testing & Polish (2-3 hours)
- Write tests
- Performance monitoring
- Error handling improvements

**Total Estimated Time**: 7-11 hours

---

## Dependencies & Prerequisites

### Required
- [x] Llama 3.1 model downloaded
- [ ] `llama-cpp-python` package installed
- [ ] Hardware detection working
- [ ] Response generator exists
- [ ] Server lifecycle framework exists

### Optional (Future Enhancements)
- [ ] GPU available for acceleration
- [ ] Streaming response support
- [ ] Response caching
- [ ] Multi-model support

---

## Future Enhancements

### Short-term (Next Sprint)
1. **Streaming Responses**: Real-time token generation for faster perceived response time
2. **Response Caching**: Cache common queries to reduce inference load
3. **Personality Profiles**: Multiple system prompts for different personality modes
4. **Context Pruning**: Smart truncation of long conversation histories

### Long-term (Future Sprints)
1. **Multi-model Support**: Swap between different LLMs based on task
2. **Fine-tuning**: Custom fine-tuned model for NAILA-specific tasks
3. **RAG Integration**: Retrieval-Augmented Generation for knowledge queries
4. **Function Calling**: Enable LLM to trigger device commands
5. **Vision-Language Integration**: Combine with vision models for multimodal understanding

---

## References

### Documentation
- [llama-cpp-python GitHub](https://github.com/abetlen/llama-cpp-python)
- [Llama 3.1 Model Card](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
- [GGUF Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)

### Related Code
- `ai-server/orchestration/response_generator.py` - Where LLM will be integrated
- `ai-server/agents/orchestrator.py` - LangGraph orchestration
- `ai-server/memory/conversation_memory.py` - Conversation context
- `ai-server/utils/hardware_detection.py` - Hardware optimization
- `ai-server/server/lifecycle.py` - Server startup/shutdown

---

## Status Tracking

### Phase 1: Service Foundation
- [ ] Create `services/llm_service.py`
- [ ] Implement `LLMService` class
- [ ] Implement `load_model()` method
- [ ] Implement `generate()` method
- [ ] Hardware optimization configured
- [ ] Basic testing complete

### Phase 2: Inference Interface
- [ ] Implement chat message format
- [ ] Implement `generate_chat()` method
- [ ] Implement prompt formatting
- [ ] Response processing working
- [ ] Multi-turn conversations tested

### Phase 3: Response Generator Integration
- [ ] Inject LLM service into ResponseGenerator
- [ ] Replace hardcoded responses
- [ ] Implement context integration
- [ ] System prompt configured
- [ ] Fallback mechanism implemented

### Phase 4: Server Lifecycle Integration
- [ ] Add model loading phase to startup
- [ ] Integrate into `lifecycle.py`
- [ ] Implement shutdown handling
- [ ] Error handling for load failures
- [ ] Status logging implemented

### Phase 5: Performance & Monitoring
- [ ] Inference metrics tracked
- [ ] Health monitoring integration
- [ ] Performance logging
- [ ] Optimization applied
- [ ] Metrics validated

### Phase 6: Testing & Validation
- [ ] Unit tests written and passing
- [ ] Integration tests written and passing
- [ ] End-to-end tests passing
- [ ] Manual testing complete
- [ ] Response quality validated

---

**Last Updated**: 2025-11-04
**Next Review**: After Phase 1 completion
