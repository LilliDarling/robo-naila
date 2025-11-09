# STT (Speech-to-Text) Service Implementation Plan

**Status**: Not Started
**Priority**: P1 - High
**Owner**: AI-Server Team
**Created**: 2025-11-04
**Model**: Whisper Small English (ggml-small.en.bin)

---

## Overview

Implement the STT (Speech-to-Text) service to transcribe audio input from the robot's microphone into text for processing by the NAILA AI-Server. This service will enable voice interaction by converting spoken commands and questions into text that can be processed by the orchestration pipeline.

---

## Current State Analysis

### What Exists
- ✅ Model downloaded: `models/stt/ggml-small.en.bin` (466MB)
- ✅ Hardware detection: `config/hardware_config.py`
- ✅ MQTT topics defined: `naila/device/{device_id}/audio`
- ✅ Orchestration pipeline ready to receive text input
- ✅ LLM service for generating responses

### What's Missing
- ❌ STT service implementation
- ❌ Audio preprocessing (resampling, normalization)
- ❌ MQTT audio message handling
- ❌ Integration with orchestration pipeline
- ❌ Real-time vs batch transcription
- ❌ Audio format validation
- ❌ Performance monitoring for transcription

### Dependencies Required
```
whisper-cpp-python>=0.1.0  # Python bindings for whisper.cpp
# OR
faster-whisper>=1.2.1     # Optimized Whisper implementation
# OR
openai-whisper>=20231117   # Official OpenAI implementation (slower)
```

**Recommendation**: Use `faster-whisper` for best performance with GGML models.

---

## Implementation Plan

### Phase 1: Service Foundation (Create the STT Service Class)

**Goal**: Create the basic STT service structure with model loading

**Files to Create**:
- `services/stt.py`
- `config/stt.py`

**Implementation Details**:

1. **Create `STTService` class** with:
   - `__init__()` - Initialize configuration
   - `async load_model()` - Load the Whisper GGML model
   - `async transcribe_audio()` - Transcribe audio bytes to text
   - `async transcribe_file()` - Transcribe audio file
   - `unload_model()` - Clean up resources
   - `is_loaded()` - Check if model is ready

2. **Configuration Parameters** (`config/stt.py`):
   - Model path from `.env`
   - Language (English only for small.en model)
   - Beam size (quality vs speed tradeoff)
   - VAD (Voice Activity Detection) settings
   - Sample rate (16kHz for Whisper)
   - Audio format support (WAV, MP3, FLAC, OGG)
   - Silence threshold
   - Min/max audio duration

3. **Hardware Optimization**:
   - Use `hardware_config.py` to determine optimal settings
   - Enable GPU acceleration if available (CUDA/Metal)
   - Set appropriate thread count for CPU inference
   - Configure compute type (float16/int8 for speed)

4. **Audio Preprocessing**:
   - Resample to 16kHz (Whisper requirement)
   - Convert to mono if stereo
   - Normalize audio levels
   - Apply VAD to trim silence
   - Validate audio format and duration

**Key Considerations**:
- Whisper Small English model optimized for English-only transcription
- GGML format provides good CPU performance
- Model loading takes 2-3 seconds
- Transcription time ~0.3-1.0x real-time on CPU
- VAD improves quality by removing silence

**Success Criteria**:
- [ ] STTService class created with all methods
- [ ] Model loads successfully without errors
- [ ] Can transcribe audio files accurately
- [ ] Audio preprocessing working correctly
- [ ] Hardware acceleration configured

---

### Phase 2: Audio Format Handling

**Goal**: Support multiple audio formats and handle MQTT audio messages

**Implementation Details**:

1. **Supported Audio Formats**:
   - WAV (preferred, no decoding overhead)
   - MP3 (common, needs decoding)
   - FLAC (lossless)
   - OGG (compressed)
   - Raw PCM (from microphone streams)

2. **Audio Validation**:
   ```python
   def validate_audio(audio_data: bytes) -> tuple[bool, str]:
       # Check audio format
       # Verify sample rate compatibility
       # Check duration (min 0.1s, max 30s)
       # Validate audio isn't corrupted
       return is_valid, error_message
   ```

3. **Audio Preprocessing Pipeline**:
   ```python
   async def preprocess_audio(audio_data: bytes) -> np.ndarray:
       # Decode audio format
       # Resample to 16kHz
       # Convert to mono
       # Normalize volume
       # Apply VAD (optional)
       return processed_audio
   ```

4. **MQTT Audio Message Format**:
   ```json
   {
       "device_id": "naila_001",
       "timestamp": "2025-11-04T12:34:56Z",
       "audio_data": "<base64_encoded_audio>",
       "format": "wav",
       "sample_rate": 16000,
       "duration_ms": 2500,
       "metadata": {
           "source": "microphone",
           "language": "en"
       }
   }
   ```

**Success Criteria**:
- [ ] Multiple audio formats supported
- [ ] Audio preprocessing pipeline working
- [ ] MQTT audio messages properly parsed
- [ ] Format validation prevents bad input
- [ ] Resampling works correctly

---

### Phase 3: Transcription Engine

**Goal**: Implement accurate and efficient audio transcription

**Implementation Details**:

1. **Transcription Methods**:
   ```python
   async def transcribe_audio(
       audio_data: bytes,
       language: str = "en",
       task: str = "transcribe",
       temperature: float = 0.0,
       beam_size: int = 5,
       vad_filter: bool = True
   ) -> TranscriptionResult:
       # Preprocess audio
       # Run Whisper inference
       # Post-process text
       # Return structured result
   ```

2. **Transcription Result**:
   ```python
   @dataclass
   class TranscriptionResult:
       text: str                    # Transcribed text
       language: str                # Detected language
       confidence: float            # Average confidence score
       duration_ms: int             # Audio duration
       transcription_time_ms: int   # Processing time
       words: List[WordSegment]     # Word-level timestamps (optional)
       segments: List[Segment]      # Sentence-level segments
   ```

3. **Post-Processing**:
   - Remove leading/trailing whitespace
   - Fix common transcription errors
   - Apply punctuation correction
   - Handle empty transcriptions
   - Filter out non-speech sounds

4. **Quality Assurance**:
   - Confidence threshold (reject low confidence)
   - Min/max text length validation
   - Profanity filter (optional)
   - Detect gibberish/nonsense

**Success Criteria**:
- [ ] Transcription accuracy > 90% (WER < 10%)
- [ ] Processing time reasonable (<2x real-time)
- [ ] Confidence scores reliable
- [ ] Post-processing improves quality
- [ ] Handles edge cases gracefully

---

### Phase 4: Integration with Orchestration Pipeline

**Goal**: Connect STT service to MQTT handlers and orchestrator

**Files to Modify**:
- `mqtt/handlers/ai_handlers.py`
- `mqtt/handlers/coordinator.py`
- `server/lifecycle.py`
- `agents/orchestrator.py`

**Implementation Details**:

1. **Create Audio Handler**:
   ```python
   # In ai_handlers.py
   async def handle_audio_input(self, message):
       # Extract audio data from MQTT message
       # Validate audio format
       # Transcribe using STT service
       # If transcription successful:
       #     Send to orchestrator as text input
       # If transcription failed:
       #     Send error response
   ```

2. **MQTT Topic Registration**:
   ```python
   # Subscribe to audio input topic
   "naila/device/{device_id}/audio"
   ```

3. **Integration Flow**:
   ```
   1. Robot sends audio via MQTT
      ↓
   2. AIHandlers receives audio message
      ↓
   3. STT Service transcribes to text
      ↓
   4. Orchestrator processes text (existing flow)
      ↓
   5. Response sent back via MQTT
   ```

4. **Error Handling**:
   - Invalid audio format → Send error message
   - Transcription failed → Request retry
   - Low confidence → Ask for clarification
   - Empty transcription → "I didn't hear anything"

**Success Criteria**:
- [ ] Audio messages properly handled
- [ ] STT integrated into MQTT pipeline
- [ ] Transcribed text flows to orchestrator
- [ ] Error handling works correctly
- [ ] End-to-end audio-to-response working

---

### Phase 5: Server Lifecycle Integration

**Goal**: Load STT during server startup, handle gracefully during shutdown

**Files to Modify**:
- `server/lifecycle.py`
- `server/naila_server.py`

**Implementation Details**:

1. **Add STT Loading Phase**:
   ```python
   # In lifecycle.py _load_ai_models()
   async def _load_ai_models(self):
       # Load LLM (existing)
       if self.llm_service:
           await self.llm_service.load_model()

       # Load STT (new)
       if self.stt_service:
           logger.info("Loading STT model...")
           success = await self.stt_service.load_model()
           if success:
               logger.info(f"STT model loaded: {self.stt_service.model_path.name}")
               self.protocol_handlers.set_stt_service(self.stt_service)
           else:
               logger.warning("STT model failed to load - audio input disabled")
   ```

2. **Initialization Sequence**:
   ```
   Phase 1: Initialize configuration
   Phase 2: Load AI models
       - Load LLM model
       - Load STT model (NEW)
   Phase 3: Register protocol handlers
   Phase 4: Start MQTT service
   Phase 5: Start health monitoring
   ```

3. **Pass STT to Components**:
   - Store STT service in `NailaAIServer`
   - Pass to `AIHandlers` for audio processing
   - Make available via dependency injection

4. **Shutdown Handling**:
   - Unload STT model during graceful shutdown
   - Free GPU/memory resources
   - Cancel any in-flight transcriptions

**Success Criteria**:
- [ ] STT loads during server startup
- [ ] Loading status visible in logs
- [ ] STT service accessible to handlers
- [ ] Graceful shutdown unloads STT properly
- [ ] Server handles STT load failures without crashing

---

### Phase 6: Performance Optimization & Monitoring

**Goal**: Ensure STT performs efficiently and track key metrics

**Implementation Details**:

1. **Performance Monitoring**:
   - Track transcription time per request
   - Monitor real-time factor (RTF)
   - Log confidence scores
   - Track model memory usage
   - Count successful vs failed transcriptions

2. **Add to Health Monitoring**:
   - Include STT status in health checks
   - Report transcription metrics in `naila/ai/metrics` topic
   - Alert if transcription times exceed thresholds

3. **Optimization Techniques**:
   - Use VAD to skip silence (faster processing)
   - Batch audio requests if possible
   - Cache common phrases (future)
   - Consider streaming transcription (future)

4. **Logging & Debugging**:
   - Log audio characteristics (duration, format)
   - Log transcription results (debug mode)
   - Track confidence scores
   - Log any transcription errors

**Metrics to Track**:
```python
{
    "stt_status": "loaded" | "unloaded" | "error",
    "model_name": "ggml-small.en.bin",
    "transcription_time_ms": 850,
    "audio_duration_ms": 2500,
    "real_time_factor": 0.34,  # <1.0 is faster than real-time
    "confidence": 0.92,
    "text_length": 45,
    "language": "en",
    "hardware_acceleration": "cpu" | "cuda" | "metal"
}
```

**Success Criteria**:
- [ ] Transcription metrics tracked and logged
- [ ] Performance data included in health metrics
- [ ] RTF < 1.0 (faster than real-time)
- [ ] No memory leaks during extended operation

---

### Phase 7: Testing & Validation

**Goal**: Ensure STT service works correctly in all scenarios

**Test Cases**:

1. **Unit Tests** (`tests/unit/test_stt_service.py`):
   - [ ] Model loading succeeds
   - [ ] Transcribe sample audio files
   - [ ] Audio preprocessing works correctly
   - [ ] Handle invalid audio gracefully
   - [ ] Format validation working
   - [ ] Unload model properly

2. **Integration Tests** (`tests/integration/test_stt_integration.py`):
   - [ ] STT loads during server startup
   - [ ] Audio handler processes MQTT messages
   - [ ] Transcribed text reaches orchestrator
   - [ ] End-to-end audio-to-response flow
   - [ ] Fallback works when STT unavailable

3. **Audio Test Cases**:
   - [ ] Clear speech, no background noise
   - [ ] Speech with background noise
   - [ ] Quiet speech (low volume)
   - [ ] Fast speech
   - [ ] Accented speech
   - [ ] Multiple speakers (should handle single speaker)
   - [ ] Empty audio (silence)
   - [ ] Very short audio (<1 second)
   - [ ] Long audio (>30 seconds)

4. **Manual Testing**:
   - [ ] Send real audio via MQTT
   - [ ] Verify transcription accuracy
   - [ ] Check response times
   - [ ] Test different audio formats
   - [ ] Test error scenarios

**Success Criteria**:
- [ ] All unit tests passing
- [ ] All integration tests passing
- [ ] Transcription accuracy acceptable (WER < 10%)
- [ ] Processing time acceptable (RTF < 1.0)

---

## Technical Architecture

### Component Diagram
```
┌─────────────────────────────────────────────────────────────┐
│                      NAILA AI Server                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐         ┌─────────────────┐               │
│  │  MQTT Layer  │────────▶│  AI Handlers    │               │
│  │              │         └────────┬────────┘               │
│  │  Audio Topic │                  │                         │
│  └──────────────┘                  ▼                         │
│                          ┌──────────────────┐               │
│                          │  STT Service     │◀──┐           │
│                          │  (NEW)           │   │           │
│                          └────────┬─────────┘   │           │
│                                   │             │           │
│                                   ▼             │           │
│                          ┌──────────────────┐   │           │
│                          │ faster-whisper   │   │           │
│                          │ (Inference)      │   │           │
│                          └────────┬─────────┘   │           │
│                                   │             │           │
│                                   ▼             │           │
│                          ┌──────────────────┐   │           │
│                          │ Whisper Small EN │   │           │
│                          │ Model (GGML)     │   │           │
│                          └──────────────────┘   │           │
│                                                  │           │
│                          Transcribed Text       │           │
│                                   │             │           │
│                                   ▼             │           │
│                          ┌──────────────────┐   │           │
│                          │  Orchestrator    │   │           │
│                          └────────┬─────────┘   │           │
│                                   │             │           │
│                                   ▼             │           │
│                          ┌──────────────────┐   │           │
│                          │  LLM Service     │───┘           │
│                          │  (Response Gen)  │               │
│                          └──────────────────┘               │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow
```
1. Robot captures audio from microphone
   ↓
2. Audio sent via MQTT (base64 encoded)
   ↓
3. AI Handler receives audio message
   ↓
4. STT Service:
   - Validates audio format
   - Preprocesses audio (resample, normalize)
   - Runs Whisper inference
   - Post-processes transcription
   ↓
5. Transcribed text sent to Orchestrator
   ↓
6. Orchestrator processes text (existing flow):
   - Intent detection
   - Context gathering
   - Response generation (LLM)
   ↓
7. Response sent back via MQTT
   ↓
8. Robot speaks response (TTS)
```

---

## Configuration

### Environment Variables (`.env`)
```bash
# STT Configuration
STT_MODEL_PATH=models/stt/ggml-small.en.bin
STT_LANGUAGE=en
STT_BEAM_SIZE=5
STT_COMPUTE_TYPE=int8  # float16, int8, float32
STT_DEVICE=auto  # cpu, cuda, auto

# Audio Settings
STT_SAMPLE_RATE=16000
STT_VAD_FILTER=true
STT_MIN_SILENCE_DURATION_MS=500
STT_SPEECH_PAD_MS=400

# Quality Settings
STT_MIN_CONFIDENCE=0.6
STT_MIN_DURATION_MS=100
STT_MAX_DURATION_MS=30000

# Performance
STT_THREADS=4  # Auto-detected if not set
STT_GPU_LAYERS=-1  # -1=auto, 0=CPU only
```

### Configuration Module (`config/stt.py`)
```python
"""STT Service Configuration Constants"""

import os
from pathlib import Path

# Model Configuration
MODEL_PATH = os.getenv("STT_MODEL_PATH", "models/stt/ggml-small.en.bin")
LANGUAGE = os.getenv("STT_LANGUAGE", "en")
BEAM_SIZE = int(os.getenv("STT_BEAM_SIZE", "5"))
COMPUTE_TYPE = os.getenv("STT_COMPUTE_TYPE", "int8")
DEVICE = os.getenv("STT_DEVICE", "auto")

# Audio Settings
SAMPLE_RATE = int(os.getenv("STT_SAMPLE_RATE", "16000"))
VAD_FILTER = os.getenv("STT_VAD_FILTER", "true").lower() == "true"
MIN_SILENCE_DURATION_MS = int(os.getenv("STT_MIN_SILENCE_DURATION_MS", "500"))
SPEECH_PAD_MS = int(os.getenv("STT_SPEECH_PAD_MS", "400"))

# Quality Settings
MIN_CONFIDENCE = float(os.getenv("STT_MIN_CONFIDENCE", "0.6"))
MIN_DURATION_MS = int(os.getenv("STT_MIN_DURATION_MS", "100"))
MAX_DURATION_MS = int(os.getenv("STT_MAX_DURATION_MS", "30000"))

# Performance
THREADS = int(os.getenv("STT_THREADS", "0"))  # 0 = auto-detect
GPU_LAYERS = int(os.getenv("STT_GPU_LAYERS", "-1"))

# Supported Formats
SUPPORTED_FORMATS = ["wav", "mp3", "flac", "ogg", "m4a"]

# Logging
LOG_TRANSCRIPTIONS = os.getenv("STT_LOG_TRANSCRIPTIONS", "true").lower() == "true"
LOG_PERFORMANCE_METRICS = True
```

---

## Risks & Mitigations

### Risk 1: Model Loading Time
- **Risk**: 2-3 second startup delay
- **Impact**: Server takes slightly longer to start
- **Mitigation**: Load model in parallel with LLM, show progress

### Risk 2: Transcription Speed
- **Risk**: Slow transcription on CPU (0.5-1.0x real-time)
- **Impact**: Delayed response to voice commands
- **Mitigation**:
  - Use faster-whisper (optimized)
  - Enable GPU acceleration
  - Use VAD to skip silence
  - Use small.en model (faster than base/medium)

### Risk 3: Accuracy Issues
- **Risk**: Background noise, accents, unclear speech
- **Impact**: Poor transcription quality, wrong commands
- **Mitigation**:
  - Use VAD to filter noise
  - Request clarification for low confidence
  - Implement noise reduction preprocessing
  - Train users to speak clearly

### Risk 4: Audio Format Compatibility
- **Risk**: Unsupported formats, sample rate mismatches
- **Impact**: Transcription fails or produces errors
- **Mitigation**:
  - Validate audio format upfront
  - Support common formats (WAV, MP3, FLAC)
  - Automatic resampling to 16kHz
  - Clear error messages for bad input

### Risk 5: Memory Usage
- **Risk**: Model uses 500MB-1GB RAM
- **Impact**: Increased memory footprint
- **Mitigation**:
  - Use int8 compute type (reduced memory)
  - Ensure sufficient system RAM
  - Unload during shutdown

---

## Success Metrics

### Functional Metrics
- [ ] Model loads successfully on server start
- [ ] Transcribes audio with >90% accuracy (WER < 10%)
- [ ] Handles multiple audio formats correctly
- [ ] Integrates seamlessly with orchestration pipeline
- [ ] Server stable during extended operation

### Performance Metrics
- Model load time: < 5 seconds
- Transcription RTF: < 1.0 (faster than real-time)
- Average transcription time: < 3 seconds for 3-second audio
- Memory usage: < 1.5GB
- No memory leaks over 24 hour operation

### Quality Metrics
- Transcription accuracy (WER): < 10%
- Confidence score reliability: High confidence → Good accuracy
- Handles background noise reasonably
- Low latency for voice commands

---

## Timeline & Milestones

### Milestone 1: Basic STT Service (3-4 hours)
- Create STTService class
- Implement model loading
- Basic audio transcription working

### Milestone 2: Audio Handling (2-3 hours)
- Audio preprocessing pipeline
- Format validation and conversion
- MQTT audio message parsing

### Milestone 3: Integration (2-3 hours)
- Integrate with AI handlers
- Connect to orchestration pipeline
- Add to server lifecycle

### Milestone 4: Testing & Polish (2-3 hours)
- Write tests
- Performance monitoring
- Error handling improvements

**Total Estimated Time**: 9-13 hours

---

## Dependencies & Prerequisites

### Required
- [x] Whisper model downloaded
- [ ] `faster-whisper` package installed
- [x] Hardware detection working
- [x] MQTT infrastructure exists
- [x] Orchestration pipeline exists

### Optional (Future Enhancements)
- [ ] GPU available for acceleration
- [ ] Streaming transcription support
- [ ] Noise reduction preprocessing
- [ ] Multi-language support

---

## Future Enhancements

### Short-term (Next Sprint)
1. **Streaming Transcription**: Real-time transcription as audio arrives
2. **Noise Reduction**: Advanced preprocessing to handle background noise
3. **Voice Activity Detection**: Better silence detection
4. **Confidence-based Retry**: Auto-retry low confidence transcriptions

### Long-term (Future Sprints)
1. **Multi-language Support**: Switch to Whisper multilingual model
2. **Speaker Diarization**: Identify different speakers
3. **Punctuation Restoration**: Improve formatting of transcriptions
4. **Custom Vocabulary**: Add domain-specific words (robot commands)
5. **Wake Word Detection**: Only transcribe after "Hey NAILA"

---

## References

### Documentation
- [faster-whisper GitHub](https://github.com/guillaumekln/faster-whisper)
- [Whisper Model Card](https://github.com/openai/whisper)
- [GGML Format Specification](https://github.com/ggerganov/ggml)
- [Audio Processing Best Practices](https://developer.mozilla.org/en-US/docs/Web/Media/Formats/Audio_concepts)

### Related Code
- `services/llm.py` - Similar service pattern to follow
- `mqtt/handlers/ai_handlers.py` - Where audio messages will be handled
- `agents/orchestrator.py` - Where transcribed text will be sent
- `config/hardware_config.py` - Hardware optimization
- `server/lifecycle.py` - Server startup/shutdown

---

## Status Tracking

### Phase 1: Service Foundation
- [ ] Create `services/stt.py`
- [ ] Create `config/stt.py`
- [ ] Implement `STTService` class
- [ ] Implement `load_model()` method
- [ ] Implement `transcribe_audio()` method
- [ ] Hardware optimization configured
- [ ] Basic testing complete

### Phase 2: Audio Format Handling
- [ ] Audio format validation
- [ ] Audio preprocessing pipeline
- [ ] Format conversion (MP3, FLAC, etc.)
- [ ] MQTT audio message parsing
- [ ] Resampling to 16kHz working

### Phase 3: Transcription Engine
- [ ] Transcription method implemented
- [ ] Post-processing pipeline
- [ ] Confidence scoring
- [ ] Quality assurance checks
- [ ] Edge case handling

### Phase 4: Orchestration Integration
- [ ] Create audio handler in AIHandlers
- [ ] MQTT topic subscription
- [ ] Integration with orchestrator
- [ ] Error handling implemented
- [ ] End-to-end flow working

### Phase 5: Server Lifecycle Integration
- [ ] Add STT loading phase to startup
- [ ] Integrate into `lifecycle.py`
- [ ] Implement shutdown handling
- [ ] Error handling for load failures
- [ ] Status logging implemented

### Phase 6: Performance & Monitoring
- [ ] Transcription metrics tracked
- [ ] Health monitoring integration
- [ ] Performance logging
- [ ] Optimization applied
- [ ] Metrics validated

### Phase 7: Testing & Validation
- [ ] Unit tests written and passing
- [ ] Integration tests written and passing
- [ ] Audio test cases validated
- [ ] Manual testing complete
- [ ] Accuracy meets expectations

---

**Last Updated**: 2025-11-04
**Next Review**: After Phase 1 completion
