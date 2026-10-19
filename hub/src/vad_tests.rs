use super::*;
use bytes::Bytes;
use opus::{Channels, Decoder};
use std::{
    io::{Cursor, Read},
    path::Path,
    process::Command,
};

#[test]
fn unchanged_question_through_opus_is_one_utterance() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let python = std::env::var_os("NAILA_TEST_PEER_PYTHON")
        .unwrap_or_else(|| root.join("devices/audio-client/.venv/bin/python").into());
    let available = Command::new(&python)
        .args(["-c", "from aiortc.codecs.opus import OpusEncoder; OpusEncoder()"])
        .output();
    if !available.is_ok_and(|output| output.status.success()) {
        eprintln!("SKIP unchanged_question_through_opus_is_one_utterance: audio-client Python or aiortc Opus encoder unavailable ({python:?})");
        return;
    }
    // aiortc 1.14.0 OpusEncoder.encode returns payloads, not a framed stream:
    // https://github.com/aiortc/aiortc/blob/1.14.0/src/aiortc/codecs/opus.py
    // Length-prefix only this test's subprocess output; use the real peer codec.
    let encoded = Command::new(&python)
        .current_dir(root)
        .env("PYTHONPATH", root)
        .args([
            "-c",
            r#"
import sys
from fractions import Fraction
from aiortc.codecs.opus import OpusEncoder
from av import AudioFrame
from benchmarks.peer import Fixture
fixture = Fixture.load('benchmarks/fixtures/question.wav')
pcm = bytes(25 * 1920) + fixture.pcm + bytes(50 * 1920)
encoder = OpusEncoder()
for offset in range(0, len(pcm), 1920):
    frame = AudioFrame(format='s16', layout='mono', samples=960)
    frame.planes[0].update(pcm[offset:offset + 1920])
    frame.sample_rate, frame.time_base, frame.pts = 48000, Fraction(1, 48000), offset // 2
    packets, _ = encoder.encode(frame)
    for packet in packets:
        sys.stdout.buffer.write(len(packet).to_bytes(2, 'little') + packet)
"#,
        ])
        .output()
        .expect("install the benchmark environment documented in benchmarks/README.md");
    assert!(
        encoded.status.success(),
        "{}",
        String::from_utf8_lossy(&encoded.stderr)
    );
    let mut input = Cursor::new(encoded.stdout);
    let mut decoder = Decoder::new(48_000, Channels::Mono).unwrap();
    let mut filters = [
        VadFilter::new(VadConfig::default()),
        VadFilter::new(VadConfig {
            hangover_threshold: 15,
            ..VadConfig::default()
        }),
    ];
    let mut events = [(0, 0); 2];
    let mut pcm = [0i16; 960];
    while input.position() < input.get_ref().len() as u64 {
        let mut length = [0u8; 2];
        input.read_exact(&mut length).unwrap();
        let mut packet = vec![0; u16::from_le_bytes(length) as usize];
        input.read_exact(&mut packet).unwrap();
        assert_eq!(decoder.decode(&packet, &mut pcm, false).unwrap(), 960);
        let data: Bytes = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
        for (filter, (starts, ends)) in filters.iter_mut().zip(&mut events) {
            let frame = AudioFrame {
                data: data.clone(),
                sample_rate: 48_000,
                timestamp: 0,
            };
            match filter.process(frame) {
                VadResult::Emit(SpeechEvent::Start, _) => *starts += 1,
                VadResult::Emit(SpeechEvent::End, _) => *ends += 1,
                _ => {}
            }
        }
    }
    assert_eq!(
        events[1],
        (2, 2),
        "15-frame control must reproduce the split"
    );
    assert_eq!(
        events[0],
        (1, 1),
        "default must bridge the internal speech pause"
    );
}
