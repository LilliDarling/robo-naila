use super::*;
use crate::{audio::AudioBus, device::run_device, metrics::HubMetrics, vad::VadConfig};
use dashmap::DashMap;
use opentelemetry_sdk::{
    error::OTelSdkResult,
    trace::{SdkTracerProvider, SpanData, SpanExporter},
};
use std::{sync::Mutex, time::Duration};

#[derive(Clone, Debug, Default)]
struct Capture(Arc<Mutex<Vec<SpanData>>>);
impl SpanExporter for Capture {
    async fn export(&self, batch: Vec<SpanData>) -> OTelSdkResult {
        self.0.lock().unwrap().extend(batch);
        Ok(())
    }
}
fn number(span: &SpanData, key: &str) -> f64 {
    match &span.attributes.iter().find(|kv| kv.key.as_str() == key).unwrap().value {
        opentelemetry::Value::F64(v) => *v,
        opentelemetry::Value::I64(v) => *v as f64,
        _ => panic!("not numeric"),
    }
}
struct Transport {
    id: &'static str,
    rx: Receiver<AudioFrame>,
}
impl AudioTransport for Transport {
    fn id(&self) -> &str {
        self.id
    }
    async fn recv(&mut self) -> Option<AudioFrame> {
        self.rx.recv().await
    }
    async fn send(&mut self, _: TtsFrame) -> Result<(), TransportError> {
        tokio::time::sleep(Duration::from_millis(25)).await;
        Ok(())
    }
}
fn frame() -> AudioFrame {
    AudioFrame { data: Bytes::from(vec![0; 1920]), sample_rate: 48000, timestamp: 0 }
}

#[tokio::test]
async fn delayed_reader_eof_cancels_only_its_device_with_retained_sender() {
    tokio::time::timeout(Duration::from_secs(3), async {
        let capture = Capture::default();
        let provider = SdkTracerProvider::builder().with_simple_exporter(capture.clone()).build();
        // Only this unit test installs an exporting provider.
        opentelemetry::global::set_tracer_provider(provider.clone());
        let (audio_tx, audio_rx) = tokio::sync::mpsc::channel(1);
        let retained_sender = audio_tx.clone();
        audio_tx.send(frame()).await.unwrap(); // force dispatch backpressure
        let cancel = CancellationToken::new();
        let other_cancel = CancellationToken::new();
        let (bus_tx, _bus_rx) = tokio::sync::mpsc::channel(256);
        let bus = Arc::new(AudioBus {
            audio_tx: bus_tx,
            tts_sub: DashMap::new(),
            device_configs: DashMap::new(),
        });
        let tasks = Arc::new(DashMap::new());
        tasks.insert(Arc::from("first"), cancel.clone());
        tasks.insert(Arc::from("other"), other_cancel.clone());
        let (other_tx, other_rx) = tokio::sync::mpsc::channel(1);
        let other = tokio::spawn(run_device(
            Transport { id: "other", rx: other_rx },
            bus.clone(),
            Arc::from("2"),
            VadConfig::default(),
            other_cancel.clone(),
            tasks.clone(),
            Arc::new(HubMetrics::new()),
            opentelemetry::Context::new(),
        ));
        let decoded = Arc::new(tokio::sync::Notify::new());
        let notify = decoded.clone();
        let mut calls = 0;
        let mut decodes = 0;
        let reader = tokio::spawn(receive_audio(
            move || {
                calls += 1;
                let index = calls;
                async move {
                    tokio::time::sleep(Duration::from_millis(60)).await;
                    (index <= 2).then(Packet::default)
                }
            },
            move |_, _| {
                std::thread::sleep(Duration::from_millis(10));
                notify.notify_one();
                decodes += 1;
                if decodes == 2 {
                    return OpusDecoder::new(48000, Channels::Mono).unwrap().decode(
                        &[0xff],
                        &mut [],
                        false,
                    );
                }
                Ok(960)
            },
            audio_tx,
            opentelemetry::Context::new(),
            cancel.clone(),
        ));
        decoded.notified().await;
        tokio::time::sleep(Duration::from_millis(30)).await;
        let device = tokio::spawn(run_device(
            Transport { id: "first", rx: audio_rx },
            bus.clone(),
            Arc::from("1"),
            VadConfig::default(),
            cancel.clone(),
            tasks.clone(),
            Arc::new(HubMetrics::new()),
            opentelemetry::Context::new(),
        ));
        loop {
            if bus.tts_sub.contains_key("first") {
                break;
            }
            tokio::task::yield_now().await;
        }
        bus.tts_sub
            .get("first")
            .unwrap()
            .send(TtsFrame {
                data: Bytes::from_static(&[1]),
                sample_rate: 48000,
                is_final: false,
                is_opus: true,
            })
            .await
            .unwrap();
        reader.await.unwrap();
        device.await.unwrap();
        assert!(cancel.is_cancelled());
        assert!(!other_cancel.is_cancelled());
        assert!(!other.is_finished());
        assert!(!bus.tts_sub.contains_key("first"));
        assert!(bus.tts_sub.contains_key("other"));
        let spans = capture.0.lock().unwrap().clone();
        let receive = spans.iter().find(|s| s.name == "hub.webrtc.receive").unwrap();
        assert_eq!(number(receive, "opus_decode_ms.count"), 2.0);
        assert_eq!(number(receive, "decode_errors"), 1.0);
        assert_eq!(number(receive, "receive_dispatch_ms.count"), 1.0);
        assert!(number(receive, "opus_decode_ms.sum") >= 20.0);
        assert!(number(receive, "opus_decode_ms.sum") < 100.0); // excludes 180ms packet waits
        assert!(number(receive, "receive_dispatch_ms.sum") >= 30.0);
        let first = spans
            .iter()
            .find(|s| {
                s.name == "hub.session"
                    && s.attributes
                        .iter()
                        .any(|kv| kv.key.as_str() == "device_id" && kv.value.as_str() == "first")
            })
            .unwrap();
        assert_eq!(number(first, "vad_ms.count"), 2.0); // both silent/suppressed
        assert_eq!(number(first, "outbound_send_ms.count"), 1.0);
        assert!(number(first, "outbound_send_ms.sum") >= 25.0);
        other_cancel.cancel();
        other.await.unwrap();
        drop((retained_sender, other_tx));
        provider.shutdown().unwrap();
    })
    .await
    .unwrap();
}
