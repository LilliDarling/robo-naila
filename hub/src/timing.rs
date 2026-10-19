//! Bounded, task-owned measurements; RTP timestamps never enter these clocks.
use crate::audio::SpeechEvent;
use opentelemetry::{
    Context, KeyValue, global,
    trace::{TraceContextExt, Tracer},
};
use std::time::{Duration, Instant};

// OTel 0.31 Span stores attributes/events, not count/sum/max populations:
// https://docs.rs/opentelemetry/0.31.0/opentelemetry/trace/trait.Span.html
// Aggregate locally to avoid emitting an event for every media frame.
#[derive(Default)]
pub(crate) struct Costs {
    count: u64,
    sum: f64,
    max: f64,
}
impl Costs {
    pub fn record(&mut self, elapsed: Duration) {
        let ms = elapsed.as_secs_f64() * 1000.0;
        self.count += 1;
        self.sum += ms;
        self.max = self.max.max(ms);
    }
    pub fn export(&self, cx: &Context, name: &str) {
        let span = cx.span();
        span.set_attribute(KeyValue::new(format!("{name}.count"), self.count as i64));
        span.set_attribute(KeyValue::new(format!("{name}.sum"), self.sum));
        span.set_attribute(KeyValue::new(format!("{name}.max"), self.max));
        if self.count > 0 {
            span.set_attribute(KeyValue::new(
                format!("{name}.mean"),
                self.sum / self.count as f64,
            ));
        }
    }
}

pub(crate) fn child(name: &'static str, parent: &Context) -> Context {
    parent.with_span(global::tracer("naila.hub").start_with_context(name, parent))
}

pub(crate) struct SessionTiming {
    pub context: Context,
    pub vad: Costs,
    pub send: Costs,
    started: Instant,
    starts: u64,
    end: Option<Instant>,
    first: Option<f64>,
    complete: bool,
    response: Option<Context>,
    opus: u64,
    pcm: u64,
    send_errors: u64,
}
impl SessionTiming {
    pub fn new(parent: &Context, onset: u32, hangover: u32) -> Self {
        let context = child("hub.session", parent);
        for (name, value) in [
            ("vad.onset_frames", onset),
            ("vad.hangover_frames", hangover),
            ("vad.onset_policy_ms", onset * 20),
            ("vad.hangover_policy_ms", hangover * 20),
        ] {
            context
                .span()
                .set_attribute(KeyValue::new(name, value as i64));
        }
        Self {
            context,
            vad: Costs::default(),
            send: Costs::default(),
            started: Instant::now(),
            starts: 0,
            end: None,
            first: None,
            complete: false,
            response: None,
            opus: 0,
            pcm: 0,
            send_errors: 0,
        }
    }
    pub fn event(&mut self, event: SpeechEvent) {
        if event == SpeechEvent::Continue {
            return;
        }
        let now = Instant::now();
        self.context.span().add_event(
            match event {
                SpeechEvent::Start => "vad.START",
                _ => "vad.END",
            },
            vec![KeyValue::new(
                "elapsed_ms",
                now.duration_since(self.started).as_secs_f64() * 1000.0,
            )],
        );
        if event == SpeechEvent::Start {
            self.starts += 1;
        } else if self.starts == 1 && self.end.is_none() {
            self.end = Some(now);
            self.response = Some(child("hub.response", &self.context));
        }
    }
    pub fn sent(&mut self, nonempty: bool, final_frame: bool, opus: bool) {
        if nonempty {
            if opus {
                self.opus += 1;
            } else {
                self.pcm += 1;
            }
        }
        if self.starts != 1 || self.complete {
            return;
        }
        if let Some(anchor) = self.end {
            let ms = anchor.elapsed().as_secs_f64() * 1000.0;
            if nonempty && self.first.is_none() {
                self.first = Some(ms);
            }
            if final_frame {
                if let Some(first) = self.first {
                    self.complete = true;
                    if let Some(cx) = self.response.take() {
                        cx.span().set_attributes([
                            KeyValue::new("hub_response_first_send_ms", first),
                            KeyValue::new("hub_response_final_send_ms", ms),
                            KeyValue::new("outcome", "ok"),
                        ]);
                        cx.span().end();
                    }
                }
            }
        }
    }
    pub fn send_error(&mut self) {
        self.send_errors += 1;
    }
}
impl Drop for SessionTiming {
    fn drop(&mut self) {
        self.vad.export(&self.context, "vad_ms");
        self.send.export(&self.context, "outbound_send_ms");
        self.context.span().set_attributes([
            KeyValue::new("utterance_count", self.starts as i64),
            KeyValue::new("multiple_utterances", self.starts > 1),
            KeyValue::new("outbound.opus_forward_count", self.opus as i64),
            KeyValue::new("outbound.pcm_encode_count", self.pcm as i64),
            KeyValue::new("outbound.send_errors", self.send_errors as i64),
            KeyValue::new(
                "outcome",
                if self.starts > 1 {
                    "multiple_utterances"
                } else if self.complete && self.send_errors == 0 {
                    "ok"
                } else {
                    "incomplete"
                },
            ),
        ]);
        if let Some(cx) = self.response.take() {
            cx.span().set_attribute(KeyValue::new(
                "outcome",
                if self.starts > 1 {
                    "multiple_utterances"
                } else {
                    "incomplete"
                },
            ));
            cx.span().end();
        }
        self.context.span().end();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn first_utterance_owns_response_and_empty_final_cannot_complete() {
        let mut timer = SessionTiming::new(&Context::new(), 3, 15);
        timer.event(SpeechEvent::Start);
        timer.event(SpeechEvent::End);
        let anchor = timer.end.unwrap();
        timer.sent(false, true, true);
        assert!(!timer.complete);
        timer.event(SpeechEvent::Start);
        timer.event(SpeechEvent::End);
        timer.sent(true, true, true); // late first response, after a second START
        assert_eq!(timer.end, Some(anchor));
        assert!(!timer.complete);
        assert_eq!(timer.starts, 2);
    }
    #[test]
    fn successful_final_requires_successful_nonempty_send() {
        let mut timer = SessionTiming::new(&Context::new(), 3, 15);
        timer.event(SpeechEvent::Start);
        timer.event(SpeechEvent::End);
        timer.end = Some(Instant::now() - Duration::from_millis(40));
        timer.sent(true, false, true);
        assert!(timer.first.unwrap() >= 40.0);
        timer.sent(false, true, true);
        assert!(timer.complete);
        let mut independent = SessionTiming::new(&Context::new(), 3, 15);
        independent.vad.record(Duration::from_micros(250));
        independent.vad.record(Duration::from_micros(750));
        assert_eq!(independent.vad.count, 2);
        assert_eq!(independent.vad.sum, 1.0);
        assert_eq!(independent.vad.max, 0.75);
        assert_eq!(timer.vad.count, 0);
    }
}
