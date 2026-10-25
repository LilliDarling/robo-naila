use opentelemetry::Value;
use opentelemetry_sdk::trace::SpanData;

// OTel 0.31 SpanData exposes attributes as Vec<KeyValue>, without a numeric lookup API:
// https://docs.rs/opentelemetry_sdk/0.31.0/opentelemetry_sdk/trace/struct.SpanData.html
pub fn number(span: &SpanData, key: &str) -> f64 {
    match &span.attributes.iter().find(|kv| kv.key.as_str() == key).unwrap().value {
        Value::F64(v) => *v,
        Value::I64(v) => *v as f64,
        other => panic!("{other:?}"),
    }
}
