//! Manual tracing only: no logging bridge and no exporter unless explicitly enabled.
use std::{collections::HashMap, time::Duration};

use opentelemetry::{
    Context, KeyValue, global,
    propagation::TextMapPropagator,
    trace::{Span, TraceContextExt, Tracer},
};
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::{
    Resource,
    propagation::TraceContextPropagator,
    trace::{BatchConfigBuilder, BatchSpanProcessor, Sampler, SdkTracerProvider},
};

pub fn init() -> Result<Option<SdkTracerProvider>, Box<dyn std::error::Error + Send + Sync>> {
    if std::env::var("NAILA_OTEL_ENABLED").as_deref() != Ok("1") {
        return Ok(None);
    }
    // OTLP/gRPC endpoint, e.g. http://collector:4317 (no /v1/traces).
    let endpoint = std::env::var("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")?;
    if endpoint.is_empty() {
        return Err("explicit OTLP/gRPC endpoint required".into());
    }
    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint)
        .with_timeout(Duration::from_secs(2))
        .build()?;
    // Tonic uses the active Tokio runtime; SDK owns queue/batching:
    // https://docs.rs/opentelemetry_sdk/0.31.0/opentelemetry_sdk/trace/struct.BatchSpanProcessor.html
    let processor = BatchSpanProcessor::builder(exporter)
        .with_batch_config(
            BatchConfigBuilder::default()
                .with_max_queue_size(2048)
                .with_max_export_batch_size(256)
                .with_scheduled_delay(Duration::from_millis(200))
                .build(),
        )
        .build();
    let provider = SdkTracerProvider::builder()
        .with_span_processor(processor)
        .with_sampler(Sampler::ParentBased(Box::new(Sampler::AlwaysOn)))
        .with_resource(
            Resource::builder()
                .with_service_name("naila.hub")
                .with_attributes([KeyValue::new(
                    "service.instance.id",
                    uuid::Uuid::new_v4().to_string(),
                )])
                .build(),
        )
        .build();
    global::set_tracer_provider(provider.clone());
    Ok(Some(provider))
}

pub fn shutdown(provider: Option<SdkTracerProvider>) {
    if let Some(provider) = provider {
        // Shutdown drains queued spans, within the shared five-second deadline.
        if provider
            .shutdown_with_timeout(Duration::from_secs(5))
            .is_err()
        {
            tracing::warn!("OTel shutdown incomplete");
        }
    }
}

pub fn extract(traceparent: &str) -> Context {
    TraceContextPropagator::new().extract(&HashMap::from([(
        "traceparent".to_owned(),
        traceparent.to_owned(),
    )]))
}

pub fn inject(context: &Context) -> String {
    let mut carrier = HashMap::new();
    TraceContextPropagator::new().inject_context(context, &mut carrier);
    carrier.remove("traceparent").unwrap_or_default()
}

/// An accepted VAD utterance gets its own immutable parent, never a device-global lookup.
/// This short acceptance span ends here; downstream tasks own their own spans.
pub fn accepted_utterance(parent: &Context) -> Context {
    let mut span = global::tracer("naila.hub").start_with_context("hub.utterance", parent);
    span.end();
    parent.with_span(span)
}

#[cfg(test)]
mod tests {
    use super::*;
    use opentelemetry::trace::FutureExt;

    #[tokio::test]
    async fn exporter_lifecycle() {
        let provider = init().unwrap();
        if provider.is_some() {
            let parent = extract("00-0123456789abcdef0123456789abcdef-0123456789abcdef-01");
            let utterance = accepted_utterance(&parent);
            assert!(inject(&utterance).contains("0123456789abcdef0123456789abcdef"));
        }
        tokio::task::spawn_blocking(move || shutdown(provider))
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn delayed_tasks_keep_explicit_parents() {
        let first = "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01";
        let second = "00-fedcba9876543210fedcba9876543210-fedcba9876543210-01";
        let mut tasks = Vec::new();
        for value in [first, second] {
            tasks.push(tokio::spawn(
                async {
                    tokio::task::yield_now().await;
                    inject(&Context::current())
                }
                .with_context(extract(value)),
            ));
        }
        assert_eq!(tasks.remove(0).await.unwrap(), first);
        assert_eq!(tasks.remove(0).await.unwrap(), second);
        assert!(inject(&extract("")).is_empty());
        assert!(inject(&extract("invalid")).is_empty());
        assert!(inject(&Context::current()).is_empty());
    }
}
