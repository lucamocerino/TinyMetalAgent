use std::pin::Pin;

use async_trait::async_trait;
use futures_core::Stream;
use futures_util::stream;
use thiserror::Error;

use crate::types::{GenerateRequest, ModelInfo, TokenEvent};

pub type Result<T> = std::result::Result<T, TinyAgentError>;
pub type TokenStream = Pin<Box<dyn Stream<Item = Result<TokenEvent>> + Send + 'static>>;

#[derive(Debug, Error)]
pub enum TinyAgentError {
    #[error("backend error: {0}")]
    Backend(String),
    #[error("configuration error: {0}")]
    Configuration(String),
    #[error("unsupported feature: {0}")]
    Unsupported(String),
}

#[async_trait]
pub trait InferenceBackend: Send + Sync {
    async fn models(&self) -> Result<Vec<ModelInfo>>;
    async fn generate(&self, request: GenerateRequest) -> Result<TokenStream>;
}

#[derive(Debug, Clone)]
pub struct StubBackend {
    model: ModelInfo,
}

impl StubBackend {
    pub fn new(model: ModelInfo) -> Self {
        Self { model }
    }
}

impl Default for StubBackend {
    fn default() -> Self {
        Self::new(ModelInfo::default_qwen_coder_stub())
    }
}

#[async_trait]
impl InferenceBackend for StubBackend {
    async fn models(&self) -> Result<Vec<ModelInfo>> {
        Ok(vec![self.model.clone()])
    }

    async fn generate(&self, request: GenerateRequest) -> Result<TokenStream> {
        if !request.tools.is_empty() {
            return Err(TinyAgentError::Unsupported(
                "tool execution is planned but not implemented in the stub backend".to_string(),
            ));
        }

        let last_user_message = request
            .messages
            .iter()
            .rev()
            .find(|message| message.role.is_user())
            .map(|message| message.content.as_str())
            .unwrap_or("");

        let text = format!(
            "TinyEngine stub backend is running for model {}. Use --backend metal with --package <model.tma> for the custom engine, or --backend llama with --gguf <model.gguf> for oracle checks. Last user message: {}",
            request.model, last_user_message
        );

        let mut events: Vec<Result<TokenEvent>> = text
            .split_whitespace()
            .map(|word| Ok(TokenEvent::token(format!("{word} "))))
            .collect();
        events.push(Ok(TokenEvent::finished("stop")));

        Ok(Box::pin(stream::iter(events)))
    }
}
