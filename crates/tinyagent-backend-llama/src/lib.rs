use std::{
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    time::Duration,
};

use async_trait::async_trait;
use futures_util::stream;
use serde::{Deserialize, Serialize};
use tinyagent_core::{
    ChatMessage, GenerateRequest, InferenceBackend, MessageRole, ModelInfo, Result, TinyAgentError,
    TokenEvent, TokenStream,
};

#[derive(Debug, Clone)]
pub struct LlamaServerConfig {
    pub executable: PathBuf,
    pub host: String,
    pub port: u16,
    pub model_path: PathBuf,
    pub profile: String,
    pub ctx_size: u32,
    pub batch_size: u32,
    pub ubatch_size: u32,
    pub gpu_layers: u32,
}

#[derive(Debug)]
pub struct LlamaServerBackend {
    config: LlamaServerConfig,
    model: ModelInfo,
    client: reqwest::Client,
    child: Option<Arc<Mutex<Child>>>,
}

impl LlamaServerBackend {
    pub async fn spawn(config: LlamaServerConfig, model: ModelInfo) -> Result<Self> {
        if !config.model_path.exists() {
            return Err(TinyAgentError::Configuration(format!(
                "GGUF model not found: {}",
                config.model_path.display()
            )));
        }

        let port = config.port.to_string();
        let ctx_size = config.ctx_size.to_string();
        let batch_size = config.batch_size.to_string();
        let ubatch_size = config.ubatch_size.to_string();
        let gpu_layers = config.gpu_layers.to_string();

        let child = Command::new(&config.executable)
            .arg("-m")
            .arg(&config.model_path)
            .arg("--host")
            .arg(&config.host)
            .arg("--port")
            .arg(&port)
            .arg("-c")
            .arg(&ctx_size)
            .arg("-b")
            .arg(&batch_size)
            .arg("-ub")
            .arg(&ubatch_size)
            .arg("-ngl")
            .arg(&gpu_layers)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| {
                TinyAgentError::Configuration(format!(
                    "failed to start llama-server at {}: {error}",
                    config.executable.display()
                ))
            })?;

        let backend = Self {
            config,
            model,
            client: reqwest::Client::new(),
            child: Some(Arc::new(Mutex::new(child))),
        };
        backend.wait_until_ready().await?;
        Ok(backend)
    }

    pub fn from_running_server(config: LlamaServerConfig, model: ModelInfo) -> Self {
        Self {
            config,
            model,
            client: reqwest::Client::new(),
            child: None,
        }
    }

    pub fn config(&self) -> &LlamaServerConfig {
        &self.config
    }

    fn base_url(&self) -> String {
        format!("http://{}:{}", self.config.host, self.config.port)
    }

    async fn wait_until_ready(&self) -> Result<()> {
        let health_url = format!("{}/health", self.base_url());
        for _ in 0..120 {
            if let Some(child) = &self.child {
                let mut child = child.lock().map_err(|_| {
                    TinyAgentError::Backend("llama-server process lock is poisoned".to_string())
                })?;
                if let Some(status) = child.try_wait().map_err(|error| {
                    TinyAgentError::Backend(format!("failed to inspect llama-server: {error}"))
                })? {
                    return Err(TinyAgentError::Backend(format!(
                        "llama-server exited before becoming ready with status {status}"
                    )));
                }
            }

            if let Ok(response) = self.client.get(&health_url).send().await {
                if response.status().is_success() {
                    return Ok(());
                }
            }

            tokio::time::sleep(Duration::from_millis(500)).await;
        }

        Err(TinyAgentError::Backend(format!(
            "llama-server did not become ready at {health_url}"
        )))
    }
}

#[async_trait]
impl InferenceBackend for LlamaServerBackend {
    async fn models(&self) -> Result<Vec<ModelInfo>> {
        let mut model = self.model.clone();
        model.backend = "llama-server".to_string();
        model.status = "configured".to_string();
        Ok(vec![model])
    }

    async fn generate(&self, request: GenerateRequest) -> Result<TokenStream> {
        if !request.tools.is_empty() {
            return Err(TinyAgentError::Unsupported(
                "tool calls are planned but not implemented in the Metal backend yet".to_string(),
            ));
        }

        let prompt = render_basic_prompt(&request.messages);
        let body = LlamaCompletionRequest {
            prompt,
            n_predict: request.max_tokens.unwrap_or(256),
            temperature: request.temperature.unwrap_or(0.7),
            stream: false,
        };

        let url = format!("{}/completion", self.base_url());
        let response = self
            .client
            .post(url)
            .json(&body)
            .send()
            .await
            .map_err(|error| {
                TinyAgentError::Backend(format!("llama-server request failed: {error}"))
            })?;

        let status = response.status();
        if !status.is_success() {
            let message = response
                .text()
                .await
                .unwrap_or_else(|_| "failed to read llama-server error response".to_string());
            return Err(TinyAgentError::Backend(format!(
                "llama-server returned {status}: {message}"
            )));
        }

        let completion: LlamaCompletionResponse = response.json().await.map_err(|error| {
            TinyAgentError::Backend(format!("failed to decode llama-server response: {error}"))
        })?;

        let mut events: Vec<Result<TokenEvent>> = completion
            .content
            .split_whitespace()
            .map(|word| Ok(TokenEvent::token(format!("{word} "))))
            .collect();
        events.push(Ok(TokenEvent::finished(if completion.stop {
            "stop"
        } else {
            "length"
        })));

        Ok(Box::pin(stream::iter(events)))
    }
}

impl Drop for LlamaServerBackend {
    fn drop(&mut self) {
        if let Some(child) = &self.child {
            if let Ok(mut child) = child.lock() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
}

#[derive(Debug, Serialize)]
struct LlamaCompletionRequest {
    prompt: String,
    n_predict: u32,
    temperature: f32,
    stream: bool,
}

#[derive(Debug, Deserialize)]
struct LlamaCompletionResponse {
    content: String,
    #[serde(default)]
    stop: bool,
}

fn render_basic_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for message in messages {
        let role = match message.role {
            MessageRole::System => "system",
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
            MessageRole::Tool => "tool",
        };
        prompt.push_str(role);
        prompt.push_str(":\n");
        prompt.push_str(&message.content);
        prompt.push_str("\n\n");
    }
    prompt.push_str("assistant:\n");
    prompt
}
