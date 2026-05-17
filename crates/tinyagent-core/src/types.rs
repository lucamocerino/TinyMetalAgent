use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

impl MessageRole {
    pub fn is_user(&self) -> bool {
        matches!(self, Self::User)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: MessageRole,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GenerateRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub stream: bool,
    #[serde(default)]
    pub tools: Vec<ToolDefinition>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenEvent {
    pub token: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

impl TokenEvent {
    pub fn token(token: String) -> Self {
        Self {
            token,
            finish_reason: None,
        }
    }

    pub fn finished(reason: impl Into<String>) -> Self {
        Self {
            token: String::new(),
            finish_reason: Some(reason.into()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelInfo {
    pub id: String,
    pub family: String,
    pub context_size: u32,
    pub recommended_context_8gb: u32,
    pub quantization: String,
    pub backend: String,
    pub status: String,
}

impl ModelInfo {
    pub fn default_qwen_coder_stub() -> Self {
        Self {
            id: "qwen2.5-coder-1.5b".to_string(),
            family: "qwen2.5-coder".to_string(),
            context_size: 32768,
            recommended_context_8gb: 4096,
            quantization: "Q4_K_M".to_string(),
            backend: "stub".to_string(),
            status: "stub".to_string(),
        }
    }
}
