use std::{
    fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub type Result<T> = std::result::Result<T, TmaFormatError>;

#[derive(Debug, Error)]
pub enum TmaFormatError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TmaPackageMetadata {
    pub format_version: u32,
    pub model_id: String,
    pub architecture: ModelArchitecture,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qwen_config: Option<QwenConfig>,
    pub source: SourceFormat,
    pub source_path: String,
    pub tokenizer_path: Option<String>,
    pub tensors: Vec<TensorDescriptor>,
    pub status: PackageStatus,
}

impl TmaPackageMetadata {
    pub fn scaffold(
        model_id: impl Into<String>,
        architecture: ModelArchitecture,
        source: SourceFormat,
        source_path: impl Into<String>,
    ) -> Self {
        Self {
            format_version: 1,
            model_id: model_id.into(),
            architecture,
            qwen_config: None,
            source,
            source_path: source_path.into(),
            tokenizer_path: None,
            tensors: Vec::new(),
            status: PackageStatus::MetadataOnly,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum PackageStatus {
    MetadataOnly,
    Converted,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum SourceFormat {
    HuggingFace,
    Gguf,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ModelArchitecture {
    Qwen25,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct QwenConfig {
    pub hidden_size: u64,
    pub intermediate_size: u64,
    pub num_hidden_layers: u64,
    pub num_attention_heads: u64,
    pub num_key_value_heads: u64,
    pub head_dim: u64,
    pub vocab_size: u64,
    pub max_position_embeddings: u64,
    pub rope_theta: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TensorDescriptor {
    pub name: String,
    pub dtype: TensorDType,
    pub shape: Vec<u64>,
    pub file: PathBuf,
    pub byte_offset: u64,
    pub byte_len: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum TensorDType {
    F16,
    F32,
    Q8Tma,
    Q4Tma,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TmaPackageInspection {
    pub package_dir: PathBuf,
    pub metadata: TmaPackageMetadata,
    pub tokenizer_exists: bool,
    pub tensor_count: usize,
    pub missing_tensors: Vec<PathBuf>,
    pub total_tensor_bytes: u64,
}

pub fn metadata_path(package_dir: impl AsRef<Path>) -> PathBuf {
    package_dir.as_ref().join("metadata.json")
}

pub fn read_metadata(package_dir: impl AsRef<Path>) -> Result<TmaPackageMetadata> {
    let data = fs::read_to_string(metadata_path(package_dir))?;
    Ok(serde_json::from_str(&data)?)
}

pub fn write_metadata(package_dir: impl AsRef<Path>, metadata: &TmaPackageMetadata) -> Result<()> {
    fs::create_dir_all(package_dir.as_ref())?;
    fs::create_dir_all(package_dir.as_ref().join("tensors"))?;
    let data = serde_json::to_string_pretty(metadata)?;
    fs::write(metadata_path(package_dir), data)?;
    Ok(())
}

pub fn inspect_package(package_dir: impl AsRef<Path>) -> Result<TmaPackageInspection> {
    let package_dir = package_dir.as_ref();
    let metadata = read_metadata(package_dir)?;
    let tokenizer_exists = metadata
        .tokenizer_path
        .as_ref()
        .map(|path| package_dir.join(path).is_file())
        .unwrap_or(false);

    let mut missing_tensors = Vec::new();
    let mut total_tensor_bytes = 0_u64;
    for tensor in &metadata.tensors {
        let tensor_path = package_dir.join(&tensor.file);
        match fs::metadata(&tensor_path) {
            Ok(file_metadata) if file_metadata.is_file() => {
                total_tensor_bytes += file_metadata.len();
            }
            _ => missing_tensors.push(tensor.file.clone()),
        }
    }

    Ok(TmaPackageInspection {
        package_dir: package_dir.to_path_buf(),
        tensor_count: metadata.tensors.len(),
        metadata,
        tokenizer_exists,
        missing_tensors,
        total_tensor_bytes,
    })
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::{
        inspect_package, read_metadata, write_metadata, ModelArchitecture, SourceFormat,
        TmaPackageMetadata,
    };

    #[test]
    fn metadata_roundtrip() {
        let package_dir =
            std::env::temp_dir().join(format!("tinyagent-tma-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&package_dir);

        let metadata = TmaPackageMetadata::scaffold(
            "qwen2.5-coder-1.5b",
            ModelArchitecture::Qwen25,
            SourceFormat::HuggingFace,
            "/models/qwen",
        );
        write_metadata(&package_dir, &metadata).expect("write metadata");
        let loaded = read_metadata(&package_dir).expect("read metadata");

        assert_eq!(loaded, metadata);
        let inspection = inspect_package(&package_dir).expect("inspect metadata-only package");
        assert_eq!(inspection.tensor_count, 0);
        assert!(!inspection.tokenizer_exists);
        let _ = fs::remove_dir_all(&package_dir);
    }
}
