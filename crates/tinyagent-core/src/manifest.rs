use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelManifest {
    pub id: String,
    pub family: String,
    pub gguf_path: Option<String>,
    pub quantization: String,
    pub context_size: u32,
    pub recommended_context_8gb: u32,
    pub estimated_weight_mb: u64,
    pub estimated_kv_cache_mb_4k: u64,
    pub backend: String,
}

impl ModelManifest {
    pub fn estimated_peak_mb(&self, ctx_size: u32) -> u64 {
        let context_ratio = ctx_size.max(1) as f64 / 4096.0;
        self.estimated_weight_mb
            + (self.estimated_kv_cache_mb_4k as f64 * context_ratio).ceil() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::ModelManifest;

    #[test]
    fn scales_kv_estimate_with_context() {
        let manifest = ModelManifest {
            id: "test".to_string(),
            family: "qwen".to_string(),
            gguf_path: None,
            quantization: "Q4_K_M".to_string(),
            context_size: 32768,
            recommended_context_8gb: 4096,
            estimated_weight_mb: 1000,
            estimated_kv_cache_mb_4k: 500,
            backend: "llama-server".to_string(),
        };

        assert_eq!(manifest.estimated_peak_mb(4096), 1500);
        assert_eq!(manifest.estimated_peak_mb(8192), 2000);
    }
}
