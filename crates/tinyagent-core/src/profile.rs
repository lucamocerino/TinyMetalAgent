use serde::{Deserialize, Serialize};

use crate::backend::{Result, TinyAgentError};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HardwareProfile {
    pub name: String,
    pub ctx_size: u32,
    pub batch_size: u32,
    pub ubatch_size: u32,
    pub threads: String,
    pub metal: bool,
    pub recommended_quant: String,
    pub max_loaded_models: u8,
    pub memory_limit_mb: u64,
}

impl HardwareProfile {
    pub fn mac_8gb() -> Self {
        Self {
            name: "mac-8gb".to_string(),
            ctx_size: 4096,
            batch_size: 128,
            ubatch_size: 64,
            threads: "auto".to_string(),
            metal: true,
            recommended_quant: "Q4_K_M".to_string(),
            max_loaded_models: 1,
            memory_limit_mb: 5500,
        }
    }

    pub fn validate_memory_estimate(&self, estimated_peak_mb: u64) -> Result<()> {
        if estimated_peak_mb > self.memory_limit_mb {
            return Err(TinyAgentError::Configuration(format!(
                "estimated model footprint is {estimated_peak_mb} MB, above {} MB limit for profile {}",
                self.memory_limit_mb, self.name
            )));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::HardwareProfile;

    #[test]
    fn mac_8gb_rejects_large_estimates() {
        let profile = HardwareProfile::mac_8gb();

        assert!(profile.validate_memory_estimate(5400).is_ok());
        assert!(profile.validate_memory_estimate(5600).is_err());
    }
}
