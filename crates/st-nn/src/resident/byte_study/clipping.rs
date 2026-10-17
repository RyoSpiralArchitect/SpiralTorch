//! Request-bound clipping, independent of the geometry design and optimizer.
use super::*;
use st_kernel_contracts::gradient_clip::GlobalNormClip;

pub(super) const REQUEST_SCHEMA: &str = "spiraltorch.byte_corpus.request.v10";

#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(tag = "kind", deny_unknown_fields)]
pub(super) enum GradientClip {
    #[serde(rename = "disabled.v1")]
    Disabled {},
    #[serde(rename = "active_global_l2.v1")]
    ActiveGlobalL2 {
        #[serde(deserialize_with = "optimizer::recipe_scalar")]
        max_norm: f32,
    },
}

impl GradientClip {
    pub(super) fn clip(self) -> Result<Option<GlobalNormClip>> {
        Ok(match self {
            Self::Disabled {} => None,
            Self::ActiveGlobalL2 { max_norm } => Some(GlobalNormClip::new(max_norm)?),
        })
    }
}

pub(super) fn explicit_recipe<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> std::result::Result<Option<GradientClip>, D::Error> {
    GradientClip::deserialize(d).map(Some)
}

impl Request {
    pub(super) fn gradient_clip_recipe(&self) -> GradientClip {
        self.gradient_clip.unwrap_or(GradientClip::Disabled {})
    }
}

pub(super) fn validate_spec(r: &Request) -> Result<()> {
    if r.schema == REQUEST_SCHEMA {
        if r.gradient_clip.is_none() {
            return Err("v10 requires an explicit gradient_clip recipe".into());
        }
    } else if r.gradient_clip.is_some() {
        return Err("gradient_clip metadata belongs to v10 only".into());
    }
    r.gradient_clip_recipe().clip()?;
    Ok(())
}

#[cfg(test)]
mod tests;
