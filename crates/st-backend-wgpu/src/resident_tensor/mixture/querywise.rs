//! Query-local routing reuses the checked wide-reduction mixture kernels.
use super::*;
use st_kernel_contracts::mixture::QuerywiseMixtureSpec;

#[derive(Clone, Debug)]
pub struct ResidentQuerywiseMixture {
    spec: QuerywiseMixtureSpec,
    grouped: ResidentHeadwiseMixture,
    output: ResidentTensor,
    weights: ResidentTensor,
}

#[derive(Clone, Debug)]
pub struct ResidentQuerywiseMixtureVjp {
    components: ResidentTensor,
    logits: ResidentTensor,
}

impl ResidentQuerywiseMixtureVjp {
    pub fn components(&self) -> &ResidentTensor {
        &self.components
    }
    pub fn logits(&self) -> &ResidentTensor {
        &self.logits
    }
}

impl ResidentTensor {
    /// Mix [B,H,Q,S,K] components with [B,H,Q,K] logits, yielding [B,H,Q,S].
    /// Values and VJPs stay resident. No causal mask or claim about causal
    /// logit provenance is implicit; the attention/router producer owns both.
    pub fn querywise_mixture(
        &self,
        logits: &Self,
    ) -> Result<ResidentQuerywiseMixture, TensorError> {
        let shape = <[usize; 5]>::try_from(self.layout.shape()).map_err(|_| MixtureError::Shape)?;
        let spec = QuerywiseMixtureSpec::new(shape)?;
        let [batch, heads, queries, keys, count] = shape;
        if logits.layout.shape() != [batch, heads, queries, count] {
            return Err(MixtureError::Length.into());
        }
        let context = self.device.runtime().context();
        logits.require_context(context)?;
        preflight(spec.grouped_spec(), &context.device().limits())?;
        let grouped = self
            .contiguous()?
            .reshape(&spec.grouped_spec().shape())?
            .headwise_mixture(
                &logits
                    .contiguous()?
                    .reshape(&[batch * heads * queries, count])?,
            )?;
        let output = grouped.output().reshape(&[batch, heads, queries, keys])?;
        let weights = grouped.weights().reshape(&[batch, heads, queries, count])?;
        Ok(ResidentQuerywiseMixture {
            spec,
            grouped,
            output,
            weights,
        })
    }
}

impl ResidentQuerywiseMixture {
    pub fn output(&self) -> &ResidentTensor {
        &self.output
    }
    pub fn weights(&self) -> &ResidentTensor {
        &self.weights
    }
    /// Component derivatives retain all five axes; logit derivatives reduce
    /// over keys only and retain batch/head/query identity.
    pub fn backward(
        &self,
        seed: &ResidentTensor,
    ) -> Result<ResidentQuerywiseMixtureVjp, TensorError> {
        if seed.layout.shape() != self.output.layout.shape() {
            return Err(MixtureError::Length.into());
        }
        seed.require_context(self.output.device.runtime().context())?;
        let grouped = self.grouped.backward(
            &seed
                .contiguous()?
                .reshape(self.grouped.output().layout().shape())?,
        )?;
        let [batch, heads, queries, _, count] = self.spec.shape();
        Ok(ResidentQuerywiseMixtureVjp {
            components: grouped.components().reshape(&self.spec.shape())?,
            logits: grouped.logits().reshape(&[batch, heads, queries, count])?,
        })
    }
}
