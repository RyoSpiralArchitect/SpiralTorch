// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Category-flavoured theoretical utilities for the observability pipeline.
//!
//! The modules hosted here encode the Pólya-counting driven recursion that
//! underpins SpiralTorch's observation DAG compression experiments and the
//! microlocal interface gauges used to stabilise boundary detection without
//! fixing an external label.

#[path = "observability.rs"]
#[doc(hidden)]
pub mod observability_impl;
#[doc(inline)]
pub use observability_impl as observability;

#[path = "observation.rs"]
#[doc(hidden)]
pub mod observation_impl;
#[doc(inline)]
pub use observation_impl as observation;

#[path = "microlocal.rs"]
#[doc(hidden)]
pub mod microlocal_impl;
#[doc(inline)]
pub use microlocal_impl as microlocal;

pub mod microlocal_bank;

#[path = "scale_persistence.rs"]
#[doc(hidden)]
pub mod scale_persistence_impl;
#[doc(inline)]
pub use scale_persistence_impl as scale_persistence;

#[path = "macro.rs"]
#[doc(hidden)]
pub mod macro_impl;
#[doc(inline)]
pub use macro_impl as macro_model;

#[cfg(feature = "experimental_zpulse")]
pub mod microlocal_experimental;

#[path = "maxwell.rs"]
#[doc(hidden)]
pub mod maxwell_impl;
#[doc(inline)]
pub use maxwell_impl as maxwell;


#[path = "zpulse.rs"]
#[doc(hidden)]
pub mod zpulse_impl;
#[doc(inline)]
pub use zpulse_impl as zpulse;

#[path = "spiral_dynamics.rs"]
#[doc(hidden)]
pub mod spiral_dynamics_impl;
#[doc(inline)]
pub use spiral_dynamics_impl as spiral_dynamics;

#[path = "sync_bridge.rs"]
#[doc(hidden)]
pub mod sync_bridge_impl;
#[doc(inline)]
pub use sync_bridge_impl as sync_bridge;

#[cfg(feature = "psi")]
#[path = "psi_sync.rs"]
#[doc(hidden)]
pub mod psi_sync_impl;
#[cfg(feature = "psi")]
#[doc(inline)]
pub use psi_sync_impl as psi_sync;

#[path = "stv.rs"]
#[doc(hidden)]
pub mod stv_impl;
#[doc(inline)]
pub use stv_impl as stv;

#[path = "general_relativity.rs"]
#[doc(hidden)]
pub mod general_relativity_impl;
#[doc(inline)]
pub use general_relativity_impl as general_relativity;
