// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

// =============================================================================
//  SpiralReality Proprietary
// Copyright (c) 2025 SpiralReality. All Rights Reserved.
//
// NOTICE: This file contains confidential and proprietary information of
// SpiralReality. ANY USE, COPYING, MODIFICATION, DISTRIBUTION, DISPLAY,
// OR DISCLOSURE OF THIS FILE, IN WHOLE OR IN PART, IS STRICTLY PROHIBITED
// WITHOUT THE PRIOR WRITTEN CONSENT OF SPIRALREALITY.
//
// NO LICENSE IS GRANTED OR IMPLIED BY THIS FILE. THIS SOFTWARE IS PROVIDED
// "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
// NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL SPIRALREALITY OR ITS
// SUPPLIERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
// AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// =============================================================================

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

#[path = "microlocal.rs"]
#[doc(hidden)]
pub mod microlocal_impl;
#[doc(inline)]
pub use microlocal_impl as microlocal;

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
