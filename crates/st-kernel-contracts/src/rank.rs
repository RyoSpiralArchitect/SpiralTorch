// SPDX-License-Identifier: AGPL-3.0-or-later

//! Static WGPU rank geometry, shared by planning and device execution.

use thiserror::Error;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kind {
    TopK,
    MidK,
    BottomK,
}

impl Kind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TopK => "topk",
            Self::MidK => "midk",
            Self::BottomK => "bottomk",
        }
    }

    pub const fn as_uniform(self) -> u32 {
        match self {
            Self::TopK => 0,
            Self::MidK => 1,
            Self::BottomK => 2,
        }
    }
}

pub fn validate_wgpu_direct(kind: Kind, rows: u32, cols: u32, k: u32) -> Result<(), String> {
    if rows == 0 || cols == 0 || k == 0 {
        return Ok(());
    }
    if k > cols {
        return Err(format!(
            "wgpu {} exact path requires k <= cols, got k={k} cols={cols}",
            kind.as_str()
        ));
    }
    if cols > 256 {
        match kind {
            Kind::TopK | Kind::BottomK if k > 1 => return Err(format!(
                "wgpu {} exact path supports k == 1 for wide rows or cols <= 256, got cols={cols} k={k}",
                kind.as_str()
            )),
            Kind::MidK => return Err(format!(
                "wgpu midk exact path supports cols <= 256, got cols={cols} k={k}"
            )),
            _ => {}
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Exact2CeGeometry {
    pub tile_cols: u32,
    pub tile_stride: u32,
    pub tiles_x: u32,
    pub input_elements: u32,
    pub output_elements: u32,
    pub scratch_elements: u32,
    pub tile_state_elements: u32,
}

impl Exact2CeGeometry {
    pub fn try_new(rows: u32, cols: u32, k: u32, tile_cols: u32) -> Result<Self, PlanError> {
        if tile_cols == 0 {
            return Err(PlanError::ZeroTile);
        }
        if k > cols {
            return Err(PlanError::KExceedsColumns { k, cols });
        }
        let tile_cols = tile_cols.min(cols.max(1));
        let tile_stride = tile_cols
            .checked_next_power_of_two()
            .ok_or(PlanError::ArithmeticOverflow("tile stride"))?;
        let tiles_x = cols.div_ceil(tile_cols);
        let input_elements = checked_elements("input", u64::from(rows) * u64::from(cols))?;
        let output_elements = checked_elements("output", u64::from(rows) * u64::from(k))?;
        let tile_state_elements =
            checked_elements("tile state", u64::from(rows) * u64::from(tiles_x))?;
        let scratch_elements = checked_elements(
            "tile scratch",
            u64::from(tile_state_elements) * u64::from(tile_stride),
        )?;
        Ok(Self {
            tile_cols,
            tile_stride,
            tiles_x,
            input_elements,
            output_elements,
            scratch_elements,
            tile_state_elements,
        })
    }
}

fn checked_elements(name: &'static str, elements: u64) -> Result<u32, PlanError> {
    u32::try_from(elements).map_err(|_| PlanError::IndexSpaceTooLarge { name, elements })
}

#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum PlanError {
    #[error("rank-k two-command execution requires tile_cols > 0")]
    ZeroTile,
    #[error("rank-k requires k <= cols, got k={k} cols={cols}")]
    KExceedsColumns { k: u32, cols: u32 },
    #[error("rank-k two-command geometry overflowed while computing {0}")]
    ArithmeticOverflow(&'static str),
    #[error("{name} requires {elements} elements, exceeding the WGSL u32 index space")]
    IndexSpaceTooLarge { name: &'static str, elements: u64 },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_and_two_stage_have_different_wide_row_support() {
        for kind in [Kind::TopK, Kind::MidK, Kind::BottomK] {
            assert!(validate_wgpu_direct(kind, 2, 257, 8).is_err());
        }
        assert!(validate_wgpu_direct(Kind::TopK, 2, 4096, 1).is_ok());
        assert!(validate_wgpu_direct(Kind::MidK, 2, 4096, 1).is_err());
        let tiled = Exact2CeGeometry::try_new(2, 257, 8, 128).unwrap();
        assert_eq!(
            (tiled.tiles_x, tiled.tile_stride, tiled.scratch_elements),
            (3, 128, 768)
        );
        assert_eq!(
            Exact2CeGeometry::try_new(2, 257, 8, 512).unwrap(),
            Exact2CeGeometry::try_new(2, 257, 8, 1024).unwrap(),
        );
    }

    #[test]
    fn padded_scratch_overflow_is_rejected_before_allocation() {
        assert!(matches!(
            Exact2CeGeometry::try_new(65535, 65535, 1, 32769),
            Err(PlanError::IndexSpaceTooLarge {
                name: "tile scratch",
                ..
            })
        ));
    }
}
