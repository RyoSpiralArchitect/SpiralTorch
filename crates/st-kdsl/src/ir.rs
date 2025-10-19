// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Lightweight IR nodes that cover subgroup-centric primitives and can be
//! lowered into WGSL snippets.  The intent is to keep the representation close
//! to how kernels are written in the SpiralTorch WGPU backend so the same
//! structures can back code generation, autotuning templates, and future
//! telemetry bridges.

use std::fmt;

/// Scalar types supported by the subgroup IR.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScalarType {
    Bool,
    F16,
    F32,
    I32,
    U32,
}

impl ScalarType {
    /// Returns the WGSL spelling for the scalar.
    pub fn wgsl(self) -> &'static str {
        match self {
            ScalarType::Bool => "bool",
            ScalarType::F16 => "f16",
            ScalarType::F32 => "f32",
            ScalarType::I32 => "i32",
            ScalarType::U32 => "u32",
        }
    }
}

/// Minimal expression node used by subgroup statements.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Expr(String);

impl Expr {
    /// Creates an identifier expression.
    pub fn ident(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Creates a literal expression (including suffixes such as `u`).
    pub fn literal(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Creates a raw expression.  Callers are responsible for providing valid
    /// WGSL snippets.
    pub fn raw(src: impl Into<String>) -> Self {
        Self(src.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for Expr {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

impl From<String> for Expr {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Subgroup primitive captured by the IR.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SubgroupOp {
    ShuffleXor { value: Expr, mask: Expr },
    ShuffleUp { value: Expr, delta: Expr },
    ShuffleDown { value: Expr, delta: Expr },
    BroadcastFirst { value: Expr },
    Ballot { predicate: Expr },
    All { predicate: Expr },
    Any { predicate: Expr },
    ReduceAdd { value: Expr },
    ReduceMin { value: Expr },
    ReduceMax { value: Expr },
}

impl SubgroupOp {
    fn emit(&self) -> String {
        match self {
            SubgroupOp::ShuffleXor { value, mask } => {
                format!("subgroupShuffleXor({}, {})", value, mask)
            }
            SubgroupOp::ShuffleUp { value, delta } => {
                format!("subgroupShuffleUp({}, {})", value, delta)
            }
            SubgroupOp::ShuffleDown { value, delta } => {
                format!("subgroupShuffleDown({}, {})", value, delta)
            }
            SubgroupOp::BroadcastFirst { value } => {
                format!("subgroupBroadcastFirst({})", value)
            }
            SubgroupOp::Ballot { predicate } => {
                format!("subgroupBallot({})", predicate)
            }
            SubgroupOp::All { predicate } => format!("subgroupAll({})", predicate),
            SubgroupOp::Any { predicate } => format!("subgroupAny({})", predicate),
            SubgroupOp::ReduceAdd { value } => format!("subgroupAdd({})", value),
            SubgroupOp::ReduceMin { value } => format!("subgroupMin({})", value),
            SubgroupOp::ReduceMax { value } => format!("subgroupMax({})", value),
        }
    }
}

/// Statement that binds the result of a subgroup primitive to a named value.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SubgroupStmt {
    pub name: String,
    pub ty: ScalarType,
    pub op: SubgroupOp,
}

impl SubgroupStmt {
    pub fn new(name: impl Into<String>, ty: ScalarType, op: SubgroupOp) -> Self {
        Self {
            name: name.into(),
            ty,
            op,
        }
    }

    fn emit(&self) -> String {
        format!(
            "let {}: {} = {};",
            self.name,
            self.ty.wgsl(),
            self.op.emit()
        )
    }
}

/// Container that groups subgroup statements and emits a WGSL helper function.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SubgroupModule {
    pub statements: Vec<SubgroupStmt>,
}

impl SubgroupModule {
    pub fn new() -> Self {
        Self {
            statements: Vec::new(),
        }
    }

    pub fn push(&mut self, stmt: SubgroupStmt) {
        self.statements.push(stmt);
    }

    /// Emits a WGSL helper function with `enable subgroups;` included.
    pub fn emit_function(&self, name: &str) -> String {
        let mut out = String::from("enable subgroups;\n\n");
        out.push_str(&format!("fn {}() {{\n", name));
        for stmt in &self.statements {
            out.push_str("    ");
            out.push_str(&stmt.emit());
            out.push('\n');
        }
        out.push_str("}\n");
        out
    }

    /// Emits the statements without wrapping them in a helper.
    pub fn emit_statements(&self) -> String {
        let mut out = String::new();
        for stmt in &self.statements {
            out.push_str(&stmt.emit());
            out.push('\n');
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emits_shuffle_reduce_snippet() {
        let mut module = SubgroupModule::new();
        module.push(SubgroupStmt::new(
            "rotated",
            ScalarType::F32,
            SubgroupOp::ShuffleXor {
                value: Expr::ident("value"),
                mask: Expr::literal("16u"),
            },
        ));
        module.push(SubgroupStmt::new(
            "sum",
            ScalarType::F32,
            SubgroupOp::ReduceAdd {
                value: Expr::ident("rotated"),
            },
        ));
        let wgsl = module.emit_function("sg_demo");
        assert!(wgsl.contains("enable subgroups"));
        assert!(wgsl.contains("let rotated: f32 = subgroupShuffleXor(value, 16u);"));
        assert!(wgsl.contains("let sum: f32 = subgroupAdd(rotated);"));
    }

    #[test]
    fn emits_predicate_helpers() {
        let mut module = SubgroupModule::new();
        module.push(SubgroupStmt::new(
            "mask",
            ScalarType::U32,
            SubgroupOp::Ballot {
                predicate: Expr::raw("gid.x < 16u"),
            },
        ));
        module.push(SubgroupStmt::new(
            "all_active",
            ScalarType::Bool,
            SubgroupOp::All {
                predicate: Expr::ident("valid"),
            },
        ));
        let wgsl = module.emit_statements();
        assert!(wgsl.contains("let mask: u32 = subgroupBallot(gid.x < 16u);"));
        assert!(wgsl.contains("let all_active: bool = subgroupAll(valid);"));
    }
}
