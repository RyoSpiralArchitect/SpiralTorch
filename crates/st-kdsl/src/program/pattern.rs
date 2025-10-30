// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::{Ctx, ExprNode, Program, Value};

#[derive(Clone, Debug)]
pub(super) struct MatchArm {
    pub(super) patterns: Vec<MatchPattern>,
    pub(super) guard: Option<ExprNode>,
    pub(super) expr: ExprNode,
}

impl MatchArm {
    pub(super) fn fold(self) -> Self {
        let patterns = self.patterns.into_iter().map(MatchPattern::fold).collect();
        let guard = self.guard.map(|expr| expr.fold());
        let expr = self.expr.fold();
        MatchArm {
            patterns,
            guard,
            expr,
        }
    }

    pub(super) fn matches(
        &self,
        program: &Program,
        ctx: &Ctx,
        locals: &[Value],
        scrutinee: &Value,
    ) -> bool {
        let pattern_match = self
            .patterns
            .iter()
            .any(|pattern| pattern.matches(program, ctx, locals, scrutinee));

        if !pattern_match {
            return false;
        }

        if let Some(guard) = &self.guard {
            guard.evaluate(program, ctx, locals).as_bool()
        } else {
            true
        }
    }

    pub(super) fn const_result(&self, scrutinee: &Value) -> Option<Value> {
        let mut matched = false;
        for pattern in &self.patterns {
            match pattern.const_matches(scrutinee) {
                Some(true) => {
                    matched = true;
                    break;
                }
                Some(false) => {}
                None => return None,
            }
        }

        if !matched {
            return None;
        }

        if let Some(guard) = &self.guard {
            if !guard.const_value()?.as_bool() {
                return None;
            }
        }

        self.expr.const_value()
    }

    pub(super) fn evaluate(&self, program: &Program, ctx: &Ctx, locals: &[Value]) -> Value {
        self.expr.evaluate(program, ctx, locals)
    }

    pub(super) fn has_wildcard(&self) -> bool {
        self.patterns
            .iter()
            .any(|pattern| matches!(pattern, MatchPattern::Wildcard))
    }
}

#[derive(Clone, Debug)]
pub(super) enum MatchPattern {
    Wildcard,
    Expr(ExprNode),
    Range {
        start: ExprNode,
        end: ExprNode,
        inclusive: bool,
    },
}

impl MatchPattern {
    fn fold(self) -> Self {
        match self {
            MatchPattern::Wildcard => MatchPattern::Wildcard,
            MatchPattern::Expr(expr) => MatchPattern::Expr(expr.fold()),
            MatchPattern::Range {
                start,
                end,
                inclusive,
            } => MatchPattern::Range {
                start: start.fold(),
                end: end.fold(),
                inclusive,
            },
        }
    }

    fn matches(&self, program: &Program, ctx: &Ctx, locals: &[Value], scrutinee: &Value) -> bool {
        match self {
            MatchPattern::Wildcard => true,
            MatchPattern::Expr(expr) => {
                let candidate = expr.evaluate(program, ctx, locals);
                candidate.equals(scrutinee)
            }
            MatchPattern::Range {
                start,
                end,
                inclusive,
            } => {
                let lo = start.evaluate(program, ctx, locals).as_f64();
                let hi = end.evaluate(program, ctx, locals).as_f64();
                let value = scrutinee.as_f64();
                if *inclusive {
                    value >= lo && value <= hi
                } else {
                    value >= lo && value < hi
                }
            }
        }
    }

    fn const_matches(&self, scrutinee: &Value) -> Option<bool> {
        match self {
            MatchPattern::Wildcard => Some(true),
            MatchPattern::Expr(expr) => {
                let candidate = expr.const_value()?;
                Some(candidate.equals(scrutinee))
            }
            MatchPattern::Range {
                start,
                end,
                inclusive,
            } => {
                let lo = start.const_value()?.as_f64();
                let hi = end.const_value()?.as_f64();
                let value = scrutinee.as_f64();
                if *inclusive {
                    Some(value >= lo && value <= hi)
                } else {
                    Some(value >= lo && value < hi)
                }
            }
        }
    }
}
