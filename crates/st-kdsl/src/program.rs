// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::fmt;

use crate::{Ctx, Err, Hard, Out, SoftRule};

use std::collections::HashMap;

mod pattern;

use pattern::{MatchArm, MatchPattern};

/// Compiled representation of a SpiralK DSL program.
#[derive(Clone, Debug)]
pub struct Program {
    stmts: Vec<Stmt>,
    locals: usize,
}

const MAX_LOOP_ITERATIONS: usize = 65_536;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FlowSignal {
    None,
    Break,
    Continue,
}

impl Program {
    /// Parses the provided source string into a [`Program`].
    pub fn parse(src: &str) -> Result<Self, Err> {
        let toks = lex(src)?;
        let mut parser = Parser::new(toks);
        let output = parser.parse_program()?;
        Ok(Self {
            stmts: output.stmts,
            locals: output.locals,
        })
    }

    /// Evaluates the program for the given [`Ctx`].
    pub fn evaluate(&self, ctx: &Ctx) -> Out {
        let mut hard = Hard::default();
        let mut soft = Vec::<SoftRule>::new();
        let mut locals = vec![Value::default(); self.locals];

        let signal = self.execute_block(&self.stmts, ctx, &mut locals, &mut hard, &mut soft);
        debug_assert_eq!(signal, FlowSignal::None, "top-level control flow escape");

        Out { hard, soft }
    }

    fn execute_block(
        &self,
        block: &[Stmt],
        ctx: &Ctx,
        locals: &mut [Value],
        hard: &mut Hard,
        soft: &mut Vec<SoftRule>,
    ) -> FlowSignal {
        for stmt in block {
            let signal = match stmt {
                Stmt::Assign(assign) => {
                    assign.apply(self, ctx, locals, hard);
                    FlowSignal::None
                }
                Stmt::Let(binding) => {
                    binding.store(self, ctx, locals);
                    FlowSignal::None
                }
                Stmt::Set(update) => {
                    update.store(self, ctx, locals);
                    FlowSignal::None
                }
                Stmt::Soft(rule) => {
                    rule.apply(self, ctx, locals, soft);
                    FlowSignal::None
                }
                Stmt::If(branch) => {
                    let cond = branch
                        .const_cond
                        .unwrap_or_else(|| branch.cond.evaluate(self, ctx, locals).as_bool());
                    if cond {
                        self.execute_block(&branch.then_branch, ctx, locals, hard, soft)
                    } else {
                        self.execute_block(&branch.else_branch, ctx, locals, hard, soft)
                    }
                }
                Stmt::While(loop_stmt) => {
                    loop_stmt.run(self, ctx, locals, hard, soft);
                    FlowSignal::None
                }
                Stmt::For(loop_stmt) => {
                    loop_stmt.run(self, ctx, locals, hard, soft);
                    FlowSignal::None
                }
                Stmt::Break => FlowSignal::Break,
                Stmt::Continue => FlowSignal::Continue,
            };

            match signal {
                FlowSignal::None => {}
                FlowSignal::Break | FlowSignal::Continue => return signal,
            }
        }
        FlowSignal::None
    }
}

#[derive(Clone, Debug)]
enum Stmt {
    Assign(AssignStmt),
    Let(LetStmt),
    Set(SetStmt),
    Soft(SoftStmt),
    If(IfStmt),
    While(WhileStmt),
    For(ForStmt),
    Break,
    Continue,
}

#[derive(Clone, Debug)]
struct AssignStmt {
    field: Field,
    expr: ExprNode,
    const_value: Option<Value>,
}

impl AssignStmt {
    fn apply(&self, program: &Program, ctx: &Ctx, locals: &[Value], hard: &mut Hard) {
        let value = self
            .const_value
            .unwrap_or_else(|| self.expr.evaluate(program, ctx, locals));
        match self.field {
            Field::U2 => hard.use_2ce = Some(value.as_bool()),
            Field::Wg => hard.wg = Some(value.as_u32()),
            Field::Kl => hard.kl = Some(value.as_u32()),
            Field::Ch => hard.ch = Some(value.as_u32()),
            Field::Algo => hard.algo = Some(value.as_u8()),
            Field::Midk => hard.midk = Some(value.as_u8()),
            Field::Bottomk => hard.bottomk = Some(value.as_u8()),
            Field::Ctile => hard.ctile = Some(value.as_u32()),
            Field::TileCols => hard.tile_cols = Some(value.as_u32()),
            Field::Radix => hard.radix = Some(value.as_u32()),
            Field::Segments => hard.segments = Some(value.as_u32()),
        }
    }
}

#[derive(Clone, Debug)]
struct LetStmt {
    id: usize,
    expr: ExprNode,
    const_value: Option<Value>,
}

impl LetStmt {
    fn store(&self, program: &Program, ctx: &Ctx, locals: &mut [Value]) {
        let value = self
            .const_value
            .unwrap_or_else(|| self.expr.evaluate(program, ctx, locals));
        locals[self.id] = value;
    }
}

#[derive(Clone, Debug)]
struct SetStmt {
    id: usize,
    expr: ExprNode,
    const_value: Option<Value>,
}

impl SetStmt {
    fn store(&self, program: &Program, ctx: &Ctx, locals: &mut [Value]) {
        let value = self
            .const_value
            .unwrap_or_else(|| self.expr.evaluate(program, ctx, locals));
        locals[self.id] = value;
    }
}

#[derive(Clone, Debug)]
struct SoftStmt {
    field: Field,
    value_expr: ExprNode,
    weight_expr: ExprNode,
    cond_expr: ExprNode,
    const_value: Option<Value>,
    const_weight: Option<f32>,
    const_cond: Option<bool>,
}

impl SoftStmt {
    fn apply(&self, program: &Program, ctx: &Ctx, locals: &[Value], soft: &mut Vec<SoftRule>) {
        let should_apply = self
            .const_cond
            .unwrap_or_else(|| self.cond_expr.evaluate(program, ctx, locals).as_bool());
        if !should_apply {
            return;
        }

        let value = self
            .const_value
            .unwrap_or_else(|| self.value_expr.evaluate(program, ctx, locals));
        let weight = self
            .const_weight
            .unwrap_or_else(|| self.weight_expr.evaluate(program, ctx, locals).as_f64() as f32);

        match self.field {
            Field::U2 => soft.push(SoftRule::U2 {
                val: value.as_bool(),
                w: weight,
            }),
            Field::Wg => soft.push(SoftRule::Wg {
                val: value.as_u32(),
                w: weight,
            }),
            Field::Kl => soft.push(SoftRule::Kl {
                val: value.as_u32(),
                w: weight,
            }),
            Field::Ch => soft.push(SoftRule::Ch {
                val: value.as_u32(),
                w: weight,
            }),
            Field::Algo => soft.push(SoftRule::Algo {
                val: value.as_u8(),
                w: weight,
            }),
            Field::Midk => soft.push(SoftRule::Midk {
                val: value.as_u8(),
                w: weight,
            }),
            Field::Bottomk => soft.push(SoftRule::Bottomk {
                val: value.as_u8(),
                w: weight,
            }),
            Field::Ctile => soft.push(SoftRule::Ctile {
                val: value.as_u32(),
                w: weight,
            }),
            Field::TileCols => soft.push(SoftRule::TileCols {
                val: value.as_u32(),
                w: weight,
            }),
            Field::Radix => soft.push(SoftRule::Radix {
                val: value.as_u32(),
                w: weight,
            }),
            Field::Segments => soft.push(SoftRule::Segments {
                val: value.as_u32(),
                w: weight,
            }),
        }
    }
}

#[derive(Clone, Debug)]
struct IfStmt {
    cond: ExprNode,
    const_cond: Option<bool>,
    then_branch: Vec<Stmt>,
    else_branch: Vec<Stmt>,
}

#[derive(Clone, Debug)]
struct WhileStmt {
    cond: ExprNode,
    const_cond: Option<bool>,
    body: Vec<Stmt>,
}

impl WhileStmt {
    fn run(
        &self,
        program: &Program,
        ctx: &Ctx,
        locals: &mut [Value],
        hard: &mut Hard,
        soft: &mut Vec<SoftRule>,
    ) {
        if matches!(self.const_cond, Some(false)) {
            return;
        }

        debug_assert_ne!(
            self.const_cond,
            Some(true),
            "constant-true while loops are rejected at parse time"
        );

        let mut iterations = 0usize;
        loop {
            let cond = self
                .const_cond
                .unwrap_or_else(|| self.cond.evaluate(program, ctx, locals).as_bool());
            if !cond {
                break;
            }

            let signal = program.execute_block(&self.body, ctx, locals, hard, soft);
            iterations += 1;
            if iterations >= MAX_LOOP_ITERATIONS {
                debug_assert!(
                    iterations < MAX_LOOP_ITERATIONS,
                    "while loop iteration cap reached"
                );
                break;
            }

            match signal {
                FlowSignal::None => {}
                FlowSignal::Continue => continue,
                FlowSignal::Break => break,
            }
        }
    }
}

#[derive(Clone, Debug)]
struct ForStmt {
    id: usize,
    start: ExprNode,
    end: ExprNode,
    const_start: Option<Value>,
    const_end: Option<Value>,
    inclusive: bool,
    body: Vec<Stmt>,
}

impl ForStmt {
    fn run(
        &self,
        program: &Program,
        ctx: &Ctx,
        locals: &mut [Value],
        hard: &mut Hard,
        soft: &mut Vec<SoftRule>,
    ) {
        let start = self
            .const_start
            .unwrap_or_else(|| self.start.evaluate(program, ctx, locals))
            .as_i64();
        let end = self
            .const_end
            .unwrap_or_else(|| self.end.evaluate(program, ctx, locals))
            .as_i64();

        let mut current = start;
        let mut iterations = 0usize;
        let step = if start <= end { 1 } else { -1 };

        loop {
            let in_range = if step > 0 {
                if self.inclusive {
                    current <= end
                } else {
                    current < end
                }
            } else if self.inclusive {
                current >= end
            } else {
                current > end
            };

            if !in_range {
                break;
            }

            locals[self.id] = Value::from_f64(current as f64);
            let signal = program.execute_block(&self.body, ctx, locals, hard, soft);
            iterations += 1;
            if iterations >= MAX_LOOP_ITERATIONS {
                debug_assert!(
                    iterations < MAX_LOOP_ITERATIONS,
                    "for loop iteration cap reached"
                );
                break;
            }

            if matches!(signal, FlowSignal::Break) {
                break;
            }

            current += step;

            if matches!(signal, FlowSignal::Continue) {
                continue;
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Field {
    U2,
    Wg,
    Kl,
    Ch,
    Algo,
    Midk,
    Bottomk,
    Ctile,
    TileCols,
    Radix,
    Segments,
}

#[derive(Clone, Debug)]
struct Parser {
    tokens: Vec<Token>,
    index: usize,
    locals: Vec<String>,
    scopes: Vec<HashMap<String, usize>>,
    loop_depth: usize,
}

#[derive(Default)]
struct ParseOutput {
    stmts: Vec<Stmt>,
    locals: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            index: 0,
            locals: Vec::new(),
            scopes: vec![HashMap::new()],
            loop_depth: 0,
        }
    }

    fn parse_program(&mut self) -> Result<ParseOutput, Err> {
        let mut output = ParseOutput::default();
        while self.peek().is_some() {
            let stmt = self.parse_stmt()?;
            output.stmts.push(stmt);
            if matches!(self.peek(), Some(Token::Semi)) {
                self.next();
            }
        }
        output.locals = self.locals.len();
        Ok(output)
    }

    fn parse_stmt(&mut self) -> Result<Stmt, Err> {
        if self.consume_soft()? {
            let field = self.parse_field()?;
            self.expect(Token::Comma)?;
            let value_expr = self.parse_expr()?;
            self.expect(Token::Comma)?;
            let weight_expr = self.parse_expr()?;
            self.expect(Token::Comma)?;
            let cond_expr = self.parse_expr()?;
            self.expect(Token::Rp)?;

            let const_value = value_expr.const_value();
            let const_weight = weight_expr.const_value().map(|v| v.as_f64() as f32);
            let const_cond = cond_expr.const_value().map(|v| v.as_bool());

            return Ok(Stmt::Soft(SoftStmt {
                field,
                value_expr,
                weight_expr,
                cond_expr,
                const_value,
                const_weight,
                const_cond,
            }));
        }

        if self.consume_let()? {
            let name = match self.next().ok_or(Err::Tok)? {
                Token::Id(id) => id,
                _ => return Err(Err::Tok),
            };
            self.expect(Token::Assign)?;
            let expr = self.parse_expr()?;
            let const_value = expr.const_value();
            let id = self.define_local(name)?;
            return Ok(Stmt::Let(LetStmt {
                id,
                expr,
                const_value,
            }));
        }

        if self.consume_set()? {
            let name = match self.next().ok_or(Err::Tok)? {
                Token::Id(id) => id,
                _ => return Err(Err::Tok),
            };
            let id = self
                .lookup_local(&name)
                .ok_or_else(|| Err::Parse(self.index.saturating_sub(1)))?;
            self.expect(Token::Assign)?;
            let expr = self.parse_expr()?;
            let const_value = expr.const_value();
            return Ok(Stmt::Set(SetStmt {
                id,
                expr,
                const_value,
            }));
        }

        if self.consume_if()? {
            return self.parse_if_stmt();
        }

        if self.consume_while()? {
            return self.parse_while_stmt();
        }

        if self.consume_for()? {
            return self.parse_for_stmt();
        }

        if self.consume_break()? {
            return Ok(Stmt::Break);
        }

        if self.consume_continue()? {
            return Ok(Stmt::Continue);
        }

        let field = self.parse_field()?;
        self.expect(Token::Colon)?;
        let expr = self.parse_expr()?;
        let const_value = expr.const_value();
        Ok(Stmt::Assign(AssignStmt {
            field,
            expr,
            const_value,
        }))
    }

    fn parse_if_stmt(&mut self) -> Result<Stmt, Err> {
        let cond = self.parse_expr()?;
        let const_cond = cond.const_value().map(|v| v.as_bool());
        let then_branch = self.parse_block()?;

        let else_branch = if self.consume_else()? {
            if self.consume_if()? {
                match self.parse_if_stmt()? {
                    Stmt::If(branch) => vec![Stmt::If(branch)],
                    _ => unreachable!(),
                }
            } else {
                self.parse_block()?
            }
        } else {
            Vec::new()
        };

        Ok(Stmt::If(IfStmt {
            cond,
            const_cond,
            then_branch,
            else_branch,
        }))
    }

    fn parse_while_stmt(&mut self) -> Result<Stmt, Err> {
        let cond = self.parse_expr()?;
        let const_cond = cond.const_value().map(|v| v.as_bool());
        if matches!(const_cond, Some(true)) {
            return Err(Err::Parse(self.index.saturating_sub(1)));
        }
        self.loop_depth += 1;
        let body_result = self.parse_block();
        self.loop_depth -= 1;
        let body = body_result?;
        Ok(Stmt::While(WhileStmt {
            cond,
            const_cond,
            body,
        }))
    }

    fn parse_for_stmt(&mut self) -> Result<Stmt, Err> {
        let name = match self.next().ok_or(Err::Tok)? {
            Token::Id(id) => id,
            _ => return Err(Err::Tok),
        };
        self.expect(Token::In)?;
        let start = self.parse_expr()?;
        let const_start = start.const_value();
        let inclusive = match self.next().ok_or(Err::Tok)? {
            Token::Range => false,
            Token::RangeEq => true,
            _ => return Err(Err::Tok),
        };
        let end = self.parse_expr()?;
        let const_end = end.const_value();

        self.loop_depth += 1;
        let for_body = (|| {
            self.expect(Token::LBrace)?;
            self.push_scope();
            let id = match self.define_local(name) {
                Ok(id) => id,
                Err(err) => {
                    self.pop_scope();
                    return Err(err);
                }
            };
            let body_result = (|| {
                let mut stmts = Vec::new();
                while !matches!(self.peek(), Some(Token::RBrace)) {
                    let stmt = self.parse_stmt()?;
                    stmts.push(stmt);
                    if matches!(self.peek(), Some(Token::Semi)) {
                        self.next();
                    }
                }
                self.expect(Token::RBrace)?;
                Ok(stmts)
            })();
            self.pop_scope();
            body_result.map(|body| (id, body))
        })();
        self.loop_depth -= 1;
        let (id, body) = for_body?;

        Ok(Stmt::For(ForStmt {
            id,
            start,
            end,
            const_start,
            const_end,
            inclusive,
            body,
        }))
    }

    fn parse_block(&mut self) -> Result<Vec<Stmt>, Err> {
        self.expect(Token::LBrace)?;
        self.push_scope();
        let result = (|| {
            let mut stmts = Vec::new();
            while !matches!(self.peek(), Some(Token::RBrace)) {
                let stmt = self.parse_stmt()?;
                stmts.push(stmt);
                if matches!(self.peek(), Some(Token::Semi)) {
                    self.next();
                }
            }
            self.expect(Token::RBrace)?;
            Ok(stmts)
        })();
        self.pop_scope();
        result
    }

    fn parse_match_expr(&mut self) -> Result<ExprNode, Err> {
        let scrutinee = self.parse_expr()?;
        self.expect(Token::LBrace)?;
        let mut arms = Vec::new();
        let mut saw_wildcard = false;

        while !matches!(self.peek(), Some(Token::RBrace)) {
            if saw_wildcard {
                return Err(Err::Parse(self.index));
            }
            let arm = self.parse_match_arm()?;
            saw_wildcard |= arm.has_wildcard();
            arms.push(arm);
            if matches!(self.peek(), Some(Token::Comma)) {
                self.next();
                if matches!(self.peek(), Some(Token::RBrace)) {
                    continue;
                }
            }
        }

        self.expect(Token::RBrace)?;
        if !saw_wildcard {
            return Err(Err::Parse(self.index.saturating_sub(1)));
        }

        Ok(ExprNode::match_expr(scrutinee, arms))
    }

    fn parse_match_arm(&mut self) -> Result<MatchArm, Err> {
        let mut patterns = Vec::new();
        loop {
            patterns.push(self.parse_match_pattern()?);
            if matches!(self.peek(), Some(Token::Pipe)) {
                self.next();
                continue;
            }
            break;
        }

        let guard = if matches!(self.peek(), Some(Token::If)) {
            self.next();
            Some(self.parse_expr()?)
        } else {
            None
        };

        self.expect(Token::FatArrow)?;
        let expr = self.parse_expr()?;

        Ok(MatchArm {
            patterns,
            guard,
            expr,
        })
    }

    fn parse_match_pattern(&mut self) -> Result<MatchPattern, Err> {
        if let Some(Token::Id(name)) = self.peek() {
            if name == "_" {
                self.next();
                return Ok(MatchPattern::Wildcard);
            }
        }

        let start = self.parse_expr()?;
        if matches!(self.peek(), Some(Token::Range)) {
            self.next();
            let end = self.parse_expr()?;
            return Ok(MatchPattern::Range {
                start,
                end,
                inclusive: false,
            });
        }
        if matches!(self.peek(), Some(Token::RangeEq)) {
            self.next();
            let end = self.parse_expr()?;
            return Ok(MatchPattern::Range {
                start,
                end,
                inclusive: true,
            });
        }

        Ok(MatchPattern::Expr(start))
    }

    fn parse_field(&mut self) -> Result<Field, Err> {
        match self.next().ok_or(Err::Tok)? {
            Token::Id(id) if id == "u2" => Ok(Field::U2),
            Token::Id(id) if id == "wg" => Ok(Field::Wg),
            Token::Id(id) if id == "kl" => Ok(Field::Kl),
            Token::Id(id) if id == "ch" => Ok(Field::Ch),
            Token::Id(id) if id == "algo" => Ok(Field::Algo),
            Token::Id(id) if id == "midk" => Ok(Field::Midk),
            Token::Id(id) if id == "bottomk" => Ok(Field::Bottomk),
            Token::Id(id) if id == "ctile" => Ok(Field::Ctile),
            Token::Id(id) if id == "tile_cols" => Ok(Field::TileCols),
            Token::Id(id) if id == "radix" => Ok(Field::Radix),
            Token::Id(id) if id == "segments" => Ok(Field::Segments),
            _ => Err(Err::Tok),
        }
    }

    fn parse_expr(&mut self) -> Result<ExprNode, Err> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<ExprNode, Err> {
        let mut lhs = self.parse_and()?;
        while matches!(self.peek(), Some(Token::Op(Operator::Or))) {
            self.next();
            let rhs = self.parse_and()?;
            lhs = ExprNode::binary(BinaryOp::Or, lhs, rhs);
        }
        Ok(lhs)
    }

    fn parse_and(&mut self) -> Result<ExprNode, Err> {
        let mut lhs = self.parse_cmp()?;
        while matches!(self.peek(), Some(Token::Op(Operator::And))) {
            self.next();
            let rhs = self.parse_cmp()?;
            lhs = ExprNode::binary(BinaryOp::And, lhs, rhs);
        }
        Ok(lhs)
    }

    fn parse_cmp(&mut self) -> Result<ExprNode, Err> {
        let lhs = self.parse_add()?;
        if let Some(Token::Op(op)) = self.peek().cloned() {
            let cmp = matches!(
                op,
                Operator::Less
                    | Operator::LessEq
                    | Operator::Greater
                    | Operator::GreaterEq
                    | Operator::Eq
                    | Operator::NotEq
            );
            if cmp {
                self.next();
                let rhs = self.parse_add()?;
                return Ok(ExprNode::binary(BinaryOp::from(op), lhs, rhs));
            }
        }
        Ok(lhs)
    }

    fn parse_add(&mut self) -> Result<ExprNode, Err> {
        let mut lhs = self.parse_mul()?;
        loop {
            let op = match self.peek() {
                Some(Token::Op(Operator::Add)) => BinaryOp::Add,
                Some(Token::Op(Operator::Sub)) => BinaryOp::Sub,
                _ => break,
            };
            self.next();
            let rhs = self.parse_mul()?;
            lhs = ExprNode::binary(op, lhs, rhs);
        }
        Ok(lhs)
    }

    fn parse_mul(&mut self) -> Result<ExprNode, Err> {
        let mut lhs = self.parse_unary()?;
        loop {
            let op = match self.peek() {
                Some(Token::Op(Operator::Mul)) => BinaryOp::Mul,
                Some(Token::Op(Operator::Div)) => BinaryOp::Div,
                Some(Token::Op(Operator::Mod)) => BinaryOp::Mod,
                _ => break,
            };
            self.next();
            let rhs = self.parse_unary()?;
            lhs = ExprNode::binary(op, lhs, rhs);
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> Result<ExprNode, Err> {
        if matches!(self.peek(), Some(Token::Op(Operator::Sub))) {
            self.next();
            let expr = self.parse_unary()?;
            return Ok(ExprNode::neg(expr));
        }
        if matches!(self.peek(), Some(Token::Bang)) {
            self.next();
            let expr = self.parse_unary()?;
            return Ok(ExprNode::not(expr));
        }
        self.parse_atom()
    }

    fn parse_atom(&mut self) -> Result<ExprNode, Err> {
        let token = self.next().ok_or(Err::Tok)?;
        match token {
            Token::Num(n) => Ok(ExprNode::number(n)),
            Token::True => Ok(ExprNode::bool(true)),
            Token::False => Ok(ExprNode::bool(false)),
            Token::Id(id) if id == "r" => Ok(ExprNode::field(FieldRef::R)),
            Token::Id(id) if id == "c" => Ok(ExprNode::field(FieldRef::C)),
            Token::Id(id) if id == "k" => Ok(ExprNode::field(FieldRef::K)),
            Token::Id(id) if id == "sg" => Ok(ExprNode::field(FieldRef::Sg)),
            Token::Id(id) if id == "sgc" => Ok(ExprNode::field(FieldRef::Sgc)),
            Token::Id(id) if id == "kc" => Ok(ExprNode::field(FieldRef::Kc)),
            Token::Id(id) if id == "tile_cols" => Ok(ExprNode::field(FieldRef::TileCols)),
            Token::Id(id) if id == "radix" => Ok(ExprNode::field(FieldRef::Radix)),
            Token::Id(id) if id == "segments" => Ok(ExprNode::field(FieldRef::Segments)),
            Token::Id(id) if id == "log2" => self.parse_function_call(Function::Log2),
            Token::Id(id) if id == "sel" => self.parse_function_call(Function::Select),
            Token::Id(id) if id == "clamp" => self.parse_function_call(Function::Clamp),
            Token::Id(id) if id == "min" => self.parse_function_call(Function::Min),
            Token::Id(id) if id == "max" => self.parse_function_call(Function::Max),
            Token::Id(id) if id == "abs" => self.parse_function_call(Function::Abs),
            Token::Id(id) if id == "floor" => self.parse_function_call(Function::Floor),
            Token::Id(id) if id == "ceil" => self.parse_function_call(Function::Ceil),
            Token::Id(id) if id == "round" => self.parse_function_call(Function::Round),
            Token::Id(id) if id == "sqrt" => self.parse_function_call(Function::Sqrt),
            Token::Id(id) if id == "pow" => self.parse_function_call(Function::Pow),
            Token::Id(id) => {
                if let Some(index) = self.lookup_local(&id) {
                    Ok(ExprNode::local(index))
                } else {
                    Err(Err::Tok)
                }
            }
            Token::Match => self.parse_match_expr(),
            Token::Lp => {
                let expr = self.parse_expr()?;
                self.expect(Token::Rp)?;
                Ok(expr)
            }
            Token::LBrace => {
                let expr = self.parse_expr()?;
                self.expect(Token::RBrace)?;
                Ok(expr)
            }
            _ => Err(Err::Tok),
        }
    }

    fn parse_function_call(&mut self, function: Function) -> Result<ExprNode, Err> {
        self.expect(Token::Lp)?;
        let mut args = Vec::new();
        if !matches!(self.peek(), Some(Token::Rp)) {
            loop {
                args.push(self.parse_expr()?);
                if matches!(self.peek(), Some(Token::Comma)) {
                    self.next();
                    continue;
                }
                break;
            }
        }
        self.expect(Token::Rp)?;
        let (min, max) = function.arity();
        if args.len() < min || max.map(|limit| args.len() > limit).unwrap_or(false) {
            return Err(Err::Parse(self.index));
        }
        Ok(ExprNode::call(function, args))
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.index)
    }

    fn next(&mut self) -> Option<Token> {
        let token = self.tokens.get(self.index).cloned();
        if token.is_some() {
            self.index += 1;
        }
        token
    }

    fn expect(&mut self, want: Token) -> Result<(), Err> {
        let token = self.next().ok_or(Err::Tok)?;
        if token == want {
            Ok(())
        } else {
            Err(Err::Tok)
        }
    }

    fn consume_soft(&mut self) -> Result<bool, Err> {
        if matches!(self.peek(), Some(Token::Soft)) {
            self.next();
            self.expect(Token::Lp)?;
            return Ok(true);
        }
        Ok(false)
    }

    fn consume_let(&mut self) -> Result<bool, Err> {
        if matches!(self.peek(), Some(Token::Let)) {
            self.next();
            return Ok(true);
        }
        Ok(false)
    }

    fn consume_set(&mut self) -> Result<bool, Err> {
        if matches!(self.peek(), Some(Token::Set)) {
            self.next();
            return Ok(true);
        }
        Ok(false)
    }

    fn consume_if(&mut self) -> Result<bool, Err> {
        if matches!(self.peek(), Some(Token::If)) {
            self.next();
            return Ok(true);
        }
        Ok(false)
    }

    fn consume_while(&mut self) -> Result<bool, Err> {
        if matches!(self.peek(), Some(Token::While)) {
            self.next();
            return Ok(true);
        }
        Ok(false)
    }

    fn consume_for(&mut self) -> Result<bool, Err> {
        if matches!(self.peek(), Some(Token::For)) {
            self.next();
            return Ok(true);
        }
        Ok(false)
    }

    fn consume_else(&mut self) -> Result<bool, Err> {
        if matches!(self.peek(), Some(Token::Else)) {
            self.next();
            return Ok(true);
        }
        Ok(false)
    }

    fn consume_break(&mut self) -> Result<bool, Err> {
        if matches!(self.peek(), Some(Token::Break)) {
            if self.loop_depth == 0 {
                return Err(Err::Parse(self.index));
            }
            self.next();
            return Ok(true);
        }
        Ok(false)
    }

    fn consume_continue(&mut self) -> Result<bool, Err> {
        if matches!(self.peek(), Some(Token::Continue)) {
            if self.loop_depth == 0 {
                return Err(Err::Parse(self.index));
            }
            self.next();
            return Ok(true);
        }
        Ok(false)
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop().expect("scope underflow");
    }

    fn define_local(&mut self, name: String) -> Result<usize, Err> {
        if let Some(scope) = self.scopes.last_mut() {
            if scope.contains_key(&name) {
                return Err(Err::Parse(self.index));
            }
            let id = self.locals.len();
            self.locals.push(name.clone());
            scope.insert(name, id);
            Ok(id)
        } else {
            Err(Err::Parse(self.index))
        }
    }

    fn lookup_local(&self, name: &str) -> Option<usize> {
        for scope in self.scopes.iter().rev() {
            if let Some(id) = scope.get(name) {
                return Some(*id);
            }
        }
        None
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Token {
    Id(String),
    Num(f64),
    True,
    False,
    Lp,
    Rp,
    LBrace,
    RBrace,
    Comma,
    Semi,
    Colon,
    Assign,
    Bang,
    Soft,
    Let,
    Set,
    If,
    Else,
    While,
    For,
    In,
    Match,
    Range,
    RangeEq,
    Break,
    Continue,
    FatArrow,
    Pipe,
    Op(Operator),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Operator {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Less,
    LessEq,
    Greater,
    GreaterEq,
    Eq,
    NotEq,
    And,
    Or,
}

impl Operator {
    fn from_bytes(first: u8, second: Option<u8>) -> Option<(Self, usize)> {
        use Operator::*;
        match (first, second) {
            (b'+', _) => Some((Add, 1)),
            (b'-', _) => Some((Sub, 1)),
            (b'*', _) => Some((Mul, 1)),
            (b'/', _) => Some((Div, 1)),
            (b'%', _) => Some((Mod, 1)),
            (b'<', Some(b'=')) => Some((LessEq, 2)),
            (b'<', _) => Some((Less, 1)),
            (b'>', Some(b'=')) => Some((GreaterEq, 2)),
            (b'>', _) => Some((Greater, 1)),
            (b'=', Some(b'=')) => Some((Eq, 2)),
            (b'!', Some(b'=')) => Some((NotEq, 2)),
            (b'&', Some(b'&')) => Some((And, 2)),
            (b'|', Some(b'|')) => Some((Or, 2)),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Less,
    LessEq,
    Greater,
    GreaterEq,
    Eq,
    NotEq,
    And,
    Or,
}

impl From<Operator> for BinaryOp {
    fn from(op: Operator) -> Self {
        match op {
            Operator::Add => BinaryOp::Add,
            Operator::Sub => BinaryOp::Sub,
            Operator::Mul => BinaryOp::Mul,
            Operator::Div => BinaryOp::Div,
            Operator::Mod => BinaryOp::Mod,
            Operator::Less => BinaryOp::Less,
            Operator::LessEq => BinaryOp::LessEq,
            Operator::Greater => BinaryOp::Greater,
            Operator::GreaterEq => BinaryOp::GreaterEq,
            Operator::Eq => BinaryOp::Eq,
            Operator::NotEq => BinaryOp::NotEq,
            Operator::And => BinaryOp::And,
            Operator::Or => BinaryOp::Or,
        }
    }
}

impl BinaryOp {
    fn apply(self, lhs: Value, rhs: Value) -> Value {
        use BinaryOp::*;
        match self {
            Add => Value::from_f64(lhs.as_f64() + rhs.as_f64()),
            Sub => Value::from_f64(lhs.as_f64() - rhs.as_f64()),
            Mul => Value::from_f64(lhs.as_f64() * rhs.as_f64()),
            Div => Value::from_f64(lhs.as_f64() / rhs.as_f64()),
            Mod => Value::from_f64(lhs.as_f64() % rhs.as_f64()),
            Less => Value::from_bool(lhs.as_f64() < rhs.as_f64()),
            LessEq => Value::from_bool(lhs.as_f64() <= rhs.as_f64()),
            Greater => Value::from_bool(lhs.as_f64() > rhs.as_f64()),
            GreaterEq => Value::from_bool(lhs.as_f64() >= rhs.as_f64()),
            Eq => Value::from_bool(lhs.as_f64() == rhs.as_f64()),
            NotEq => Value::from_bool(lhs.as_f64() != rhs.as_f64()),
            And => Value::from_bool(lhs.as_bool() && rhs.as_bool()),
            Or => Value::from_bool(lhs.as_bool() || rhs.as_bool()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum FieldRef {
    R,
    C,
    K,
    Sg,
    Sgc,
    Kc,
    TileCols,
    Radix,
    Segments,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Function {
    Log2,
    Select,
    Clamp,
    Min,
    Max,
    Abs,
    Floor,
    Ceil,
    Round,
    Sqrt,
    Pow,
}

#[derive(Clone, Debug)]
enum ExprNode {
    Number(f64),
    Bool(bool),
    Field(FieldRef),
    Local(usize),
    UnaryNeg(Box<ExprNode>),
    UnaryNot(Box<ExprNode>),
    Binary {
        op: BinaryOp,
        lhs: Box<ExprNode>,
        rhs: Box<ExprNode>,
    },
    Call(Function, Vec<ExprNode>),
    Match {
        scrutinee: Box<ExprNode>,
        arms: Vec<MatchArm>,
    },
}

impl ExprNode {
    fn number(n: f64) -> Self {
        ExprNode::Number(n)
    }

    fn bool(b: bool) -> Self {
        ExprNode::Bool(b)
    }

    fn field(field: FieldRef) -> Self {
        ExprNode::Field(field)
    }

    fn local(id: usize) -> Self {
        ExprNode::Local(id)
    }

    fn neg(expr: ExprNode) -> Self {
        ExprNode::UnaryNeg(Box::new(expr)).fold()
    }

    fn not(expr: ExprNode) -> Self {
        ExprNode::UnaryNot(Box::new(expr)).fold()
    }

    fn binary(op: BinaryOp, lhs: ExprNode, rhs: ExprNode) -> Self {
        ExprNode::Binary {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
        .fold()
    }

    fn call(function: Function, args: Vec<ExprNode>) -> Self {
        ExprNode::Call(function, args).fold()
    }

    fn match_expr(scrutinee: ExprNode, arms: Vec<MatchArm>) -> Self {
        ExprNode::Match {
            scrutinee: Box::new(scrutinee),
            arms,
        }
        .fold()
    }

    fn fold(self) -> Self {
        use ExprNode::*;
        match self {
            Number(_) | Bool(_) | Field(_) | Local(_) => self,
            UnaryNeg(expr) => {
                let expr = expr.fold();
                let node = UnaryNeg(Box::new(expr));
                node.const_value().map_or(node, ExprNode::from_value)
            }
            UnaryNot(expr) => {
                let expr = expr.fold();
                let node = UnaryNot(Box::new(expr));
                node.const_value().map_or(node, ExprNode::from_value)
            }
            Binary { op, lhs, rhs } => {
                let lhs = lhs.fold();
                let rhs = rhs.fold();
                if let (Some(l), Some(r)) = (lhs.const_value(), rhs.const_value()) {
                    return ExprNode::from_value(op.apply(l, r));
                }
                Binary {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                }
            }
            Call(function, args) => {
                let mut folded_args = Vec::with_capacity(args.len());
                for arg in args {
                    folded_args.push(arg.fold());
                }

                if function == Function::Select {
                    if let Some(cond) = folded_args[0].const_value() {
                        return if cond.as_bool() {
                            folded_args[1].clone()
                        } else {
                            folded_args[2].clone()
                        };
                    }
                }

                if let Some(values) = folded_args
                    .iter()
                    .map(ExprNode::const_value)
                    .collect::<Option<Vec<_>>>()
                {
                    return ExprNode::from_value(function.eval(&values));
                }

                Call(function, folded_args)
            }
            Match { scrutinee, arms } => {
                let scrutinee = scrutinee.fold();
                let folded_arms = arms.into_iter().map(MatchArm::fold).collect();
                let node = ExprNode::Match {
                    scrutinee: Box::new(scrutinee),
                    arms: folded_arms,
                };
                node.const_value().map_or(node, ExprNode::from_value)
            }
        }
    }

    fn const_value(&self) -> Option<Value> {
        use ExprNode::*;
        match self {
            Number(n) => Some(Value::from_f64(*n)),
            Bool(b) => Some(Value::from_bool(*b)),
            Field(_) | Local(_) => None,
            UnaryNeg(expr) => expr.const_value().map(|v| Value::from_f64(-v.as_f64())),
            UnaryNot(expr) => expr.const_value().map(|v| Value::from_bool(!v.as_bool())),
            Binary { op, lhs, rhs } => {
                let l = lhs.const_value()?;
                let r = rhs.const_value()?;
                Some(op.apply(l, r))
            }
            Call(function, args) => {
                if function == &Function::Select {
                    let cond = args[0].const_value()?;
                    if cond.as_bool() {
                        return args[1].const_value();
                    } else {
                        return args[2].const_value();
                    }
                }

                let values = args
                    .iter()
                    .map(ExprNode::const_value)
                    .collect::<Option<Vec<_>>>()?;
                Some(function.eval(&values))
            }
            Match { scrutinee, arms } => {
                let value = scrutinee.const_value()?;
                for arm in arms {
                    if let Some(result) = arm.const_result(value) {
                        return Some(result);
                    }
                }
                None
            }
        }
    }

    fn evaluate(&self, program: &Program, ctx: &Ctx, locals: &[Value]) -> Value {
        use ExprNode::*;
        match self {
            Number(n) => Value::from_f64(*n),
            Bool(b) => Value::from_bool(*b),
            Field(field) => field.read(ctx),
            Local(id) => locals[*id],
            UnaryNeg(expr) => Value::from_f64(-expr.evaluate(program, ctx, locals).as_f64()),
            UnaryNot(expr) => Value::from_bool(!expr.evaluate(program, ctx, locals).as_bool()),
            Binary { op, lhs, rhs } => match op {
                BinaryOp::And => {
                    let left = lhs.evaluate(program, ctx, locals);
                    if !left.as_bool() {
                        Value::from_bool(false)
                    } else {
                        Value::from_bool(rhs.evaluate(program, ctx, locals).as_bool())
                    }
                }
                BinaryOp::Or => {
                    let left = lhs.evaluate(program, ctx, locals);
                    if left.as_bool() {
                        Value::from_bool(true)
                    } else {
                        Value::from_bool(rhs.evaluate(program, ctx, locals).as_bool())
                    }
                }
                _ => {
                    let l = lhs.evaluate(program, ctx, locals);
                    let r = rhs.evaluate(program, ctx, locals);
                    op.apply(l, r)
                }
            },
            Call(function, args) => function.eval_runtime(program, ctx, locals, args),
            Match { scrutinee, arms } => {
                let value = scrutinee.evaluate(program, ctx, locals);
                for arm in arms {
                    if arm.matches(program, ctx, locals, value) {
                        return arm.evaluate(program, ctx, locals);
                    }
                }
                panic!("non-exhaustive match expression");
            }
        }
    }

    fn from_value(value: Value) -> Self {
        match value {
            Value::F(v) => ExprNode::Number(v),
            Value::B(b) => ExprNode::Bool(b),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Value {
    F(f64),
    B(bool),
}

impl Value {
    fn from_f64(v: f64) -> Self {
        Value::F(v)
    }

    fn from_bool(v: bool) -> Self {
        Value::B(v)
    }

    fn as_f64(self) -> f64 {
        match self {
            Value::F(v) => v,
            Value::B(true) => 1.0,
            Value::B(false) => 0.0,
        }
    }

    fn as_bool(self) -> bool {
        match self {
            Value::B(v) => v,
            Value::F(v) => v != 0.0,
        }
    }

    fn as_i64(self) -> i64 {
        self.as_f64().round() as i64
    }

    fn as_u32(self) -> u32 {
        self.as_f64().round() as u32
    }

    fn as_u8(self) -> u8 {
        self.as_f64().round() as u8
    }

    fn equals(self, other: Value) -> bool {
        match (self, other) {
            (Value::B(lhs), Value::B(rhs)) => lhs == rhs,
            (Value::F(lhs), Value::F(rhs)) => lhs == rhs,
            (Value::F(lhs), Value::B(rhs)) => lhs == if rhs { 1.0 } else { 0.0 },
            (Value::B(lhs), Value::F(rhs)) => (if lhs { 1.0 } else { 0.0 }) == rhs,
        }
    }
}

impl Default for Value {
    fn default() -> Self {
        Value::F(0.0)
    }
}

impl FieldRef {
    fn read(self, ctx: &Ctx) -> Value {
        match self {
            FieldRef::R => Value::from_f64(ctx.r as f64),
            FieldRef::C => Value::from_f64(ctx.c as f64),
            FieldRef::K => Value::from_f64(ctx.k as f64),
            FieldRef::Sg => Value::from_bool(ctx.sg),
            FieldRef::Sgc => Value::from_f64(ctx.sgc as f64),
            FieldRef::Kc => Value::from_f64(ctx.kc as f64),
            FieldRef::TileCols => Value::from_f64(ctx.tile_cols as f64),
            FieldRef::Radix => Value::from_f64(ctx.radix as f64),
            FieldRef::Segments => Value::from_f64(ctx.segments as f64),
        }
    }
}

impl Function {
    fn arity(self) -> (usize, Option<usize>) {
        match self {
            Function::Log2
            | Function::Abs
            | Function::Floor
            | Function::Ceil
            | Function::Round
            | Function::Sqrt => (1, Some(1)),
            Function::Pow => (2, Some(2)),
            Function::Select | Function::Clamp => (3, Some(3)),
            Function::Min | Function::Max => (2, None),
        }
    }

    fn eval(&self, values: &[Value]) -> Value {
        match self {
            Function::Log2 => Value::from_f64(values[0].as_f64().log2()),
            Function::Select => {
                if values[0].as_bool() {
                    values[1]
                } else {
                    values[2]
                }
            }
            Function::Clamp => {
                let v = values[0].as_f64();
                let lo = values[1].as_f64();
                let hi = values[2].as_f64();
                Value::from_f64(v.max(lo).min(hi))
            }
            Function::Min => {
                let mut iter = values.iter();
                let first = iter.next().copied().unwrap();
                let mut best = first.as_f64();
                for v in iter {
                    best = best.min(v.as_f64());
                }
                Value::from_f64(best)
            }
            Function::Max => {
                let mut iter = values.iter();
                let first = iter.next().copied().unwrap();
                let mut best = first.as_f64();
                for v in iter {
                    best = best.max(v.as_f64());
                }
                Value::from_f64(best)
            }
            Function::Abs => Value::from_f64(values[0].as_f64().abs()),
            Function::Floor => Value::from_f64(values[0].as_f64().floor()),
            Function::Ceil => Value::from_f64(values[0].as_f64().ceil()),
            Function::Round => Value::from_f64(values[0].as_f64().round()),
            Function::Sqrt => Value::from_f64(values[0].as_f64().sqrt()),
            Function::Pow => Value::from_f64(values[0].as_f64().powf(values[1].as_f64())),
        }
    }

    fn eval_runtime(
        &self,
        program: &Program,
        ctx: &Ctx,
        locals: &[Value],
        args: &[ExprNode],
    ) -> Value {
        match self {
            Function::Select => {
                let cond = args[0].evaluate(program, ctx, locals);
                if cond.as_bool() {
                    args[1].evaluate(program, ctx, locals)
                } else {
                    args[2].evaluate(program, ctx, locals)
                }
            }
            Function::Clamp => {
                let v = args[0].evaluate(program, ctx, locals).as_f64();
                let lo = args[1].evaluate(program, ctx, locals).as_f64();
                let hi = args[2].evaluate(program, ctx, locals).as_f64();
                Value::from_f64(v.max(lo).min(hi))
            }
            Function::Min => {
                let mut iter = args.iter();
                let mut best = iter
                    .next()
                    .map(|expr| expr.evaluate(program, ctx, locals).as_f64())
                    .unwrap();
                for expr in iter {
                    best = best.min(expr.evaluate(program, ctx, locals).as_f64());
                }
                Value::from_f64(best)
            }
            Function::Max => {
                let mut iter = args.iter();
                let mut best = iter
                    .next()
                    .map(|expr| expr.evaluate(program, ctx, locals).as_f64())
                    .unwrap();
                for expr in iter {
                    best = best.max(expr.evaluate(program, ctx, locals).as_f64());
                }
                Value::from_f64(best)
            }
            Function::Log2
            | Function::Abs
            | Function::Floor
            | Function::Ceil
            | Function::Round
            | Function::Sqrt => {
                let value = args[0].evaluate(program, ctx, locals).as_f64();
                match self {
                    Function::Log2 => Value::from_f64(value.log2()),
                    Function::Abs => Value::from_f64(value.abs()),
                    Function::Floor => Value::from_f64(value.floor()),
                    Function::Ceil => Value::from_f64(value.ceil()),
                    Function::Round => Value::from_f64(value.round()),
                    Function::Sqrt => Value::from_f64(value.sqrt()),
                    _ => unreachable!(),
                }
            }
            Function::Pow => {
                let base = args[0].evaluate(program, ctx, locals).as_f64();
                let exp = args[1].evaluate(program, ctx, locals).as_f64();
                Value::from_f64(base.powf(exp))
            }
        }
    }
}

fn lex(src: &str) -> Result<Vec<Token>, Err> {
    let bytes = src.as_bytes();
    let mut tokens = Vec::new();
    let mut index = 0usize;

    while index < bytes.len() {
        let b = bytes[index];
        let ch = b as char;
        if ch.is_whitespace() {
            index += 1;
            continue;
        }

        match b {
            b'(' => {
                tokens.push(Token::Lp);
                index += 1;
            }
            b')' => {
                tokens.push(Token::Rp);
                index += 1;
            }
            b'{' => {
                tokens.push(Token::LBrace);
                index += 1;
            }
            b'}' => {
                tokens.push(Token::RBrace);
                index += 1;
            }
            b',' => {
                tokens.push(Token::Comma);
                index += 1;
            }
            b';' => {
                tokens.push(Token::Semi);
                index += 1;
            }
            b':' => {
                tokens.push(Token::Colon);
                index += 1;
            }
            b'=' => {
                if index + 1 < bytes.len() && bytes[index + 1] == b'>' {
                    tokens.push(Token::FatArrow);
                    index += 2;
                } else if index + 1 < bytes.len() && bytes[index + 1] == b'=' {
                    tokens.push(Token::Op(Operator::Eq));
                    index += 2;
                } else {
                    tokens.push(Token::Assign);
                    index += 1;
                }
            }
            b'!' => {
                if index + 1 < bytes.len() && bytes[index + 1] == b'=' {
                    tokens.push(Token::Op(Operator::NotEq));
                    index += 2;
                } else {
                    tokens.push(Token::Bang);
                    index += 1;
                }
            }
            b'|' => {
                if index + 1 < bytes.len() && bytes[index + 1] == b'|' {
                    tokens.push(Token::Op(Operator::Or));
                    index += 2;
                } else {
                    tokens.push(Token::Pipe);
                    index += 1;
                }
            }
            b'.' => {
                if index + 1 < bytes.len() && bytes[index + 1] == b'.' {
                    if index + 2 < bytes.len() && bytes[index + 2] == b'=' {
                        tokens.push(Token::RangeEq);
                        index += 3;
                    } else {
                        tokens.push(Token::Range);
                        index += 2;
                    }
                    continue;
                }

                let start = index;
                index += 1;
                if index >= bytes.len() || !bytes[index].is_ascii_digit() {
                    return Err(Err::Parse(start));
                }
                while index < bytes.len() && bytes[index].is_ascii_digit() {
                    index += 1;
                }
                let token = std::str::from_utf8(&bytes[start..index])
                    .map_err(|_| Err::Parse(start))?
                    .parse::<f64>()
                    .map_err(|_| Err::Parse(start))?;
                tokens.push(Token::Num(token));
            }
            b'0'..=b'9' => {
                let start = index;
                index += 1;
                while index < bytes.len() && bytes[index].is_ascii_digit() {
                    index += 1;
                }
                if index < bytes.len() && bytes[index] == b'.' {
                    if index + 1 < bytes.len() && bytes[index + 1].is_ascii_digit() {
                        index += 1;
                        while index < bytes.len() && bytes[index].is_ascii_digit() {
                            index += 1;
                        }
                    } else if index + 1 < bytes.len() && bytes[index + 1] == b'.' {
                        // leave the dot for the range operator
                    } else {
                        index += 1;
                    }
                }
                let token = std::str::from_utf8(&bytes[start..index])
                    .map_err(|_| Err::Parse(start))?
                    .parse::<f64>()
                    .map_err(|_| Err::Parse(start))?;
                tokens.push(Token::Num(token));
            }
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => {
                let start = index;
                index += 1;
                while index < bytes.len() {
                    let c = bytes[index] as char;
                    if c.is_ascii_alphanumeric() || c == '_' {
                        index += 1;
                    } else {
                        break;
                    }
                }
                let ident = std::str::from_utf8(&bytes[start..index])
                    .map_err(|_| Err::Parse(start))?
                    .to_string();
                match ident.as_str() {
                    "true" => tokens.push(Token::True),
                    "false" => tokens.push(Token::False),
                    "soft" => tokens.push(Token::Soft),
                    "let" => tokens.push(Token::Let),
                    "set" => tokens.push(Token::Set),
                    "if" => tokens.push(Token::If),
                    "else" => tokens.push(Token::Else),
                    "while" => tokens.push(Token::While),
                    "for" => tokens.push(Token::For),
                    "in" => tokens.push(Token::In),
                    "match" => tokens.push(Token::Match),
                    "break" => tokens.push(Token::Break),
                    "continue" => tokens.push(Token::Continue),
                    _ => tokens.push(Token::Id(ident)),
                }
            }
            _ => {
                let next = if index + 1 < bytes.len() {
                    Some(bytes[index + 1])
                } else {
                    None
                };
                if let Some((op, consumed)) = Operator::from_bytes(b, next) {
                    tokens.push(Token::Op(op));
                    index += consumed;
                } else {
                    return Err(Err::Parse(index));
                }
            }
        }
    }

    Ok(tokens)
}

impl fmt::Display for Operator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Operator::*;
        let s = match self {
            Add => "+",
            Sub => "-",
            Mul => "*",
            Div => "/",
            Mod => "%",
            Less => "<",
            LessEq => "<=",
            Greater => ">",
            GreaterEq => ">=",
            Eq => "==",
            NotEq => "!=",
            And => "&&",
            Or => "||",
        };
        write!(f, "{s}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> Ctx {
        Ctx {
            r: 1024,
            c: 16384,
            k: 512,
            sg: true,
            sgc: 256,
            kc: 1024,
            tile_cols: 128,
            radix: 4,
            segments: 2,
        }
    }

    #[test]
    fn parses_and_evaluates_assignments() {
        let program = Program::parse("wg: 256; radix: log2(k);").unwrap();
        let out = program.evaluate(&ctx());
        assert_eq!(out.hard.wg, Some(256));
        assert_eq!(out.hard.radix, Some(9));
    }

    #[test]
    fn folds_constants() {
        let program = Program::parse("radix: clamp(16, 2, 8);").unwrap();
        let out = program.evaluate(&ctx());
        assert_eq!(out.hard.radix, Some(8));
    }

    #[test]
    fn for_loop_accumulates_exclusive() {
        let program = Program::parse(
            "
            let total = 0;
            for i in 0..4 {
                set total = total + i;
            }
            radix: total;
            ",
        )
        .unwrap();
        let out = program.evaluate(&ctx());
        assert_eq!(out.hard.radix, Some(6));
    }

    #[test]
    fn for_loop_inclusive_and_descending() {
        let program = Program::parse(
            "
            let up = 0;
            for i in 2..=4 {
                set up = up + i;
            }
            let down = 0;
            for j in 4..0 {
                set down = down + j;
            }
            radix: up + down;
            ",
        )
        .unwrap();
        let out = program.evaluate(&ctx());
        assert_eq!(out.hard.radix, Some(19));
    }

    #[test]
    fn for_loop_break_and_continue() {
        let program = Program::parse(
            "
            let total = 0;
            for i in 0..10 {
                if i == 6 {
                    break;
                }
                if i % 2 == 0 {
                    continue;
                }
                set total = total + i;
            }
            radix: total;
            ",
        )
        .unwrap();
        let out = program.evaluate(&ctx());
        assert_eq!(out.hard.radix, Some(9));
    }

    #[test]
    fn evaluates_soft_rules() {
        let script = "soft (wg, r / 4, 0.25, true);";
        let program = Program::parse(script).unwrap();
        let out = program.evaluate(&ctx());
        assert_eq!(out.soft.len(), 1);
    }

    #[test]
    fn supports_locals_and_modulo() {
        let script = "let base = r / 4; wg: base; soft (wg, base, 0.5, base % 2 == 0);";
        let program = Program::parse(script).unwrap();
        let out = program.evaluate(&ctx());
        assert_eq!(out.hard.wg, Some(256));
        assert_eq!(out.soft.len(), 1);
    }

    #[test]
    fn advanced_math_functions_fold() {
        let script = "radix: max(min(pow(2, 3), sqrt(64)), abs(-4));";
        let program = Program::parse(script).unwrap();
        let out = program.evaluate(&ctx());
        assert_eq!(out.hard.radix, Some(8));
    }

    #[test]
    fn conditionals_and_unary_not_work() {
        let script = r#"
            let base = r / 4;
            if base % 2 == 0 {
                wg: base;
            } else {
                wg: 1;
            }

            if !sg {
                soft (wg, base, 0.25, true);
            } else {
                soft (wg, base * 2, 0.5, true);
            }
        "#;

        let program = Program::parse(script).unwrap();

        let out = program.evaluate(&ctx());
        assert_eq!(out.hard.wg, Some(256));
        assert_eq!(out.soft.len(), 1);
        match &out.soft[0] {
            SoftRule::Wg { val, w } => {
                assert_eq!(*val, 512);
                assert!((*w - 0.5).abs() < f32::EPSILON);
            }
            _ => panic!("expected wg soft rule"),
        }

        let mut alt_ctx = ctx();
        alt_ctx.sg = false;
        let out_alt = program.evaluate(&alt_ctx);
        assert_eq!(out_alt.hard.wg, Some(256));
        assert_eq!(out_alt.soft.len(), 1);
        match &out_alt.soft[0] {
            SoftRule::Wg { val, w } => {
                assert_eq!(*val, 256);
                assert!((*w - 0.25).abs() < f32::EPSILON);
            }
            _ => panic!("expected wg soft rule"),
        }
    }

    #[test]
    fn block_scopes_shadow_cleanly() {
        let script = r#"
            let base = 4;
            if true {
                let base = 8;
                soft (wg, base, 1.0, true);
            }
            soft (wg, base, 1.0, true);
        "#;

        let program = Program::parse(script).unwrap();
        let out = program.evaluate(&ctx());
        assert_eq!(out.soft.len(), 2);
        match &out.soft[0] {
            SoftRule::Wg { val, w } => {
                assert_eq!(*val, 8);
                assert!((*w - 1.0).abs() < f32::EPSILON);
            }
            _ => panic!("expected wg soft rule"),
        }
        match &out.soft[1] {
            SoftRule::Wg { val, w } => {
                assert_eq!(*val, 4);
                assert!((*w - 1.0).abs() < f32::EPSILON);
            }
            _ => panic!("expected wg soft rule"),
        }
    }

    #[test]
    fn while_loops_and_set_statements_work() {
        let script = r#"
            let acc = 1;
            let i = 0;
            while i < 3 {
                set acc = acc * 2;
                set i = i + 1;
            }
            wg: acc;
        "#;

        let program = Program::parse(script).unwrap();
        let out = program.evaluate(&ctx());
        assert_eq!(out.hard.wg, Some(8));
    }

    #[test]
    fn break_statements_exit_loops() {
        let script = r#"
            let acc = 0;
            let i = 0;
            while i < 10 {
                if i == 5 {
                    break;
                }
                set acc = acc + i;
                set i = i + 1;
            }
            wg: acc;
        "#;

        let program = Program::parse(script).unwrap();
        let out = program.evaluate(&ctx());
        assert_eq!(out.hard.wg, Some(10));
    }

    #[test]
    fn continue_skips_to_next_iteration() {
        let script = r#"
            let acc = 0;
            let i = 0;
            while i < 5 {
                set i = i + 1;
                if i % 2 == 0 {
                    continue;
                }
                set acc = acc + i;
            }
            wg: acc;
        "#;

        let program = Program::parse(script).unwrap();
        let out = program.evaluate(&ctx());
        assert_eq!(out.hard.wg, Some(9));
    }

    #[test]
    fn rejects_constant_true_while_loops() {
        let script = "while true { wg: 1; }";
        assert!(Program::parse(script).is_err());
    }

    #[test]
    fn rejects_break_and_continue_outside_loops() {
        assert!(Program::parse("break;").is_err());
        assert!(Program::parse("continue;").is_err());
    }

    #[test]
    fn match_expressions_drive_assignments() {
        let script = r#"
            let size = match r {
                0..512 => 1,
                512..=1024 => 2,
                _ => 4,
            };
            wg: size;
        "#;

        let program = Program::parse(script).unwrap();
        let out = program.evaluate(&ctx());
        assert_eq!(out.hard.wg, Some(2));

        let mut bigger = ctx();
        bigger.r = 1536;
        let out_big = program.evaluate(&bigger);
        assert_eq!(out_big.hard.wg, Some(4));
    }

    #[test]
    fn match_guards_and_pipes_work() {
        let script = r#"
            let tuned = match sg {
                true if r > 2048 => 9,
                true | false if k > 256 => 7,
                _ => 1,
            };
            wg: tuned;
        "#;

        let program = Program::parse(script).unwrap();
        let mut rich = ctx();
        rich.sg = true;
        rich.r = 4096;
        let out_rich = program.evaluate(&rich);
        assert_eq!(out_rich.hard.wg, Some(9));

        let mut mid = ctx();
        mid.sg = false;
        mid.k = 512;
        let out_mid = program.evaluate(&mid);
        assert_eq!(out_mid.hard.wg, Some(7));

        let mut poor = ctx();
        poor.sg = false;
        poor.k = 64;
        let out_poor = program.evaluate(&poor);
        assert_eq!(out_poor.hard.wg, Some(1));
    }

    #[test]
    fn match_requires_wildcard_arm() {
        let script = r#"
            let size = match r {
                0..512 => 1,
                512..=1024 => 2,
            };
            wg: size;
        "#;

        assert!(Program::parse(script).is_err());
    }
}
