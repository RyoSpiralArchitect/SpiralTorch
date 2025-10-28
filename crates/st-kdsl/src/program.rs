// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::fmt;

use crate::{Ctx, Err, Hard, Out, SoftRule};

/// Compiled representation of a SpiralK DSL program.
#[derive(Clone, Debug)]
pub struct Program {
    stmts: Vec<Stmt>,
}

impl Program {
    /// Parses the provided source string into a [`Program`].
    pub fn parse(src: &str) -> Result<Self, Err> {
        let toks = lex(src)?;
        let mut parser = Parser {
            tokens: toks,
            index: 0,
        };
        let stmts = parser.parse_program()?;
        Ok(Self { stmts })
    }

    /// Evaluates the program for the given [`Ctx`].
    pub fn evaluate(&self, ctx: &Ctx) -> Out {
        let mut hard = Hard::default();
        let mut soft = Vec::<SoftRule>::new();

        for stmt in &self.stmts {
            match stmt {
                Stmt::Assign(field, expr) => {
                    let value = expr.evaluate(ctx);
                    match field {
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
                Stmt::Soft(field, value_expr, weight_expr, cond_expr) => {
                    if cond_expr.evaluate(ctx).as_bool() {
                        let value = value_expr.evaluate(ctx);
                        let weight = weight_expr.evaluate(ctx).as_f64() as f32;
                        match field {
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
            }
        }

        Out { hard, soft }
    }
}

#[derive(Clone, Debug)]
enum Stmt {
    Assign(Field, ExprNode),
    Soft(Field, ExprNode, ExprNode, ExprNode),
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
}

impl Parser {
    fn parse_program(&mut self) -> Result<Vec<Stmt>, Err> {
        let mut stmts = Vec::new();
        while self.peek().is_some() {
            stmts.push(self.parse_stmt()?);
            if matches!(self.peek(), Some(Token::Semi)) {
                self.next();
            }
        }
        Ok(stmts)
    }

    fn parse_stmt(&mut self) -> Result<Stmt, Err> {
        if self.consume_soft()? {
            let field = self.parse_field()?;
            self.expect(Token::Comma)?;
            let value_expr = self.parse_expr()?;
            self.expect(Token::Comma)?;
            let weight_expr = self.parse_expr()?;
            self.expect(Token::Comma)?;
            let condition_expr = self.parse_expr()?;
            self.expect(Token::Rp)?;
            return Ok(Stmt::Soft(field, value_expr, weight_expr, condition_expr));
        }

        let field = self.parse_field()?;
        self.expect(Token::Colon)?;
        let expr = self.parse_expr()?;
        Ok(Stmt::Assign(field, expr))
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
            Token::Id(id) if id == "log2" => {
                self.expect(Token::Lp)?;
                let expr = self.parse_expr()?;
                self.expect(Token::Rp)?;
                Ok(ExprNode::log2(expr))
            }
            Token::Id(id) if id == "sel" => {
                self.expect(Token::Lp)?;
                let cond = self.parse_expr()?;
                self.expect(Token::Comma)?;
                let when_true = self.parse_expr()?;
                self.expect(Token::Comma)?;
                let when_false = self.parse_expr()?;
                self.expect(Token::Rp)?;
                Ok(ExprNode::select(cond, when_true, when_false))
            }
            Token::Id(id) if id == "clamp" => {
                self.expect(Token::Lp)?;
                let value = self.parse_expr()?;
                self.expect(Token::Comma)?;
                let lo = self.parse_expr()?;
                self.expect(Token::Comma)?;
                let hi = self.parse_expr()?;
                self.expect(Token::Rp)?;
                Ok(ExprNode::clamp(value, lo, hi))
            }
            Token::Lp => {
                let expr = self.parse_expr()?;
                self.expect(Token::Rp)?;
                Ok(expr)
            }
            _ => Err(Err::Tok),
        }
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
        if let Some(Token::Id(id)) = self.peek() {
            if id == "soft" {
                self.next();
                self.expect(Token::Lp)?;
                return Ok(true);
            }
        }
        Ok(false)
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
    Comma,
    Semi,
    Colon,
    Op(Operator),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Operator {
    Add,
    Sub,
    Mul,
    Div,
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

#[derive(Clone, Debug)]
enum ExprNode {
    Number(f64),
    Bool(bool),
    Field(FieldRef),
    UnaryNeg(Box<ExprNode>),
    Binary {
        op: BinaryOp,
        lhs: Box<ExprNode>,
        rhs: Box<ExprNode>,
    },
    Log2(Box<ExprNode>),
    Select {
        cond: Box<ExprNode>,
        when_true: Box<ExprNode>,
        when_false: Box<ExprNode>,
    },
    Clamp {
        value: Box<ExprNode>,
        lo: Box<ExprNode>,
        hi: Box<ExprNode>,
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

    fn neg(expr: ExprNode) -> Self {
        ExprNode::UnaryNeg(Box::new(expr)).fold()
    }

    fn binary(op: BinaryOp, lhs: ExprNode, rhs: ExprNode) -> Self {
        ExprNode::Binary {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
        .fold()
    }

    fn log2(expr: ExprNode) -> Self {
        ExprNode::Log2(Box::new(expr)).fold()
    }

    fn select(cond: ExprNode, when_true: ExprNode, when_false: ExprNode) -> Self {
        ExprNode::Select {
            cond: Box::new(cond),
            when_true: Box::new(when_true),
            when_false: Box::new(when_false),
        }
        .fold()
    }

    fn clamp(value: ExprNode, lo: ExprNode, hi: ExprNode) -> Self {
        ExprNode::Clamp {
            value: Box::new(value),
            lo: Box::new(lo),
            hi: Box::new(hi),
        }
        .fold()
    }

    fn fold(self) -> Self {
        use ExprNode::*;
        match self {
            Number(_) | Bool(_) | Field(_) => self,
            UnaryNeg(expr) => {
                let expr = expr.fold();
                let node = UnaryNeg(Box::new(expr));
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
            Log2(expr) => {
                let expr = expr.fold();
                let node = Log2(Box::new(expr));
                node.const_value().map_or(node, ExprNode::from_value)
            }
            Select {
                cond,
                when_true,
                when_false,
            } => {
                let cond = cond.fold();
                let when_true = when_true.fold();
                let when_false = when_false.fold();
                if let Some(cond_val) = cond.const_value() {
                    if cond_val.as_bool() {
                        return when_true;
                    } else {
                        return when_false;
                    }
                }
                let node = Select {
                    cond: Box::new(cond),
                    when_true: Box::new(when_true),
                    when_false: Box::new(when_false),
                };
                node.const_value().map_or(node, ExprNode::from_value)
            }
            Clamp { value, lo, hi } => {
                let value = value.fold();
                let lo = lo.fold();
                let hi = hi.fold();
                if let (Some(v), Some(l), Some(h)) =
                    (value.const_value(), lo.const_value(), hi.const_value())
                {
                    let clamped = v.as_f64().max(l.as_f64()).min(h.as_f64());
                    return ExprNode::number(clamped);
                }
                Clamp {
                    value: Box::new(value),
                    lo: Box::new(lo),
                    hi: Box::new(hi),
                }
            }
        }
    }

    fn const_value(&self) -> Option<Value> {
        use ExprNode::*;
        match self {
            Number(n) => Some(Value::from_f64(*n)),
            Bool(b) => Some(Value::from_bool(*b)),
            Field(_) => None,
            UnaryNeg(expr) => expr.const_value().map(|v| Value::from_f64(-v.as_f64())),
            Binary { op, lhs, rhs } => {
                let l = lhs.const_value()?;
                let r = rhs.const_value()?;
                Some(op.apply(l, r))
            }
            Log2(expr) => expr
                .const_value()
                .map(|v| Value::from_f64(v.as_f64().log2())),
            Select {
                cond,
                when_true,
                when_false,
            } => {
                let cond = cond.const_value()?;
                if cond.as_bool() {
                    when_true.const_value()
                } else {
                    when_false.const_value()
                }
            }
            Clamp { value, lo, hi } => {
                let v = value.const_value()?;
                let lo = lo.const_value()?;
                let hi = hi.const_value()?;
                Some(Value::from_f64(
                    v.as_f64().max(lo.as_f64()).min(hi.as_f64()),
                ))
            }
        }
    }

    fn evaluate(&self, ctx: &Ctx) -> Value {
        use ExprNode::*;
        match self {
            Number(n) => Value::from_f64(*n),
            Bool(b) => Value::from_bool(*b),
            Field(field) => field.read(ctx),
            UnaryNeg(expr) => Value::from_f64(-expr.evaluate(ctx).as_f64()),
            Binary { op, lhs, rhs } => {
                let l = lhs.evaluate(ctx);
                let r = rhs.evaluate(ctx);
                op.apply(l, r)
            }
            Log2(expr) => Value::from_f64(expr.evaluate(ctx).as_f64().log2()),
            Select {
                cond,
                when_true,
                when_false,
            } => {
                if cond.evaluate(ctx).as_bool() {
                    when_true.evaluate(ctx)
                } else {
                    when_false.evaluate(ctx)
                }
            }
            Clamp { value, lo, hi } => {
                let val = value.evaluate(ctx).as_f64();
                let lo_v = lo.evaluate(ctx).as_f64();
                let hi_v = hi.evaluate(ctx).as_f64();
                Value::from_f64(val.max(lo_v).min(hi_v))
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

    fn as_u32(self) -> u32 {
        self.as_f64().round() as u32
    }

    fn as_u8(self) -> u8 {
        self.as_f64().round() as u8
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
            b'0'..=b'9' | b'.' => {
                let start = index;
                index += 1;
                while index < bytes.len() && (bytes[index].is_ascii_digit() || bytes[index] == b'.')
                {
                    index += 1;
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
    fn evaluates_soft_rules() {
        let script = "soft (wg, r / 4, 0.25, true);";
        let program = Program::parse(script).unwrap();
        let out = program.evaluate(&ctx());
        assert_eq!(out.soft.len(), 1);
    }
}
