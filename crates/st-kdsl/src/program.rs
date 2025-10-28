// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::fmt;

use crate::{Ctx, Err, Hard, Out, SoftRule};

use std::collections::HashMap;

/// Compiled representation of a SpiralK DSL program.
#[derive(Clone, Debug)]
pub struct Program {
    stmts: Vec<Stmt>,
    locals: usize,
}

const MAX_LOOP_ITERATIONS: usize = 65_536;

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

        Self::execute_block(&self.stmts, ctx, &mut locals, &mut hard, &mut soft);

        Out { hard, soft }
    }

    fn execute_block(
        block: &[Stmt],
        ctx: &Ctx,
        locals: &mut [Value],
        hard: &mut Hard,
        soft: &mut Vec<SoftRule>,
    ) {
        for stmt in block {
            match stmt {
                Stmt::Assign(assign) => assign.apply(ctx, locals, hard),
                Stmt::Let(binding) => binding.store(ctx, locals),
                Stmt::Set(update) => update.store(ctx, locals),
                Stmt::Soft(rule) => rule.apply(ctx, locals, soft),
                Stmt::If(branch) => {
                    let cond = branch
                        .const_cond
                        .unwrap_or_else(|| branch.cond.evaluate(ctx, locals).as_bool());
                    if cond {
                        Self::execute_block(&branch.then_branch, ctx, locals, hard, soft);
                    } else {
                        Self::execute_block(&branch.else_branch, ctx, locals, hard, soft);
                    }
                }
                Stmt::While(loop_stmt) => {
                    loop_stmt.run(ctx, locals, hard, soft);
                }
            }
        }
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
}

#[derive(Clone, Debug)]
struct AssignStmt {
    field: Field,
    expr: ExprNode,
    const_value: Option<Value>,
}

impl AssignStmt {
    fn apply(&self, ctx: &Ctx, locals: &[Value], hard: &mut Hard) {
        let value = self
            .const_value
            .unwrap_or_else(|| self.expr.evaluate(ctx, locals));
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
    fn store(&self, ctx: &Ctx, locals: &mut [Value]) {
        let value = self
            .const_value
            .unwrap_or_else(|| self.expr.evaluate(ctx, locals));
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
    fn store(&self, ctx: &Ctx, locals: &mut [Value]) {
        let value = self
            .const_value
            .unwrap_or_else(|| self.expr.evaluate(ctx, locals));
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
    fn apply(&self, ctx: &Ctx, locals: &mut [Value], soft: &mut Vec<SoftRule>) {
        let should_apply = self
            .const_cond
            .unwrap_or_else(|| self.cond_expr.evaluate(ctx, locals).as_bool());
        if !should_apply {
            return;
        }

        let value = self
            .const_value
            .unwrap_or_else(|| self.value_expr.evaluate(ctx, locals));
        let weight = self
            .const_weight
            .unwrap_or_else(|| self.weight_expr.evaluate(ctx, locals).as_f64() as f32);

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
    fn run(&self, ctx: &Ctx, locals: &mut [Value], hard: &mut Hard, soft: &mut Vec<SoftRule>) {
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
                .unwrap_or_else(|| self.cond.evaluate(ctx, locals).as_bool());
            if !cond {
                break;
            }

            Program::execute_block(&self.body, ctx, locals, hard, soft);
            iterations += 1;
            if iterations >= MAX_LOOP_ITERATIONS {
                debug_assert!(
                    iterations < MAX_LOOP_ITERATIONS,
                    "while loop iteration cap reached"
                );
                break;
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
        let body = self.parse_block()?;
        Ok(Stmt::While(WhileStmt {
            cond,
            const_cond,
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

    fn consume_else(&mut self) -> Result<bool, Err> {
        if matches!(self.peek(), Some(Token::Else)) {
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
        }
    }

    fn evaluate(&self, ctx: &Ctx, locals: &[Value]) -> Value {
        use ExprNode::*;
        match self {
            Number(n) => Value::from_f64(*n),
            Bool(b) => Value::from_bool(*b),
            Field(field) => field.read(ctx),
            Local(id) => locals[*id],
            UnaryNeg(expr) => Value::from_f64(-expr.evaluate(ctx, locals).as_f64()),
            UnaryNot(expr) => Value::from_bool(!expr.evaluate(ctx, locals).as_bool()),
            Binary { op, lhs, rhs } => match op {
                BinaryOp::And => {
                    let left = lhs.evaluate(ctx, locals);
                    if !left.as_bool() {
                        Value::from_bool(false)
                    } else {
                        Value::from_bool(rhs.evaluate(ctx, locals).as_bool())
                    }
                }
                BinaryOp::Or => {
                    let left = lhs.evaluate(ctx, locals);
                    if left.as_bool() {
                        Value::from_bool(true)
                    } else {
                        Value::from_bool(rhs.evaluate(ctx, locals).as_bool())
                    }
                }
                _ => {
                    let l = lhs.evaluate(ctx, locals);
                    let r = rhs.evaluate(ctx, locals);
                    op.apply(l, r)
                }
            },
            Call(function, args) => function.eval_runtime(ctx, locals, args),
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

    fn eval_runtime(&self, ctx: &Ctx, locals: &[Value], args: &[ExprNode]) -> Value {
        match self {
            Function::Select => {
                let cond = args[0].evaluate(ctx, locals);
                if cond.as_bool() {
                    args[1].evaluate(ctx, locals)
                } else {
                    args[2].evaluate(ctx, locals)
                }
            }
            Function::Clamp => {
                let v = args[0].evaluate(ctx, locals).as_f64();
                let lo = args[1].evaluate(ctx, locals).as_f64();
                let hi = args[2].evaluate(ctx, locals).as_f64();
                Value::from_f64(v.max(lo).min(hi))
            }
            Function::Min => {
                let mut iter = args.iter();
                let mut best = iter
                    .next()
                    .map(|expr| expr.evaluate(ctx, locals).as_f64())
                    .unwrap();
                for expr in iter {
                    best = best.min(expr.evaluate(ctx, locals).as_f64());
                }
                Value::from_f64(best)
            }
            Function::Max => {
                let mut iter = args.iter();
                let mut best = iter
                    .next()
                    .map(|expr| expr.evaluate(ctx, locals).as_f64())
                    .unwrap();
                for expr in iter {
                    best = best.max(expr.evaluate(ctx, locals).as_f64());
                }
                Value::from_f64(best)
            }
            Function::Log2
            | Function::Abs
            | Function::Floor
            | Function::Ceil
            | Function::Round
            | Function::Sqrt => {
                let value = args[0].evaluate(ctx, locals).as_f64();
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
                let base = args[0].evaluate(ctx, locals).as_f64();
                let exp = args[1].evaluate(ctx, locals).as_f64();
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
                if index + 1 < bytes.len() && bytes[index + 1] == b'=' {
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
                    "soft" => tokens.push(Token::Soft),
                    "let" => tokens.push(Token::Let),
                    "set" => tokens.push(Token::Set),
                    "if" => tokens.push(Token::If),
                    "else" => tokens.push(Token::Else),
                    "while" => tokens.push(Token::While),
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
    fn rejects_constant_true_while_loops() {
        let script = "while true { wg: 1; }";
        assert!(Program::parse(script).is_err());
    }
}
