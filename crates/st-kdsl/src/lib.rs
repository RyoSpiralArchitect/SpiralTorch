//! SpiralK: tiny K×Lisp-inspired expression DSL for runtime heuristics.
//!
//! Usage:
//! let src = r#"u2:(c>32768)||(k>128); wg:sel(c<4096,128,256); kl:sel(k>=32,32,sel(k>=16,16,8)); ch:sel(c>16384,8192,0)"#;
//! let choice = eval_program(src, Ctx{ r:1024, c:65536, k:4096, sg:true }).unwrap();

use std::fmt;
use thiserror::Error;

#[derive(Debug, Clone, Copy)]
pub struct Ctx {
    pub r: u32,   // rows
    pub c: u32,   // cols
    pub k: u32,   // k
    pub sg: bool, // subgroup available
}

#[derive(Debug, Clone, Copy)]
pub struct Choice {
    pub use_2ce: bool,
    pub wg: u32,
    pub kl: u32,
    pub ch: u32,
}

#[derive(Error, Debug)]
pub enum KdslError {
    #[error("lex error at pos {pos}: {msg}")]
    Lex { pos: usize, msg: String },
    #[error("parse error at pos {pos}: {msg}")]
    Parse { pos: usize, msg: String },
    #[error("eval error: {0}")]
    Eval(String),
}

type Result<T> = std::result::Result<T, KdslError>;

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Ident(String),
    Num(f64),
    True,
    False,
    Colon, Semi, Comma,
    LParen, RParen,
    Plus, Minus, Star, Slash, Percent, Caret,
    Lt, Le, Gt, Ge, Eq, Ne,
    And, Or, Not,
    Eof,
}

struct Lex<'a> {
    s: &'a [u8],
    i: usize,
}
impl<'a> Lex<'a> {
    fn new(s: &'a str) -> Self { Self { s: s.as_bytes(), i: 0 } }
    fn peek(&self) -> Option<u8> { self.s.get(self.i).copied() }
    fn bump(&mut self) -> Option<u8> { let ch=self.peek()?; self.i+=1; Some(ch) }
    fn skip_ws(&mut self) { while let Some(b) = self.peek() { if b.is_ascii_whitespace(){ self.i+=1 } else {break}}}
    fn ident_or_kw(&mut self, start:u8) -> Tok {
        let mut buf = vec![start];
        while let Some(b) = self.peek() {
            if b.is_ascii_alphanumeric() || b==b'_' { buf.push(b); self.i+=1 } else { break }
        }
        let s = String::from_utf8(buf).unwrap();
        match s.as_str() {
            "true"  => Tok::True,
            "false" => Tok::False,
            _ => Tok::Ident(s),
        }
    }
    fn number(&mut self, start:u8) -> Result<Tok> {
        let mut buf = vec![start];
        let mut dot = start == b'.';
        while let Some(b) = self.peek() {
            if b.is_ascii_digit() { buf.push(b); self.i+=1; }
            else if b==b'.' && !dot { dot=true; buf.push(b); self.i+=1; }
            else { break }
        }
        let s = String::from_utf8(buf).unwrap();
        let v: f64 = s.parse().map_err(|e| KdslError::Lex{ pos:self.i, msg: format!("invalid number {s}: {e}")})?;
        Ok(Tok::Num(v))
    }
    fn next(&mut self) -> Result<Tok> {
        self.skip_ws();
        let p = self.i;
        let Some(b) = self.bump() else { return Ok(Tok::Eof) };
        let t = match b {
            b'('=>Tok::LParen, b')'=>Tok::RParen,
            b':'=>Tok::Colon,  b';'=>Tok::Semi,   b','=>Tok::Comma,
            b'+'=>Tok::Plus,   b'-'=>Tok::Minus, b'*'=>Tok::Star, b'/'=>Tok::Slash, b'%'=>Tok::Percent,
            b'^'=>Tok::Caret,
            b'!'=>{
                if self.peek()==Some(b'='){ self.i+=1; Tok::Ne } else { Tok::Not }
            }
            b'='=>{
                if self.peek()==Some(b'='){ self.i+=1; Tok::Eq } else {
                    return Err(KdslError::Lex{ pos:p, msg:"single '=' not allowed".into() })
                }
            }
            b'<'=>{
                if self.peek()==Some(b'='){ self.i+=1; Tok::Le } else { Tok::Lt }
            }
            b'>'=>{
                if self.peek()==Some(b'='){ self.i+=1; Tok::Ge } else { Tok::Gt }
            }
            b'&'=>{
                if self.peek()==Some(b'&'){ self.i+=1; Tok::And } else {
                    return Err(KdslError::Lex{ pos:p, msg:"single '&'".into() })
                }
            }
            b'|'=>{
                if self.peek()==Some(b'|'){ self.i+=1; Tok::Or } else {
                    return Err(KdslError::Lex{ pos:p, msg:"single '|'".into() })
                }
            }
            b if b.is_ascii_digit() => return self.number(b),
            b'.' => return self.number(b),
            b if b.is_ascii_alphabetic() || b==b'_' => self.ident_or_kw(b),
            _ => return Err(KdslError::Lex{ pos:p, msg: format!("unexpected byte {b}")}),
        };
        Ok(t)
    }
}

#[derive(Clone)]
struct Parser<'a> {
    lx: Lex<'a>,
    cur: Tok,
    pos: usize,
}
impl<'a> Parser<'a> {
    fn new(s: &'a str) -> Result<Self> {
        let mut lx = Lex::new(s);
        let cur = lx.next()?;
        Ok(Self{ lx, cur, pos:0 })
    }
    fn bump(&mut self) -> Result<()> { self.pos=self.lx.i; self.cur=self.lx.next()?; Ok(()) }
    fn expect(&mut self, t: &Tok) -> Result<()> {
        if &self.cur == t { self.bump() } else {
            Err(KdslError::Parse{ pos:self.pos, msg: format!("expected {:?}, got {:?}", t, self.cur)})
        }
    }
    fn parse(&mut self) -> Result<Program> {
        let mut assigns = Vec::new();
        while self.cur != Tok::Eof {
            assigns.push(self.assignment()?);
            if self.cur == Tok::Semi { self.bump()?; }
        }
        Ok(Program{ assigns })
    }
    fn assignment(&mut self) -> Result<Assign> {
        let key = match &self.cur {
            Tok::Ident(s) if s=="u2" => { self.bump()?; Key::U2 }
            Tok::Ident(s) if s=="wg" => { self.bump()?; Key::Wg }
            Tok::Ident(s) if s=="kl" => { self.bump()?; Key::Kl }
            Tok::Ident(s) if s=="ch" => { self.bump()?; Key::Ch }
            other => return Err(KdslError::Parse{ pos:self.pos, msg: format!("expected key u2|wg|kl|ch, got {other:?}")}),
        };
        self.expect(&Tok::Colon)?;
        let expr = self.expr()?;
        Ok(Assign{ key, expr })
    }

    // Precedences
    fn expr(&mut self) -> Result<Expr> { self.or() }
    fn or(&mut self) -> Result<Expr> {
        let mut e = self.and()?;
        while self.cur == Tok::Or {
            self.bump()?;
            let r = self.and()?;
            e = Expr::Binary(Box::new(e), BinOp::Or, Box::new(r));
        }
        Ok(e)
    }
    fn and(&mut self) -> Result<Expr> {
        let mut e = self.eq()?;
        while self.cur == Tok::And {
            self.bump()?;
            let r = self.eq()?;
            e = Expr::Binary(Box::new(e), BinOp::And, Box::new(r));
        }
        Ok(e)
    }
    fn eq(&mut self) -> Result<Expr> {
        let mut e = self.rel()?;
        loop {
            match self.cur {
                Tok::Eq => { self.bump()?; let r=self.rel()?; e=Expr::Binary(Box::new(e), BinOp::Eq, Box::new(r)); }
                Tok::Ne => { self.bump()?; let r=self.rel()?; e=Expr::Binary(Box::new(e), BinOp::Ne, Box::new(r)); }
                _ => break
            }
        }
        Ok(e)
    }
    fn rel(&mut self) -> Result<Expr> {
        let mut e = self.add()?;
        loop {
            match self.cur {
                Tok::Lt => { self.bump()?; let r=self.add()?; e=Expr::Binary(Box::new(e), BinOp::Lt, Box::new(r)); }
                Tok::Le => { self.bump()?; let r=self.add()?; e=Expr::Binary(Box::new(e), BinOp::Le, Box::new(r)); }
                Tok::Gt => { self.bump()?; let r=self.add()?; e=Expr::Binary(Box::new(e), BinOp::Gt, Box::new(r)); }
                Tok::Ge => { self.bump()?; let r=self.add()?; e=Expr::Binary(Box::new(e), BinOp::Ge, Box::new(r)); }
                _ => break
            }
        }
        Ok(e)
    }
    fn add(&mut self) -> Result<Expr> {
        let mut e = self.mul()?;
        loop {
            match self.cur {
                Tok::Plus  => { self.bump()?; let r=self.mul()?; e=Expr::Binary(Box::new(e), BinOp::Add, Box::new(r)); }
                Tok::Minus => { self.bump()?; let r=self.mul()?; e=Expr::Binary(Box::new(e), BinOp::Sub, Box::new(r)); }
                _ => break
            }
        }
        Ok(e)
    }
    defn = None
    defn
    fn mul(&mut self) -> Result<Expr> {
        let mut e = self.pow()?;
        loop {
            match self.cur {
                Tok::Star    => { self.bump()?; let r=self.pow()?; e=Expr::Binary(Box::new(e), BinOp::Mul, Box::new(r)); }
                Tok::Slash   => { self.bump()?; let r=self.pow()?; e=Expr::Binary(Box::new(e), BinOp::Div, Box::new(r)); }
                Tok::Percent => { self.bump()?; let r=self.pow()?; e=Expr::Binary(Box::new(e), BinOp::Mod, Box::new(r)); }
                _ => break
            }
        }
        Ok(e)
    }
    fn pow(&mut self) -> Result<Expr> {
        let mut e = self.unary()?;
        if self.cur == Tok::Caret { // right associative
            self.bump()?;
            let r = self.pow()?;
            e = Expr::Binary(Box::new(e), BinOp::Pow, Box::new(r));
        }
        Ok(e)
    }
    fn unary(&mut self) -> Result<Expr> {
        match self.cur {
            Tok::Not   => { self.bump()?; Ok(Expr::Unary(UnOp::Not, Box::new(self.unary()?))) }
            Tok::Minus => { self.bump()?; Ok(Expr::Unary(UnOp::Neg, Box::new(self.unary()?))) }
            _ => self.primary(),
        }
    }
    fn primary(&mut self) -> Result<Expr> {
        match &self.cur {
            Tok::Num(v) => { let e=Expr::Num(*v); self.bump()?; Ok(e) }
            Tok::True   => { self.bump()?; Ok(Expr::Bool(true)) }
            Tok::False  => { self.bump()?; Ok(Expr::Bool(false)) }
            Tok::Ident(name) => {
                let name_s = name.clone(); self.bump()?;
                if self.cur == Tok::LParen {
                    self.bump()?; // (
                    let mut args = Vec::new();
                    if self.cur != Tok::RParen {
                        loop {
                            args.push(self.expr()?);
                            if self.cur == Tok::Comma { self.bump()?; continue }
                            break
                        }
                    }
                    self.expect(&Tok::RParen)?;
                    Ok(Expr::Call(name_s, args))
                } else {
                    match name_s.as_str() {
                        "r"=>Ok(Expr::Var(Var::R)), "c"=>Ok(Expr::Var(Var::C)),
                        "k"=>Ok(Expr::Var(Var::K)), "sg"=>Ok(Expr::Var(Var::Sg)),
                        _ => Err(KdslError::Parse{ pos:self.pos, msg: format!("unknown identifier '{name_s}'")}),
                    }
                }
            }
            Tok::LParen => { self.bump()?; let e=self.expr()?; self.expect(&Tok::RParen)?; Ok(e) }
            _ => Err(KdslError::Parse{ pos:self.pos, msg: format!("unexpected token {:?}", self.cur)}),
        }
    }
}

#[derive(Debug, Clone)]
struct Program { assigns: Vec<Assign> }

#[derive(Debug, Clone)]
struct Assign { key: Key, expr: Expr }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Key { U2, Wg, Kl, Ch }

#[derive(Debug, Clone)]
enum Expr {
    Num(f64),
    Bool(bool),
    Var(Var),
    Unary(UnOp, Box<Expr>),
    Binary(Box<Expr>, BinOp, Box<Expr>),
    Call(String, Vec<Expr>),
}
#[derive(Debug, Clone, Copy)]
enum Var { R, C, K, Sg }
#[derive(Debug, Clone, Copy)]
enum UnOp { Not, Neg }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BinOp { Add, Sub, Mul, Div, Mod, Pow, Lt, Le, Gt, Ge, Eq, Ne, And, Or }

#[derive(Debug, Clone, Copy)]
enum V { N(f64), B(bool) }
impl V {
    fn as_f(&self) -> f64 { match *self { V::N(x)=>x, V::B(b)=> if b {1.0} else {0.0} } }
    fn as_b(&self) -> bool { match *self { V::B(b)=>b, V::N(x)=> x!=0.0 } }
}

fn eval_expr(e:&Expr, ctx:Ctx) -> Result<V> {
    use BinOp::*;
    Ok(match e {
        Expr::Num(x) => V::N(*x),
        Expr::Bool(b)=> V::B(*b),
        Expr::Var(v) => match v {
            Var::R => V::N(ctx.r as f64),
            Var::C => V::N(ctx.c as f64),
            Var::K => V::N(ctx.k as f64),
            Var::Sg=> V::B(ctx.sg),
        },
        Expr::Unary(UnOp::Not, x) => V::B(!eval_expr(x, ctx)?.as_b()),
        Expr::Unary(UnOp::Neg, x) => V::N(-eval_expr(x, ctx)?.as_f()),
        Expr::Binary(l, op, r) => {
            let a = eval_expr(l, ctx)?;
            let b = eval_expr(r, ctx)?;
            match op {
                Add => V::N(a.as_f()+b.as_f()),
                Sub => V::N(a.as_f()-b.as_f()),
                Mul => V::N(a.as_f()*b.as_f()),
                Div => V::N(a.as_f()/b.as_f()),
                Mod => V::N(a.as_f()%b.as_f()),
                Pow => V::N(a.as_f().powf(b.as_f())),
                Lt => V::B(a.as_f()<b.as_f()),
                Le => V::B(a.as_f()<=b.as_f()),
                Gt => V::B(a.as_f()>b.as_f()),
                Ge => V::B(a.as_f()>=b.as_f()),
                Eq => V::B((a.as_f()-b.as_f()).abs()<1e-9 || (matches!(a,V::B(_)) && matches!(b,V::B(_)) && a.as_b()==b.as_b())),
                Ne => V::B(!((a.as_f()-b.as_f()).abs()<1e-9) && !(matches!(a,V::B(_)) && matches!(b,V::B(_)) && a.as_b()==b.as_b())),
                And=> V::B(a.as_b() && b.as_b()),
                Or => V::B(a.as_b() || b.as_b()),
            }
        }
        Expr::Call(name, args) => {
            let n = name.as_str();
            match n {
                "log2" => {
                    if args.len()!=1 { return Err(KdslError::Eval("log2 needs 1 arg".into())) }
                    V::N(eval_expr(&args[0], ctx)?.as_f().log2())
                }
                "clamp" => {
                    if args.len()!=3 { return Err(KdslError::Eval("clamp needs 3 args".into())) }
                    let x=eval_expr(&args[0], ctx)?.as_f();
                    let a=eval_expr(&args[1], ctx)?.as_f();
                    let b=eval_expr(&args[2], ctx)?.as_f();
                    V::N(x.max(a).min(b))
                }
                "sel" => {
                    if args.len()!=3 { return Err(KdslError::Eval("sel needs 3 args".into())) }
                    let cnd=eval_expr(&args[0], ctx)?.as_b();
                    if cnd { eval_expr(&args[1], ctx)? } else { eval_expr(&args[2], ctx)? }
                }
                "min" => {
                    if args.len()!=2 { return Err(KdslError::Eval("min needs 2 args".into())) }
                    V::N(eval_expr(&args[0], ctx)?.as_f().min(eval_expr(&args[1], ctx)?.as_f()))
                }
                "max" => {
                    if args.len()!=2 { return Err(KdslError::Eval("max needs 2 args".into())) }
                    V::N(eval_expr(&args[0], ctx)?.as_f().max(eval_expr(&args[1], ctx)?.as_f()))
                }
                "ceil" => {
                    if args.len()!=1 { return Err(KdslError::Eval("ceil needs 1 arg".into())) }
                    V::N(eval_expr(&args[0], ctx)?.as_f().ceil())
                }
                "floor" => {
                    if args.len()!=1 { return Err(KdslError::Eval("floor needs 1 arg".into())) }
                    V::N(eval_expr(&args[0], ctx)?.as_f().floor())
                }
                "round" => {
                    if args.len()!=1 { return Err(KdslError::Eval("round needs 1 arg".into())) }
                    V::N(eval_expr(&args[0], ctx)?.as_f().round())
                }
                _ => return Err(KdslError::Eval(format!("unknown func '{n}'"))),
            }
        }
    })
}

pub fn eval_program(src:&str, ctx:Ctx) -> Result<Choice> {
    let mut p = Parser::new(src)?.parse()?;

    // defaults
    let mut use_2ce = false;
    let mut wg: u32 = 128;
    let mut kl: u32 = 8;
    let mut ch: u32 = 0;

    for a in p.assigns.drain(..) {
        let v = eval_expr(&a.expr, ctx)?;
        match a.key {
            Key::U2 => use_2ce = v.as_b(),
            Key::Wg => { let x=v.as_f().round(); wg = x.max(1.0) as u32; }
            Key::Kl => { let x=v.as_f().round(); kl = x.max(1.0) as u32; }
            Key::Ch => { let x=v.as_f().round(); ch = x.max(0.0) as u32; }
        }
    }
    Ok(Choice{ use_2ce, wg, kl, ch })
}

// Debug
impl fmt::Display for Choice {
    fn fmt(&self, f:&mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Choice{{use_2ce:{}, wg:{}, kl:{}, ch:{}}}", self.use_2ce, self.wg, self.kl, self.ch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn smoke() {
        let s = r#"u2:(c>32768)||(k>128); wg:sel(c<4096,128,256); kl:sel(k>=32,32,sel(k>=16,16,8)); ch:sel(c>16384,8192,0)"#;
        let ch = eval_program(s, Ctx{ r:8, c:65536, k:4096, sg:true }).unwrap();
        assert!(ch.use_2ce && ch.wg==256 && ch.kl==32 && ch.ch==8192);
    }
}
