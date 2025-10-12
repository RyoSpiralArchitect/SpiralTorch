use thiserror::Error;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, Default)]
pub struct Ctx { pub r:u32, pub c:u32, pub k:u32, pub sg: bool }

#[derive(Clone, Copy, Debug)]
pub struct Choice { pub use_2ce: Option<bool>, pub wg: Option<u32>, pub kl: Option<u32>, pub ch: Option<u32> }
impl Default for Choice {
    fn default() -> Self { Self { use_2ce: None, wg: None, kl: None, ch: None } }
}

#[derive(Clone, Copy, Debug)]
pub enum SoftRule {
    U2{ val: bool,  w:f32 },
    Wg{ val: u32,   w:f32 },
    Kl{ val: u32,   w:f32 },
    Ch{ val: u32,   w:f32 },
}

#[derive(Error, Debug)]
pub enum Err {
    #[error("parse error at pos {0}")]
    Parse(usize),
    #[error("invalid token")]
    Tok,
}

#[derive(Clone, Debug, PartialEq)]
enum Tok { Id(String), Num(f64), True, False, Lp, Rp, Comma, Semi, Colon, Op(String), Eof }

fn lex(src:&str)->Result<Vec<Tok>,Err>{
    let mut v=Vec::new(); let s=src.as_bytes(); let mut i=0;
    while i<s.len(){
        let c=s[i] as char;
        if c.is_whitespace(){ i+=1; continue; }
        match c {
            '('=>{v.push(Tok::Lp); i+=1;}
            ')'=>{v.push(Tok::Rp); i+=1;}
            ','=>{v.push(Tok::Comma); i+=1;}
            ';'=>{v.push(Tok::Semi); i+=1;}
            ':'=>{v.push(Tok::Colon); i+=1;}
            '0'..='9'|'.'=>{
                let st=i; i+=1;
                while i<s.len() and ((s[i] as char).is_ascii_digit() or s[i]==b'.'){ i+=1; }
                let n=std::str::from_utf8(&s[st..i]).unwrap().parse::<f64>().map_err(|_|Err::Parse(st))?;
                v.push(Tok::Num(n));
            }
            'a'..='z'|'A'..='Z'|'_'=>{
                let st=i; i+=1;
                while i<s.len(){ let ch=s[i] as char; if ch.is_ascii_alphanumeric()||ch=='_' {i+=1;} else {break;} }
                let id=std::str::from_utf8(&s[st..i]).unwrap().to_string();
                v.push(match id.as_str(){ "true"=>Tok::True, "false"=>Tok::False, _=>Tok::Id(id) });
            }
            _=>{
                if i+1<s.len(){
                    let two = &s[i..i+2];
                    let op = std::str::from_utf8(two).unwrap();
                    if ["<=",">=","==","!=","&&","||"].contains(&op){
                        v.push(Tok::Op(op.into())); i+=2; continue;
                    }
                }
                if ["+","-","*","/","<",">"].contains(&&*c.to_string()){
                    v.push(Tok::Op(c.to_string())); i+=1; continue;
                }
                return Err::Parse(i);
            }
        }
    }
    v.push(Tok::Eof);
    Ok(v)
}

#[derive(Clone)]
struct P{ t:Vec<Tok>, i:usize }
impl P{
    fn peek(&self)->&Tok{ self.t.get(self.i).unwrap() }
    fn eat(&mut self)->Tok{ let x=self.t[self.i].clone(); self.i+=1; x }
    fn accept(&mut self, want:&Tok)->bool{ if &self.t[self.i]==want { self.i+=1; true } else { false } }
    fn expect(&mut self, want:&Tok)->Result<(),Err>{ if self.accept(want){ Ok(()) } else { Err(Err::Tok) } }
}

#[derive(Clone,Debug)]
enum Expr{ F(f64), B(bool), Var(String), Call(String, Vec<Expr>), Bin(Box<Expr>,String,Box<Expr>), Neg(Box<Expr>) }

#[derive(Default)]
struct Program{
    defs: HashMap<String, (Vec<String>, Expr)>,
    assigns: Vec<(String, Expr)>,
    softs: Vec<(String, Expr, Expr, Expr)>, // soft(field,val,weight,cond)
}

fn parse(src:&str)->Result<Program,Err>{
    let toks=lex(src)?; let mut p=P{t:toks,i:0};
    let mut prog=Program::default();
    while *p.peek()!=Tok::Eof {
        match p.peek() {
            Tok::Id(id) if id=="def" => {
                p.eat(); // def
                let name = match p.eat(){ Tok::Id(s)=>s, _=>return Err(Err::Tok)};
                p.expect(&Tok::Lp)?;
                let mut params=Vec::new();
                if !p.accept(&Tok::Rp){
                    loop{
                        match p.eat(){ Tok::Id(s)=>params.push(s), _=>return Err(Err::Tok) }
                        if p.accept(&Tok::Comma){ continue } else { p.expect(&Tok::Rp)?; break }
                    }
                }
                p.expect(&Tok::Colon)?;
                let e=parse_expr(&mut p)?;
                prog.defs.insert(name, (params, e));
                let _ = p.accept(&Tok::Semi);
            }
            Tok::Id(id) if id=="soft" => {
                p.eat(); p.expect(&Tok::Lp)?;
                let field = match p.eat(){ Tok::Id(s)=>s, _=>return Err(Err::Tok)};
                p.expect(&Tok::Comma)?;
                let val = parse_expr(&mut p)?; p.expect(&Tok::Comma)?;
                let w   = parse_expr(&mut p)?; p.expect(&Tok::Comma)?;
                let cond= parse_expr(&mut p)?; p.expect(&Tok::Rp)?;
                prog.softs.push((field,val,w,cond));
                let _=p.accept(&Tok::Semi);
            }
            Tok::Id(id) if ["u2","wg","kl","ch"].contains(&id.as_str()) => {
                let key = if let Tok::Id(s) = p.eat(){ s } else { unreachable!() };
                p.expect(&Tok::Colon)?;
                let e = parse_expr(&mut p)?;
                prog.assigns.push((key,e));
                let _=p.accept(&Tok::Semi);
            }
            _ => { return Err(Err::Tok) }
        }
    }
    Ok(prog)
}

fn parse_expr(p:&mut P)->Result<Expr,Err>{ parse_or(p) }
fn parse_or(p:&mut P)->Result<Expr,Err>{
    let mut e=parse_and(p)?;
    loop { if let Tok::Op(op)=p.peek(){ if op=="||" { p.eat(); let r=parse_and(p)?; e=Expr::Bin(Box::new(e), "||".into(), Box::new(r)); continue } } break }
    Ok(e)
}
fn parse_and(p:&mut P)->Result<Expr,Err>{
    let mut e=parse_cmp(p)?;
    loop { if let Tok::Op(op)=p.peek(){ if op=="&&" { p.eat(); let r=parse_cmp(p)?; e=Expr::Bin(Box::new(e), "&&".into(), Box::new(r)); continue } } break }
    Ok(e)
}
fn parse_cmp(p:&mut P)->Result<Expr,Err>{
    let e=parse_add(p)?;
    if let Tok::Op(op)=p.peek().clone(){
        if ["<","<=",">",">=","==","!="].contains(&op.as_str()){
            p.eat(); let r=parse_add(p)?; return Ok(Expr::Bin(Box::new(e),op,Box::new(r)));
        }
    }
    Ok(e)
}
fn parse_add(p:&mut P)->Result<Expr,Err>{
    let mut e=parse_mul(p)?;
    loop {
        match p.peek() {
            Tok::Op(op) if op=="+" || op=="-" => { let op=op.clone(); p.eat(); let r=parse_mul(p)?; e=Expr::Bin(Box::new(e),op,Box::new(r)); }
            _=>break
        }
    }
    Ok(e)
}
fn parse_mul(p:&mut P)->Result<Expr,Err>{
    let mut e=parse_unary(p)?;
    loop {
        match p.peek() {
            Tok::Op(op) if op=="*" || op=="/" => { let op=op.clone(); p.eat(); let r=parse_unary(p)?; e=Expr::Bin(Box::new(e),op,Box::new(r)); }
            _=>break
        }
    }
    Ok(e)
}
fn parse_unary(p:&mut P)->Result<Expr,Err>{
    match p.peek() {
        Tok::Op(op) if op=="-" => { p.eat(); let x=parse_unary(p)?; Ok(Expr::Neg(Box::new(x))) }
        _ => parse_atom(p)
    }
}
fn parse_atom(p:&mut P)->Result<Expr,Err>{
    match p.eat(){
        Tok::Num(n)=>Ok(Expr::F(n)),
        Tok::True =>Ok(Expr::B(true)),
        Tok::False=>Ok(Expr::B(false)),
        Tok::Id(id) if id=="r" || id=="c" || id=="k" || id=="sg" => Ok(Expr::Var(id)),
        Tok::Id(id) if id=="log2" => { expect_lp(p)?; let x=parse_expr(p)?; expect_rp(p)?; Ok(Expr::Call(id, vec![x])) }
        Tok::Id(id) if id=="sel"  => { expect_lp(p)?; let a=parse_expr(p)?; expect_comma(p)?; let b=parse_expr(p)?; expect_comma(p)?; let c=parse_expr(p)?; expect_rp(p)?; Ok(Expr::Call(id, vec![a,b,c])) }
        Tok::Id(id) if id=="clamp"=> { expect_lp(p)?; let a=parse_expr(p)?; expect_comma(p)?; let b=parse_expr(p)?; expect_comma(p)?; let c=parse_expr(p)?; expect_rp(p)?; Ok(Expr::Call(id, vec![a,b,c])) }
        Tok::Id(id) => {
            if !matches!(p.peek(), Tok::Lp) { return Err(Err::Tok) }
            p.eat(); // (
            let mut args=Vec::new();
            if !matches!(p.peek(), Tok::Rp){
                loop{ let e=parse_expr(p)?; args.push(e); if matches!(p.peek(),Tok::Comma){ p.eat(); continue } break }
            }
            expect_rp(p)?;
            Ok(Expr::Call(id, args))
        }
        Tok::Lp => { let e=parse_expr(p)?; expect_rp(p)?; Ok(e) }
        _=>Err(Err::Tok)
    }
}
fn expect_lp(p:&mut P)->Result<(),Err>{ p.expect(&Tok::Lp) }
fn expect_rp(p:&mut P)->Result<(),Err>{ p.expect(&Tok::Rp) }
fn expect_comma(p:&mut P)->Result<(),Err>{ p.expect(&Tok::Comma) }

#[derive(Clone,Copy)]
enum V{ F(f64), B(bool) }
impl V{ fn as_f(self)->f64{ match self{V::F(x)=>x,V::B(b)=> if b{1.0}else{0.0}} } fn as_b(self)->bool{ match self{V::B(b)=>b,V::F(x)=> x!=0.0 } } }

fn eval_bin(op:&str, a:V, b:V)->V{
    match op {
        "+"=>V::F(a.as_f()+b.as_f()),
        "-"=>V::F(a.as_f()-b.as_f()),
        "*"=>V::F(a.as_f()*b.as_f()),
        "/"=>V::F(a.as_f()/b.as_f()),
        "<"=>V::B(a.as_f()<b.as_f()),
        "<="=>V::B(a.as_f()<=b.as_f()),
        ">"=>V::B(a.as_f()>b.as_f()),
        ">="=>V::B(a.as_f()>=b.as_f()),
        "=="=>V::B((a.as_f()-b.as_f()).abs()<1e-12 || (a.as_b()==b.as_b())),
        "!="=>V::B(!((a.as_f()-b.as_f()).abs()<1e-12) && !(a.as_b()==b.as_b())),
        "&&"=>V::B(a.as_b() && b.as_b()),
        "||"=>V::B(a.as_b() || b.as_b()),
        _=>V::F(0.0)
    }
}

fn eval_expr(e:&Expr, ctx:&Ctx, defs:&HashMap<String,(Vec<String>,Expr)>, locals:&HashMap<String,V>)->V{
    match e {
        Expr::F(x)=>V::F(*x),
        Expr::B(b)=>V::B(*b),
        Expr::Var(s)=>{
            if let Some(v)=locals.get(s){ *v } else {
                match s.as_str(){ "r"=>V::F(ctx.r as f64), "c"=>V::F(ctx.c as f64), "k"=>V::F(ctx.k as f64), "sg"=>V::B(ctx.sg), _=>V::F(0.0) }
            }
        }
        Expr::Neg(x)=>{ let v=eval_expr(x, ctx, defs, locals); V::F(-v.as_f()) }
        Expr::Bin(a,op,b)=>{ let va=eval_expr(a, ctx, defs, locals); let vb=eval_expr(b, ctx, defs, locals); eval_bin(op,va,vb) }
        Expr::Call(name, args)=>{
            match name.as_str(){
                "log2" => { let x=eval_expr(&args[0],ctx,defs,locals).as_f(); V::F(x.log2()) }
                "clamp"=> { let x=eval_expr(&args[0],ctx,defs,locals).as_f(); let lo=eval_expr(&args[1],ctx,defs,locals).as_f(); let hi=eval_expr(&args[2],ctx,defs,locals).as_f(); V::F(x.max(lo).min(hi)) }
                "sel"  => { let c=eval_expr(&args[0],ctx,defs,locals).as_b(); if c { eval_expr(&args[1],ctx,defs,locals) } else { eval_expr(&args[2],ctx,defs,locals) } }
                _ => {
                    if let Some((params, body)) = defs.get(name){
                        let mut map = HashMap::<String,V>::new();
                        for (i,pn) in params.iter().enumerate(){
                            let v = eval_expr(&args[i], ctx, defs, locals);
                            map.insert(pn.clone(), v);
                        }
                        return eval_expr(body, ctx, defs, &map);
                    } else {
                        V::F(0.0)
                    }
                }
            }
        }
    }
}

pub struct Out { pub hard: Choice, pub soft: Vec<SoftRule> }

pub fn eval_program(src:&str, ctx:&Ctx) -> Result<Out, Err> {
    let prog = parse(src)?;
    let mut hard=Choice::default();
    let empty_locals = HashMap::<String,V>::new();
    for (k,e) in prog.assigns.iter(){
        let v = eval_expr(e, ctx, &prog.defs, &empty_locals);
        match k.as_str(){
            "u2" => { hard.use_2ce = Some(v.as_b()); }
            "wg" => { hard.wg      = Some(v.as_f().round() as u32); }
            "kl" => { hard.kl      = Some(v.as_f().round() as u32); }
            "ch" => { hard.ch      = Some(v.as_f().round() as u32); }
            _ => {}
        }
    }
    let mut soft = Vec::<SoftRule>::new();
    for (field, val, w, cond) in prog.softs.iter(){
        let cnd = eval_expr(cond, ctx, &prog.defs, &empty_locals).as_b();
        if !cnd { continue; }
        let wt  = eval_expr(w,    ctx, &prog.defs, &empty_locals).as_f() as f32;
        match field.as_str(){
            "u2" => soft.push(SoftRule::U2{ val: eval_expr(val,ctx,&prog.defs,&empty_locals).as_b(), w: wt }),
            "wg" => soft.push(SoftRule::Wg{ val: eval_expr(val,ctx,&prog.defs,&empty_locals).as_f().round() as u32, w: wt }),
            "kl" => soft.push(SoftRule::Kl{ val: eval_expr(val,ctx,&prog.defs,&empty_locals).as_f().round() as u32, w: wt }),
            "ch" => soft.push(SoftRule::Ch{ val: eval_expr(val,ctx,&prog.defs,&empty_locals).as_f().round() as u32, w: wt }),
            _ => {}
        }
    }
    Ok(Out{ hard, soft })
}
