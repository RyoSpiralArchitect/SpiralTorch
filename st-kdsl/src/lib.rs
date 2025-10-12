use thiserror::Error;

#[derive(Clone, Copy, Debug, Default)]
pub struct Ctx { pub r:u32, pub c:u32, pub k:u32, pub sg: bool }

#[derive(Clone, Copy, Debug, Default)]
pub struct Choice { pub use_2ce: Option<bool>, pub wg: Option<u32>, pub kl: Option<u32>, pub ch: Option<u32>, pub mk: Option<u32> }
// mk: 0=bitonic, 1=shared, 2=warp

#[derive(Clone, Copy, Debug)]
pub enum SoftRule {
    U2{ val: bool,  w:f32 },
    Wg{ val: u32,   w:f32 },
    Kl{ val: u32,   w:f32 },
    Ch{ val: u32,   w:f32 },
    Mk{ val: u32,   w:f32 },
}

#[derive(Error, Debug)]
pub enum Err {
    #[error("parse error at pos {0}")]
    Parse(usize),
    #[error("invalid token")]
    Tok,
}

#[derive(Clone, Debug, PartialEq)]
enum Tok { Id(String), Num(f64), True, False, Lp, Rp, Comma, Semi, Colon, Op(String) }

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
                v.push(match id.as_str(){ "true"=>Tok::True,"false"=>Tok::False,_=>Tok::Id(id) });
            }
            _=>{
                let two = if i+1<s.len(){Some(((s[i]as char).to_string()+&(s[i+1]as char).to_string()))} else {None};
                if let Some(op) = two.as_ref().map(|x|x.as_str()){
                    if ["<=",">=","==","!=","&&","||"].contains(&op){ v.push(Tok::Op(op.to_string())); i+=2; continue; }
                }
                if ["+","-","*","/","<",">"].contains(&&*c.to_string()){ v.push(Tok::Op(c.to_string())); i+=1; continue; }
                return Err::Parse(i);
            }
        }
    }
    Ok(v)
}

#[derive(Clone)] struct P{ t:Vec<Tok>, i:usize }
impl P{ fn peek(&self)->Option<&Tok>{ self.t.get(self.i) } fn eat(&mut self)->Option<Tok>{ let x=self.t.get(self.i).cloned(); if x.is_some(){self.i+=1;} x } fn expect(&mut self, w:&Tok)->Result<(),Err>{ let x=self.eat().ok_or(Err::Tok)?; if &x==w {Ok(())} else {Err::Tok} } }

#[derive(Clone,Copy)] enum E{ F(f64), B(bool) }
impl E{ fn as_f(self)->f64{ match self{E::F(x)=>x,E::B(b)=> if b{1.0}else{0.0}} } fn as_b(self)->bool{ match self{E::B(b)=>b,E::F(x)=> x!=0.0 } } }

#[derive(Clone)]
enum Stmt{ Assign(Field, Box<dyn Fn(&Ctx)->E>), Soft(Field, Box<dyn Fn(&Ctx)->u32>, Box<dyn Fn(&Ctx)->f64>, Box<dyn Fn(&Ctx)->bool>) }

#[derive(Clone,Copy,PartialEq,Eq,Debug)]
enum Field { U2, Wg, Kl, Ch, Mk }

fn parse_field(p:&mut P)->Result<Field,Err>{
    match p.eat().ok_or(Err::Tok)?{
        Tok::Id(s) if s=="u2"=>Ok(Field::U2),
        Tok::Id(s) if s=="wg"=>Ok(Field::Wg),
        Tok::Id(s) if s=="kl"=>Ok(Field::Kl),
        Tok::Id(s) if s=="ch"=>Ok(Field::Ch),
        Tok::Id(s) if s=="mk"=>Ok(Field::Mk),
        _=>Err::Tok
    }
}

fn parse_stmt(p:&mut P)->Result<Stmt,Err>{
    if let Some(Tok::Id(id))=p.peek(){ if id=="soft" {
        p.eat(); p.expect(&Tok::Lp)?;
        let f=parse_field(p)?; p.expect(&Tok::Comma)?;
        let vf=parse_expr_u32(p)?; p.expect(&Tok::Comma)?;
        let wf=parse_expr_f64(p)?; p.expect(&Tok::Comma)?;
        let cf=parse_expr_bool(p)?; p.expect(&Tok::Rp)?;
        return Ok(Stmt::Soft(f, vf, wf, cf));
    }}
    let f=parse_field(p)?; p.expect(&Tok::Colon)?;
    let ef=parse_expr(p)?;
    Ok(Stmt::Assign(f, ef))
}

fn parse_prog(p:&mut P)->Result<Vec<Stmt>,Err>{ let mut v=Vec::new(); while p.peek().is_some(){ v.push(parse_stmt(p)?); if matches!(p.peek(),Some(Tok::Semi)){p.eat();} } Ok(v) }

fn parse_expr(p:&mut P)->Result<Box<dyn Fn(&Ctx)->E>,Err>{ parse_or(p) }
fn parse_or(p:&mut P)->Result<Box<dyn Fn(&Ctx)->E>,Err>{ let mut lhs=parse_and(p)?; while let Some(Tok::Op(op))=p.peek(){ if op=="||" { p.eat(); let rhs=parse_and(p)?; let l=lhs; lhs=Box::new(move |c| E::B(l(c).as_b()||rhs(c).as_b())); } else { break; } } Ok(lhs) }
fn parse_and(p:&mut P)->Result<Box<dyn Fn(&Ctx)->E>,Err>{ let mut lhs=parse_cmp(p)?; while let Some(Tok::Op(op))=p.peek(){ if op=="&&" { p.eat(); let rhs=parse_cmp(p)?; let l=lhs; lhs=Box::new(move |c| E::B(l(c).as_b()&&rhs(c).as_b())); } else { break; } } Ok(lhs) }
fn parse_cmp(p:&mut P)->Result<Box<dyn Fn(&Ctx)->E>,Err>{ let lhs=parse_add(p)?; if let Some(Tok::Op(op))=p.peek().cloned(){ if ["<","<=",">",">=","==","!="].contains(&op.as_str()){
    p.eat(); let rhs=parse_add(p)?; return Ok(Box::new(move |c| { let a=lhs(c).as_f(); let b=rhs(c).as_f();
        E::B(match op.as_str(){ "<"=>(a<b), "<="=>(a<=b), ">"=>(a>b), ">="=>(a>=b), "=="=>(a==b), "!="=>(a!=b), _=>false }) })); } } Ok(lhs) }
fn parse_add(p:&mut P)->Result<Box<dyn Fn(&Ctx)->E>,Err>{ let mut lhs=parse_mul(p)?; loop { match p.peek(){ Some(Tok::Op(op)) if op=="+" || op=="-" => { let op=op.clone(); p.eat(); let rhs=parse_mul(p)?; let l=lhs; lhs=Box::new(move |c| { let a=l(c).as_f(); let b=rhs(c).as_f(); E::F(if op=="+" { a+b } else { a-b }) }); } _=>break } } Ok(lhs) }
fn parse_mul(p:&mut P)->Result<Box<dyn Fn(&Ctx)->E>,Err>{ let mut lhs=parse_unary(p)?; loop { match p.peek(){ Some(Tok::Op(op)) if op=="*" || op=="/" => { let op=op.clone(); p.eat(); let rhs=parse_unary(p)?; let l=lhs; lhs=Box::new(move |c| { let a=l(c).as_f(); let b=rhs(c).as_f(); E::F(if op=="*" { a*b } else { a/b }) }); } _=>break } } Ok(lhs) }
fn parse_unary(p:&mut P)->Result<Box<dyn Fn(&Ctx)->E>,Err>{ match p.peek(){ Some(Tok::Op(op)) if op=="-" => { p.eat(); let x=parse_unary(p)?; Ok(Box::new(move |c| E::F(-x(c).as_f()))) } _=>parse_atom(p) } }
fn parse_atom(p:&mut P)->Result<Box<dyn Fn(&Ctx)->E>,Err>{
    match p.eat().ok_or(Err::Tok)?{
        Tok::Num(n)=>Ok(Box::new(move |_| E::F(n))),
        Tok::True =>Ok(Box::new(move |_| E::B(true))),
        Tok::False=>Ok(Box::new(move |_| E::B(false))),
        Tok::Id(id) if id=="r" => Ok(Box::new(move |c| E::F(c.r as f64))),
        Tok::Id(id) if id=="c" => Ok(Box::new(move |c| E::F(c.c as f64))),
        Tok::Id(id) if id=="k" => Ok(Box::new(move |c| E::F(c.k as f64))),
        Tok::Id(id) if id=="sg"=> Ok(Box::new(move |c| E::B(c.sg))),
        Tok::Id(id) if id=="log2" => { expect_lp(p)?; let x=parse_expr_f64(p)?; expect_rp(p)?; Ok(Box::new(move |c| E::F((x(c)).log2()))) }
        Tok::Id(id) if id=="sel" => { expect_lp(p)?; let cb=parse_expr_bool(p)?; expect_comma(p)?; let a=parse_expr(p)?; expect_comma(p)?; let b=parse_expr(p)?; expect_rp(p)?; Ok(Box::new(move |c| if cb(c){a(c)} else {b(c)})) }
        Tok::Id(id) if id=="clamp" => { expect_lp(p)?; let x=parse_expr_f64(p)?; expect_comma(p)?; let lo=parse_expr_f64(p)?; expect_comma(p)?; let hi=parse_expr_f64(p)?; expect_rp(p)?; Ok(Box::new(move |c| { let v=x(c); E::F(v.max(lo(c)).min(hi(c))) })) }
        Tok::Lp => { let e=parse_expr(p)?; expect_rp(p)?; Ok(e) }
        _=> Err(Err::Tok)
    }
}
fn expect_lp(p:&mut P)->Result<(),Err>{ p.expect(&Tok::Lp) } fn expect_rp(p:&mut P)->Result<(),Err>{ p.expect(&Tok::Rp) } fn expect_comma(p:&mut P)->Result<(),Err>{ p.expect(&Tok::Comma) }
fn parse_expr_f64(p:&mut P)->Result<Box<dyn Fn(&Ctx)->f64>,Err>{ let e=parse_expr(p)?; Ok(Box::new(move |c| e(c).as_f())) }
fn parse_expr_u32(p:&mut P)->Result<Box<dyn Fn(&Ctx)->u32>,Err>{ let e=parse_expr(p)?; Ok(Box::new(move |c| e(c).as_f().round() as u32)) }
fn parse_expr_bool(p:&mut P)->Result<Box<dyn Fn(&Ctx)->bool>,Err>{ let e=parse_expr(p)?; Ok(Box::new(move |c| e(c).as_b())) }

pub struct Out { pub hard: Choice, pub soft: Vec<SoftRule> }

pub fn eval_program(src:&str, ctx:&Ctx) -> Result<Out, Err> {
    let toks=lex(src)?; let mut p=P{t:toks,i:0};
    let prog= parse_prog(&mut p)?;
    let mut hard=Choice::default();
    let mut soft=Vec::<SoftRule>::new();
    for s in prog {
        match s {
            Stmt::Assign(f, ef) => {
                match f {
                    Field::U2 => { hard.use_2ce = Some( ef(ctx).as_b() ); }
                    Field::Wg => { hard.wg      = Some( ef(ctx).as_f().round() as u32 ); }
                    Field::Kl => { hard.kl      = Some( ef(ctx).as_f().round() as u32 ); }
                    Field::Ch => { hard.ch      = Some( ef(ctx).as_f().round() as u32 ); }
                    Field::Mk => { hard.mk      = Some( ef(ctx).as_f().round() as u32 ); }
                }
            }
            Stmt::Soft(f, vf, wf, cf) => {
                if cf(ctx){
                    let w = wf(ctx) as f32;
                    match f {
                        Field::U2 => soft.push(SoftRule::U2{ val: vf(ctx)!=0, w }),
                        Field::Wg => soft.push(SoftRule::Wg{ val: vf(ctx), w }),
                        Field::Kl => soft.push(SoftRule::Kl{ val: vf(ctx), w }),
                        Field::Ch => soft.push(SoftRule::Ch{ val: vf(ctx), w }),
                        Field::Mk => soft.push(SoftRule::Mk{ val: vf(ctx), w }),
                    }
                }
            }
        }
    }
    Ok(Out{ hard, soft })
}
