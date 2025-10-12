// SpiralK: ultra-small K-like expression DSL for heuristics.
// Supported:
//   - variables: r (rows), c (cols), k (k), sg (subgroup: 0/1)
//   - ops: + - * / ^, < > <= >= == !=, && ||
//   - functions: log2(x), clamp(x,a,b), sel(cond,a,b)
//   - returns a 4-tuple as "u2,wg,kl,ch" (bool,use_2ce; u32 wg_size,k_lane,chunk_cols)
// Example program (string):
//   "u2:(c>32768)||(k>128); wg:sel(c<4096,128,256); kl:sel(k>=32,32,sel(k>=16,16,8)); ch:sel(c>16384,8192,0)"
use std::str::Chars;

#[derive(Clone, Debug)]
pub enum Tok {
    Num(f64), Id(String),
    Plus, Minus, Star, Slash, Caret,
    LParen, RParen, Colon, Semi, Comma,
    Lt, Gt, Le, Ge, Eq, Ne, And, Or,
}
fn is_id_start(c:char)->bool{ c.is_ascii_alphabetic() || c=='_' }
fn is_id(c:char)->bool{ c.is_ascii_alphanumeric() || c=='_' }
pub fn lex(s:&str)->Vec<Tok>{
    let mut it = s.chars().peekable(); let mut out=Vec::new();
    while let Some(&c)=it.peek(){
        match c {
            ' ' | '\t' | '\n' | '\r' => { it.next(); }
            '+' => { it.next(); out.push(Tok::Plus); }
            '-' => { it.next(); out.push(Tok::Minus); }
            '*' => { it.next(); out.push(Tok::Star); }
            '/' => { it.next(); out.push(Tok::Slash); }
            '^' => { it.next(); out.push(Tok::Caret); }
            '(' => { it.next(); out.push(Tok::LParen); }
            ')' => { it.next(); out.push(Tok::RParen); }
            ':' => { it.next(); out.push(Tok::Colon); }
            ';' => { it.next(); out.push(Tok::Semi); }
            ',' => { it.next(); out.push(Tok::Comma); }
            '<' => { it.next(); if it.peek()==Some(&'='){it.next();out.push(Tok::Le)} else {out.push(Tok::Lt)} }
            '>' => { it.next(); if it.peek()==Some(&'='){it.next();out.push(Tok::Ge)} else {out.push(Tok::Gt)} }
            '=' => { it.next(); if it.peek()==Some(&'='){it.next();out.push(Tok::Eq)} }
            '!' => { it.next(); if it.peek()==Some(&'='){it.next();out.push(Tok::Ne)} }
            '&' => { it.next(); if it.peek()==Some(&'&'){it.next();out.push(Tok::And)} }
            '|' => { it.next(); if it.peek()==Some(&'|'){it.next();out.push(Tok::Or)} }
            c if c.is_ascii_digit() => {
                let mut num=String::new();
                while let Some(&d)=it.peek(){ if d.is_ascii_digit() || d=='.' { num.push(d); it.next(); } else { break; } }
                out.push(Tok::Num(num.parse().unwrap()));
            }
            c if is_id_start(c) => {
                let mut id=String::new();
                while let Some(&d)=it.peek(){ if is_id(d){ id.push(d); it.next(); } else { break; } }
                out.push(Tok::Id(id));
            }
            _ => { it.next(); }
        }
    }
    out
}

#[derive(Clone)]
pub struct Parser{ toks: Vec<Tok>, i: usize }
impl Parser{
    pub fn new(toks:Vec<Tok>)->Self{ Self{toks, i:0} }
    fn peek(&self)->Option<&Tok>{ self.toks.get(self.i) }
    fn bump(&mut self)->Option<Tok>{ if self.i<self.toks.len(){ self.i+=1; Some(self.toks[self.i-1].clone()) } else { None } }
    fn eat(&mut self, tk:&Tok)->bool{ if let Some(p)=self.peek(){ std::mem::discriminant(p)==std::mem::discriminant(tk) } else { false } }

    fn parse_expr(&mut self)->f64{ self.parse_or() }
    fn parse_or(&mut self)->f64{
        let mut v = self.parse_and();
        while matches!(self.peek(), Some(Tok::Or)){ self.bump(); let r=self.parse_and(); v = if v!=0.0 || r!=0.0 {1.0} else {0.0}; }
        v
    }
    fn parse_and(&mut self)->f64{
        let mut v = self.parse_cmp();
        while matches!(self.peek(), Some(Tok::And)){ self.bump(); let r=self.parse_cmp(); v = if v!=0.0 && r!=0.0 {1.0} else {0.0}; }
        v
    }
    fn parse_cmp(&mut self)->f64{
        let mut v = self.parse_add();
        loop{
            match self.peek() {
                Some(Tok::Lt)|Some(Tok::Gt)|Some(Tok::Le)|Some(Tok::Ge)|Some(Tok::Eq)|Some(Tok::Ne) => {
                    let op=self.bump().unwrap(); let r=self.parse_add();
                    v = match op {
                        Tok::Lt => (v<r) as i32 as f64,
                        Tok::Gt => (v>r) as i32 as f64,
                        Tok::Le => (v<=r) as i32 as f64,
                        Tok::Ge => (v>=r) as i32 as f64,
                        Tok::Eq => (v==r) as i32 as f64,
                        Tok::Ne => (v!=r) as i32 as f64,
                        _ => v
                    };
                }
                _ => break
            }
        }
        v
    }
    fn parse_add(&mut self)->f64{
        let mut v = self.parse_mul();
        loop{
            match self.peek(){
                Some(Tok::Plus) => { self.bump(); v += self.parse_mul(); }
                Some(Tok::Minus)=> { self.bump(); v -= self.parse_mul(); }
                _ => break
            }
        }
        v
    }
    fn parse_mul(&mut self)->f64{
        let mut v = self.parse_pow();
        loop{
            match self.peek(){
                Some(Tok::Star) => { self.bump(); v *= self.parse_pow(); }
                Some(Tok::Slash)=> { self.bump(); v /= self.parse_pow(); }
                _ => break
            }
        }
        v
    }
    fn parse_pow(&mut self)->f64{
        let mut v = self.parse_primary();
        while matches!(self.peek(), Some(Tok::Caret)) { self.bump(); let r=self.parse_primary(); v = v.powf(r); }
        v
    }
    fn parse_primary(&mut self)->f64{
        match self.bump().unwrap() {
            Tok::Num(n) => n,
            Tok::Id(id) => {
                if id=="log2" {
                    assert!(matches!(self.bump(), Some(Tok::LParen))); let v = self.parse_expr();
                    assert!(matches!(self.bump(), Some(Tok::RParen))); v.log2()
                } else if id=="clamp" {
                    assert!(matches!(self.bump(), Some(Tok::LParen))); let a=self.parse_expr();
                    assert!(matches!(self.bump(), Some(Tok::Comma))); let b=self.parse_expr();
                    assert!(matches!(self.bump(), Some(Tok::Comma))); let c=self.parse_expr();
                    assert!(matches!(self.bump(), Some(Tok::RParen))); a.max(b).min(c)
                } else if id=="sel" {
                    assert!(matches!(self.bump(), Some(Tok::LParen))); let cond=self.parse_expr();
                    assert!(matches!(self.bump(), Some(Tok::Comma))); let tv=self.parse_expr();
                    assert!(matches!(self.bump(), Some(Tok::Comma))); let fv=self.parse_expr();
                    assert!(matches!(self.bump(), Some(Tok::RParen))); if cond!=0.0 { tv } else { fv }
                } else if id=="r" || id=="c" || id=="k" || id=="sg" {
                    // sentinel: put back as variable read
                    self.i-=1; f64::NAN
                } else { panic!("unknown id {id}") }
            }
            Tok::LParen => { let v=self.parse_expr(); assert!(matches!(self.bump(), Some(Tok::RParen))); v }
            _ => panic!("syntax")
        }
    }
}

pub fn eval(expr:&str, r:f64, c:f64, k:f64, sg:f64)->f64{
    // A simplistic variable substitution pass for ids r,c,k,sg
    let replaced = expr.replace("r","(R)").replace("c","(C)").replace("k","(K)").replace("sg","(S)");
    let mut code = replaced.replace("R", &format!("{r}"))
                           .replace("C", &format!("{c}"))
                           .replace("K", &format!("{k}"))
                           .replace("S", &format!("{sg}"));
    let toks = lex(&code);
    let mut p = Parser::new(toks);
    p.parse_expr()
}

pub fn choose_from_program(prog:&str, rows:u32, cols:u32, k:u32, subgroup: bool) -> Option<(bool,u32,u32,u32)> {
    // Program format: "u2:<expr>; wg:<expr>; kl:<expr>; ch:<expr>"
    let r = rows as f64; let c = cols as f64; let kk = k as f64; let sg = if subgroup {1.0} else {0.0};
    fn find_block<'a>(s:&'a str, key:&str)->Option<&'a str>{
        let pat = format!("{key}:");
        s.find(&pat).map(|i| {
            let rest=&s[i+pat.len()..];
            let end=rest.find(';').unwrap_or(rest.len());
            &rest[..end]
        })
    }
    let u2s = find_block(prog,"u2")?; let wgs = find_block(prog,"wg")?; let kls = find_block(prog,"kl")?; let chs = find_block(prog,"ch")?;
    let u2v = eval(u2s, r,c,kk,sg) != 0.0;
    let wgv = eval(wgs, r,c,kk,sg).max(1.0).round() as u32;
    let klv = eval(kls, r,c,kk,sg).max(1.0).round() as u32;
    let chv = eval(chs, r,c,kk,sg).max(0.0).round() as u32;
    Some((u2v, wgv, klv, chv))
}
