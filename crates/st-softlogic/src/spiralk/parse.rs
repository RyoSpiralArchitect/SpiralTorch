//! Parser for the SpiralK soft-logic DSL.

use anyhow::{anyhow, bail, Context, Result};

use super::ir::{
    Backend, Document, FeedbackBlock, Layout, Precision, RefractBlock, RefractOpPolicy, SyncBlock,
    TargetSpec,
};

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Ident(String),
    Number(String),
    String(String),
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Colon,
    Comma,
    Arrow,
}

pub fn parse_spiralk(src: &str) -> Result<Document> {
    let tokens = tokenize(src)?;
    let mut parser = Parser::new(tokens);
    parser.parse_document()
}

fn tokenize(src: &str) -> Result<Vec<Token>> {
    let mut tokens = Vec::new();
    let mut chars = src.chars().peekable();

    while let Some(&ch) = chars.peek() {
        if ch.is_whitespace() {
            chars.next();
            continue;
        }
        match ch {
            '{' => {
                chars.next();
                tokens.push(Token::LBrace);
            }
            '}' => {
                chars.next();
                tokens.push(Token::RBrace);
            }
            '[' => {
                chars.next();
                tokens.push(Token::LBracket);
            }
            ']' => {
                chars.next();
                tokens.push(Token::RBracket);
            }
            ':' => {
                chars.next();
                tokens.push(Token::Colon);
            }
            ',' => {
                chars.next();
                tokens.push(Token::Comma);
            }
            '-' => {
                chars.next();
                match chars.peek() {
                    Some('>') => {
                        chars.next();
                        tokens.push(Token::Arrow);
                    }
                    Some(next) if next.is_ascii_digit() => {
                        let mut number = String::from("-");
                        while let Some(&c) = chars.peek() {
                            if c.is_ascii_digit() || c == '.' {
                                number.push(c);
                                chars.next();
                            } else {
                                break;
                            }
                        }
                        tokens.push(Token::Number(number));
                    }
                    _ => bail!("unexpected '-' without '>' or digits in SpiralK source"),
                }
            }
            '"' => {
                chars.next();
                let mut content = String::new();
                let mut closed = false;
                while let Some(&c) = chars.peek() {
                    chars.next();
                    if c == '"' {
                        closed = true;
                        break;
                    }
                    if c == '\\' {
                        if let Some(&next_c) = chars.peek() {
                            chars.next();
                            content.push(next_c);
                        } else {
                            bail!("unterminated escape sequence in string literal");
                        }
                    } else {
                        content.push(c);
                    }
                }
                if !closed {
                    bail!("unterminated string literal in SpiralK source");
                }
                tokens.push(Token::String(content));
            }
            _ => {
                if is_ident_start(ch) {
                    let mut ident = String::new();
                    ident.push(ch);
                    chars.next();
                    while let Some(&c) = chars.peek() {
                        if is_ident_continue(c) {
                            ident.push(c);
                            chars.next();
                        } else {
                            break;
                        }
                    }
                    tokens.push(Token::Ident(ident));
                } else if ch.is_ascii_digit() {
                    let mut number = String::new();
                    number.push(ch);
                    chars.next();
                    while let Some(&c) = chars.peek() {
                        if c.is_ascii_digit() || c == '.' {
                            number.push(c);
                            chars.next();
                        } else {
                            break;
                        }
                    }
                    tokens.push(Token::Number(number));
                } else {
                    bail!("unexpected character '{ch}' in SpiralK source");
                }
            }
        }
    }

    Ok(tokens)
}

fn is_ident_start(ch: char) -> bool {
    ch == '_' || ch.is_alphabetic()
}

fn is_ident_continue(ch: char) -> bool {
    ch == '_' || ch == '-' || ch.is_alphanumeric()
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn parse_document(&mut self) -> Result<Document> {
        let mut doc = Document::default();
        while !self.is_eof() {
            if self.peek_is_rbrace() {
                bail!("unexpected '}}' at document scope");
            }
            match self.peek_ident()? {
                Some(keyword) if keyword == "refract" => {
                    self.next_token();
                    let block = self.parse_refract_block()?;
                    doc.refracts.push(block);
                }
                Some(keyword) if keyword == "sync" => {
                    self.next_token();
                    let block = self.parse_sync_block()?;
                    doc.syncs.push(block);
                }
                Some(keyword) if keyword == "feedback" => {
                    self.next_token();
                    let block = self.parse_feedback_block()?;
                    doc.feedbacks.push(block);
                }
                Some(keyword) => {
                    bail!("unexpected keyword '{keyword}' at document scope");
                }
                None => break,
            }
        }
        Ok(doc)
    }

    fn parse_refract_block(&mut self) -> Result<RefractBlock> {
        let name = self.expect_ident()?;
        self.expect_token(Token::LBrace)?;

        let mut target: Option<TargetSpec> = None;
        let mut precision: Option<Precision> = None;
        let mut layout: Option<Layout> = None;
        let mut schedule: Option<String> = None;
        let mut backend: Option<Backend> = None;
        let mut policies: Vec<RefractOpPolicy> = Vec::new();

        while !self.consume_token(Token::RBrace)? {
            let key = self.expect_ident()?;
            match key.as_str() {
                "target" => {
                    self.expect_token(Token::Colon)?;
                    target = Some(self.parse_target_spec()?);
                }
                "precision" => {
                    self.expect_token(Token::Colon)?;
                    let ident = self.expect_ident()?;
                    let value = match ident.as_str() {
                        "fp32" => Precision::Fp32,
                        "fp16" => Precision::Fp16,
                        "bf16" => Precision::Bf16,
                        other => bail!("unknown precision '{other}'"),
                    };
                    precision = Some(value);
                }
                "layout" => {
                    self.expect_token(Token::Colon)?;
                    let ident = self.expect_ident()?;
                    let value = match ident.as_str() {
                        "nhwc" => Layout::NHWC,
                        "nchw" => Layout::NCHW,
                        "blocked" => Layout::Blocked,
                        other => bail!("unknown layout '{other}'"),
                    };
                    layout = Some(value);
                }
                "schedule" => {
                    self.expect_token(Token::Colon)?;
                    let ident = self.expect_ident()?;
                    schedule = Some(ident);
                }
                "backend" => {
                    self.expect_token(Token::Colon)?;
                    let ident = self.expect_ident()?;
                    let value = match ident.as_str() {
                        "WGPU" | "wgpu" => Backend::WGPU,
                        "MPS" | "mps" => Backend::MPS,
                        "CUDA" | "cuda" => Backend::CUDA,
                        "CPU" | "cpu" => Backend::CPU,
                        other => bail!("unknown backend '{other}'"),
                    };
                    backend = Some(value);
                }
                "op" => {
                    self.expect_token(Token::Colon)?;
                    let op_name = self.expect_ident()?;
                    let mut flags = Vec::new();
                    if self.consume_token(Token::Arrow)? {
                        loop {
                            let flag = self.expect_ident()?;
                            flags.push(flag);
                            if !self.consume_token(Token::Comma)? {
                                break;
                            }
                        }
                    }
                    policies.push(RefractOpPolicy { op: op_name, flags });
                }
                other => bail!("unknown statement '{other}' in refract block"),
            }
            self.consume_token(Token::Comma)?;
        }

        let target = target.ok_or_else(|| anyhow!("refract block '{name}' missing target"))?;

        Ok(RefractBlock {
            name,
            target,
            precision,
            layout,
            schedule,
            backend,
            policies,
        })
    }

    fn parse_sync_block(&mut self) -> Result<SyncBlock> {
        let name = self.expect_ident()?;
        self.expect_token(Token::LBrace)?;

        let mut pairs: Option<Vec<String>> = None;
        let mut tolerance: Option<f32> = None;

        while !self.consume_token(Token::RBrace)? {
            let key = self.expect_ident()?;
            match key.as_str() {
                "pairs" => {
                    self.expect_token(Token::Colon)?;
                    self.expect_token(Token::LBracket)?;
                    let mut values = Vec::new();
                    while !self.consume_token(Token::RBracket)? {
                        let ident = self.expect_ident()?;
                        values.push(ident);
                        if !self.consume_token(Token::Comma)? {
                            self.expect_token(Token::RBracket)?;
                            break;
                        }
                    }
                    pairs = Some(values);
                }
                "tolerance" => {
                    self.expect_token(Token::Colon)?;
                    let value = self.expect_number()?;
                    tolerance = Some(value);
                }
                other => bail!("unknown field '{other}' in sync block"),
            }
            self.consume_token(Token::Comma)?;
        }

        let pairs = pairs.ok_or_else(|| anyhow!("sync block '{name}' missing pairs"))?;
        let tolerance =
            tolerance.ok_or_else(|| anyhow!("sync block '{name}' missing tolerance"))?;

        Ok(SyncBlock {
            name,
            pairs,
            tolerance,
        })
    }

    fn parse_feedback_block(&mut self) -> Result<FeedbackBlock> {
        let name = self.expect_ident()?;
        self.expect_token(Token::LBrace)?;

        let mut export_path: Option<String> = None;
        let mut metrics: Option<Vec<String>> = None;

        while !self.consume_token(Token::RBrace)? {
            let key = self.expect_ident()?;
            match key.as_str() {
                "export" => {
                    self.expect_token(Token::Colon)?;
                    let value = match self.next_token() {
                        Some(Token::String(s)) => s,
                        other => {
                            bail!("expected string literal for export, got {:?}", other);
                        }
                    };
                    export_path = Some(value);
                }
                "metrics" => {
                    self.expect_token(Token::Colon)?;
                    self.expect_token(Token::LBracket)?;
                    let mut values = Vec::new();
                    while !self.consume_token(Token::RBracket)? {
                        let ident = self.expect_ident()?;
                        values.push(ident);
                        if !self.consume_token(Token::Comma)? {
                            self.expect_token(Token::RBracket)?;
                            break;
                        }
                    }
                    metrics = Some(values);
                }
                other => bail!("unknown field '{other}' in feedback block"),
            }
            self.consume_token(Token::Comma)?;
        }

        let export_path =
            export_path.ok_or_else(|| anyhow!("feedback block '{name}' missing export"))?;
        let metrics = metrics.unwrap_or_default();

        Ok(FeedbackBlock {
            name,
            export_path,
            metrics,
        })
    }

    fn parse_target_spec(&mut self) -> Result<TargetSpec> {
        let kind = self.expect_ident()?;
        self.expect_token(Token::Colon)?;
        let name = self.expect_ident()?;
        let spec = match kind.as_str() {
            "graph" => TargetSpec::Graph(name),
            "prsn" => TargetSpec::Prsn(name),
            other => bail!("unknown target kind '{other}'"),
        };
        Ok(spec)
    }

    fn expect_ident(&mut self) -> Result<String> {
        match self.next_token() {
            Some(Token::Ident(s)) => Ok(s),
            other => bail!("expected identifier, got {:?}", other),
        }
    }

    fn expect_number(&mut self) -> Result<f32> {
        match self.next_token() {
            Some(Token::Number(n)) => n
                .parse::<f32>()
                .with_context(|| format!("failed to parse '{n}' as float")),
            other => bail!("expected number, got {:?}", other),
        }
    }

    fn expect_token(&mut self, expected: Token) -> Result<()> {
        match self.next_token() {
            Some(tok) if tok == expected => Ok(()),
            other => bail!("expected {:?}, got {:?}", expected, other),
        }
    }

    fn consume_token(&mut self, target: Token) -> Result<bool> {
        if matches!(self.peek_token(), Some(tok) if tok == &target) {
            self.next_token();
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn peek_ident(&self) -> Result<Option<String>> {
        match self.peek_token() {
            Some(Token::Ident(s)) => Ok(Some(s.clone())),
            Some(_) => Ok(None),
            None => Ok(None),
        }
    }

    fn peek_token(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn next_token(&mut self) -> Option<Token> {
        if self.pos >= self.tokens.len() {
            None
        } else {
            let tok = self.tokens[self.pos].clone();
            self.pos += 1;
            Some(tok)
        }
    }

    fn is_eof(&self) -> bool {
        self.pos >= self.tokens.len()
    }

    fn peek_is_rbrace(&self) -> bool {
        matches!(self.peek_token(), Some(Token::RBrace))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_sample_document() {
        let src = r#"
        refract main {
          target: graph:ZSpaceEncoder
          precision: bf16
          layout: nhwc
          schedule: cooperative
          backend: WGPU
          op: attention -> fuse_softmax, stable_grad
        }

        sync merge_01 {
          pairs: [Σeve, Raqel],
          tolerance: 0.06
        }

        feedback z01143A {
          export: "runs/01143A",
          metrics: [phase_deviation, collapse_resonance, kernel_cache_hits]
        }
        "#;

        let doc = parse_spiralk(src).expect("parse should succeed");
        assert_eq!(doc.refracts.len(), 1);
        assert_eq!(doc.syncs.len(), 1);
        assert_eq!(doc.feedbacks.len(), 1);

        let refract = &doc.refracts[0];
        assert_eq!(refract.name, "main");
        assert!(matches!(refract.target, TargetSpec::Graph(ref g) if g == "ZSpaceEncoder"));
        assert!(matches!(refract.precision, Some(Precision::Bf16)));
        assert!(matches!(refract.layout, Some(Layout::NHWC)));
        assert_eq!(refract.policies.len(), 1);
        assert_eq!(
            refract.policies[0].flags,
            vec!["fuse_softmax", "stable_grad"]
        );

        let sync = &doc.syncs[0];
        assert_eq!(sync.pairs, vec!["Σeve", "Raqel"]);
        assert!((sync.tolerance - 0.06).abs() < f32::EPSILON);

        let feedback = &doc.feedbacks[0];
        assert_eq!(feedback.metrics.len(), 3);
    }
}
