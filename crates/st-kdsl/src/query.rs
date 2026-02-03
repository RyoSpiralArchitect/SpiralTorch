// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::collections::BTreeMap;

use crate::Err;

#[derive(Clone, Debug, PartialEq, Eq)]
struct QueryToken {
    text: String,
    start: usize,
}

fn query_error_pos(tokens: &[QueryToken], cursor: usize, input: &str) -> usize {
    tokens
        .get(cursor)
        .map(|token| token.start)
        .unwrap_or_else(|| input.len())
}

/// Ordering applied to query results.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OrderDirection {
    Asc,
    Desc,
}

/// High level representation of a KDsl query.
#[derive(Clone, Debug, PartialEq)]
pub struct QueryPlan {
    pub selects: Vec<String>,
    pub filters: Vec<Filter>,
    pub order: Option<(String, OrderDirection)>,
    pub limit: Option<usize>,
}

impl QueryPlan {
    pub fn new() -> Self {
        Self {
            selects: Vec::new(),
            filters: Vec::new(),
            order: None,
            limit: None,
        }
    }

    /// Executes the query plan against an in-memory dataset.
    pub fn execute(&self, rows: &[BTreeMap<String, f64>]) -> Vec<BTreeMap<String, f64>> {
        let mut matches: Vec<_> = rows
            .iter()
            .filter(|row| self.filters.iter().all(|filter| filter.matches(row)))
            .cloned()
            .collect();

        if let Some((column, direction)) = &self.order {
            matches.sort_by(|a, b| {
                let lhs = a.get(column).copied().unwrap_or_default();
                let rhs = b.get(column).copied().unwrap_or_default();
                match direction {
                    OrderDirection::Asc => lhs.partial_cmp(&rhs).unwrap(),
                    OrderDirection::Desc => rhs.partial_cmp(&lhs).unwrap(),
                }
            });
        }

        if let Some(limit) = self.limit {
            matches.truncate(limit);
        }

        if self.selects.is_empty() {
            return matches;
        }

        matches
            .into_iter()
            .map(|row| {
                self.selects
                    .iter()
                    .filter_map(|col| row.get(col).copied().map(|value| (col.clone(), value)))
                    .collect()
            })
            .collect()
    }
}

impl Default for QueryPlan {
    fn default() -> Self {
        Self::new()
    }
}

/// Supported filter predicates.
#[derive(Clone, Debug, PartialEq)]
pub enum Filter {
    Eq(String, f64),
    Neq(String, f64),
    Gt(String, f64),
    Lt(String, f64),
    Ge(String, f64),
    Le(String, f64),
}

impl Filter {
    fn matches(&self, row: &BTreeMap<String, f64>) -> bool {
        let column = match self {
            Filter::Eq(column, _)
            | Filter::Neq(column, _)
            | Filter::Gt(column, _)
            | Filter::Lt(column, _)
            | Filter::Ge(column, _)
            | Filter::Le(column, _) => column,
        };

        let candidate = row.get(column).copied().unwrap_or_default();
        match self {
            Filter::Eq(_, value) => (candidate - value).abs() <= f64::EPSILON,
            Filter::Neq(_, value) => (candidate - value).abs() > f64::EPSILON,
            Filter::Gt(_, value) => candidate > *value,
            Filter::Lt(_, value) => candidate < *value,
            Filter::Ge(_, value) => candidate >= *value,
            Filter::Le(_, value) => candidate <= *value,
        }
    }
}

/// Parses a KDsl query string into a structured plan.
pub fn compile(input: &str) -> Result<QueryPlan, Err> {
    let tokens = lex_query(input)?;
    if tokens.is_empty() {
        return Ok(QueryPlan::new());
    }

    let mut cursor = 0usize;
    let mut plan = QueryPlan::new();

    expect_keyword(&tokens, &mut cursor, "select", input)?;
    while cursor < tokens.len() {
        match tokens[cursor].text.as_str() {
            "where" => {
                cursor += 1;
                break;
            }
            "order" | "limit" => break,
            "," => cursor += 1,
            ident => {
                plan.selects.push(ident.to_string());
                cursor += 1;
            }
        }
    }

    if cursor < tokens.len() && tokens[cursor].text == "where" {
        cursor += 1;
    }

    while cursor < tokens.len() {
        match tokens[cursor].text.as_str() {
            "order" | "limit" => break,
            "and" => cursor += 1,
            ident => {
                let column = ident.to_string();
                cursor += 1;
                if cursor >= tokens.len() {
                    return Err(Err::Parse(query_error_pos(&tokens, cursor, input)));
                }
                let op = tokens[cursor].text.as_str().to_string();
                cursor += 1;
                if cursor >= tokens.len() {
                    return Err(Err::Parse(query_error_pos(&tokens, cursor, input)));
                }
                let value: f64 = tokens[cursor]
                    .text
                    .parse()
                    .map_err(|_| Err::Parse(query_error_pos(&tokens, cursor, input)))?;
                cursor += 1;
                let filter = match op.as_str() {
                    "=" => Filter::Eq(column, value),
                    "!=" => Filter::Neq(column, value),
                    ">" => Filter::Gt(column, value),
                    "<" => Filter::Lt(column, value),
                    ">=" => Filter::Ge(column, value),
                    "<=" => Filter::Le(column, value),
                    _ => return Err(Err::Parse(query_error_pos(&tokens, cursor, input))),
                };
                plan.filters.push(filter);
            }
        }
    }

    if cursor < tokens.len() && tokens[cursor].text == "order" {
        cursor += 1;
        expect_keyword(&tokens, &mut cursor, "by", input)?;
        if cursor >= tokens.len() {
            return Err(Err::Parse(query_error_pos(&tokens, cursor, input)));
        }
        let column = tokens[cursor].text.to_string();
        cursor += 1;
        let direction = if cursor < tokens.len() {
            let token = tokens[cursor].text.as_str();
            let direction = match token {
                "asc" => OrderDirection::Asc,
                "desc" => OrderDirection::Desc,
                _ => OrderDirection::Desc,
            };
            if matches!(token, "asc" | "desc") {
                cursor += 1;
            }
            direction
        } else {
            OrderDirection::Desc
        };
        plan.order = Some((column, direction));
    }

    if cursor < tokens.len() && tokens[cursor].text == "limit" {
        cursor += 1;
        if cursor >= tokens.len() {
            return Err(Err::Parse(query_error_pos(&tokens, cursor, input)));
        }
        let limit: usize = tokens[cursor]
            .text
            .parse()
            .map_err(|_| Err::Parse(query_error_pos(&tokens, cursor, input)))?;
        plan.limit = Some(limit);
    }

    Ok(plan)
}

fn expect_keyword(
    tokens: &[QueryToken],
    cursor: &mut usize,
    keyword: &str,
    input: &str,
) -> Result<(), Err> {
    if *cursor >= tokens.len() {
        return Err(Err::Parse(query_error_pos(tokens, *cursor, input)));
    }
    if tokens[*cursor].text == keyword {
        *cursor += 1;
        Ok(())
    } else {
        Err(Err::Parse(query_error_pos(tokens, *cursor, input)))
    }
}

fn lex_query(input: &str) -> Result<Vec<QueryToken>, Err> {
    let bytes = input.as_bytes();
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut current_start = 0usize;
    let mut index = 0usize;
    while index < bytes.len() {
        let b = bytes[index];
        let ch = b as char;
        if ch.is_whitespace() {
            if !current.is_empty() {
                tokens.push(QueryToken {
                    text: current.to_ascii_lowercase(),
                    start: current_start,
                });
                current.clear();
            }
            index += 1;
            continue;
        }
        match ch {
            ',' | '=' | '!' | '<' | '>' => {
                if !current.is_empty() {
                    tokens.push(QueryToken {
                        text: current.to_ascii_lowercase(),
                        start: current_start,
                    });
                    current.clear();
                }
                let op_start = index;
                let mut op = ch.to_string();
                if index + 1 < bytes.len() {
                    let next = bytes[index + 1] as char;
                    if (next == '=' || (next == '>' && ch == '<')) && ch != ',' {
                        op.push(next);
                        index += 1;
                    }
                }
                tokens.push(QueryToken {
                    text: op,
                    start: op_start,
                });
            }
            ';' => {
                if !current.is_empty() {
                    tokens.push(QueryToken {
                        text: current.to_ascii_lowercase(),
                        start: current_start,
                    });
                    current.clear();
                }
            }
            _ => {
                if current.is_empty() {
                    current_start = index;
                }
                current.push(ch);
            }
        }
        index += 1;
    }
    if !current.is_empty() {
        tokens.push(QueryToken {
            text: current.to_ascii_lowercase(),
            start: current_start,
        });
    }

    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_basic_query() {
        let plan = compile("SELECT score WHERE score > 0.5 ORDER BY score DESC LIMIT 3").unwrap();
        assert_eq!(plan.selects, vec!["score".to_string()]);
        assert_eq!(plan.filters.len(), 1);
        assert_eq!(plan.limit, Some(3));
        assert_eq!(
            plan.order,
            Some(("score".to_string(), OrderDirection::Desc))
        );
    }

    #[test]
    fn execute_filters_and_selects() {
        let plan = compile("SELECT score,weight WHERE score >= 0.5 ORDER BY weight ASC").unwrap();
        let rows = vec![
            BTreeMap::from([("score".to_string(), 0.7), ("weight".to_string(), 0.1)]),
            BTreeMap::from([("score".to_string(), 0.2), ("weight".to_string(), 0.5)]),
        ];
        let result = plan.execute(&rows);
        assert_eq!(result.len(), 1);
        assert!(result[0].contains_key("score"));
        assert!(result[0].contains_key("weight"));
    }
}
