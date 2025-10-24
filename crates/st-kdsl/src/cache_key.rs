// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::collections::{BTreeMap, HashMap};

use serde::Serialize;

#[derive(Serialize)]
struct CacheFingerprint<'a, Z> {
    rustc: &'a str,
    target: &'a str,
    backend: &'a str,
    precision: &'a str,
    flags: &'a [String],
    defs: BTreeMap<String, String>,
    zframe: &'a Z,
}

pub fn stable_cache_key<Z: Serialize>(
    rustc: &str,
    target: &str,
    backend: &str,
    precision: &str,
    mut flags: Vec<String>,
    defs_hm: &HashMap<String, String>,
    zframe: &Z,
) -> String {
    flags.sort();
    let defs = defs_hm
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect::<BTreeMap<_, _>>();
    let fp = CacheFingerprint {
        rustc,
        target,
        backend,
        precision,
        flags: &flags,
        defs,
        zframe,
    };
    let bytes = serde_json::to_vec(&fp).expect("fingerprint serialize");
    blake3::hash(&bytes).to_hex().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Serialize;

    #[derive(Serialize)]
    struct Z {
        a: u32,
        b: u32,
    }

    #[test]
    fn stable_key_order_independent() {
        let mut d1 = HashMap::new();
        d1.insert("X".into(), "1".into());
        d1.insert("A".into(), "2".into());
        let mut d2 = HashMap::new();
        d2.insert("A".into(), "2".into());
        d2.insert("X".into(), "1".into());
        let k1 = stable_cache_key(
            "rustc1",
            "x86",
            "wgpu",
            "f32",
            vec!["-O".into(), "--foo".into()],
            &d1,
            &Z { a: 1, b: 2 },
        );
        let k2 = stable_cache_key(
            "rustc1",
            "x86",
            "wgpu",
            "f32",
            vec!["--foo".into(), "-O".into()],
            &d2,
            &Z { a: 1, b: 2 },
        );
        assert_eq!(k1, k2);
    }
}
