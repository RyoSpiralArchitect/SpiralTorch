# Licensing

SpiralTorch is licensed under the **AGPL-3.0-or-later** for open-source use, empowering the Rust/ML community to collaborate, fork, and innovate freely. This ensures that any network-based usage (e.g., SaaS deployments) requires source code disclosure, fostering transparency and collective growth—perfect for Z-space hypergrad experiments and WGPU-powered prototypes.

## Attribution is non-negotiable

The greatest risk to SpiralTorch is not code reuse—it is the erasure of its name and lineage. The AGPL allows wide remixing, but only when the origin remains visible. If "SpiralTorch" or its authorship disappear, innovation turns into appropriation.

### Why name erasure is unacceptable

When the SpiralTorch name is stripped away, the community loses:

- **Design provenance.** The ideas, architecture, and intent no longer point back to SpiralTorch.
- **Historical continuity.** Future contributors cannot trace improvements or credit prior work.
- **Ethical alignment.** Distributing without attribution violates AGPL §5 (Notices) and §13 (Remote Network Interaction) and undermines the spirit of free software.

### Common abuse patterns to watch for

| Pattern | What happens |
| --- | --- |
| ✂️ Name/LICENSE removal | Redistributors delete upstream names, histories, or license headers before publishing.
| 🧪 API/algorithm cloning | Core operators or interfaces are transplanted into a differently branded project with no credit.
| 🐙 Private, closed forks | An organization rebuilds SpiralTorch internally and hides the provenance.
| 🕵️ AGPL-violating SaaS | A hosted product embeds SpiralTorch but withholds both source code and attribution.

These tactics may be technically feasible, but they are **not compliant** with the AGPL or the SpiralTorch community’s expectations.

### Mandatory attribution requirements

Any derived work—source, binary, or network service—**must preserve**:

- The project name: **“SpiralTorch.”**
- Original authorship attribution: **“Ryo ∴ SpiralArchitect.”**
- An intact copy of the AGPL-3.0-or-later license, including §5 and §13 notices.
- SPDX headers or comment banners where provided (`/// SPDX-License-Identifier: AGPL-3.0-or-later`).

Unauthorized derivations that remove these identifiers are out of compliance and will be pursued as AGPL violations.

## Commercial License Option

For teams and enterprises seeking flexibility beyond AGPL obligations (e.g., proprietary SaaS, internal deployments without source sharing, or priority support), we offer a **Commercial License**. This allows unrestricted use, modification, and distribution while protecting your IP and avoiding viral copyleft requirements.

### Pricing Tiers (Annual Subscription, Billed Yearly)

- **Individual/Small Team (1-5 users)**: $500–$2,000  
  Ideal for indie devs prototyping Canvas Transformers or Z-space models. Includes basic support.
- **Startup/SME (5-50 users)**: $2,000–$10,000  
  Scaled by seats/devices (e.g., $0.50–$2/user/month). Perfect for distributed training with GoldenRetriever.
- **Enterprise (50+ users, Custom Deployments)**: $10,000+  
  Tailored for large-scale ops (e.g., per-instance fees at $2–$5). Includes indemnity, custom Z-space tuning, and 24/7 support.

> **First-year discount:** 50% off for early adopters! All tiers include access to premium features like advanced telemetry dashboards and fractional FFT extensions (coming soon).

## Identity-preserving practices

To make name erasure even harder, we actively bake SpiralTorch’s identity into the stack:

- Domain-specific vocabulary such as **Z-space**, **SpinoTensorVector**, and **GoldenRetriever** anchors the project’s concepts.
- Namespaces like `spiraltorch.compat.torch` ensure downstream forks visibly inherit the brand.
- Signed metadata files (e.g., `.torchmeta.json`) can be embedded in artifacts to verify provenance.
- Whitepapers and design notes document the philosophy so the community can recognize SpiralTorch-descended ideas.

You are encouraged to adopt the same practices in derivatives to keep the project’s lineage intact.

## Contact

**Ready to get started?** Email us at [kishkavsesvit@icloud.com] for a custom quote, trial license, or CLA (Contributor License Agreement) details. We're excited to partner with you on Rust-powered ML breakthroughs!

## Why Dual?

AGPL keeps the community thriving (shoutout to our 1,500+ shadow cloners!), while Commercial unlocks enterprise-scale innovation without the hassle.
