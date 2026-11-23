# Signed synthetic response: first attempt stopped before any logits

The [frozen four-variant protocol](LINE_SIGNED_RESPONSE_PROTOCOL.md)
(`03f5629`) was attempted once with the original installed CPU extension.
The worker exited during origin validation, before opening a journal and
before the first forward. **0/15 prompt readouts, 0 updates, no concentration
result.** No retry. This is separate from the successful real
[GPT-2 gradient read-back](../parameter-vjp/RESULTS.md).

The error in the harness (`1794481`) was comparing the SHA-256 of the
exported LoRA JSON with a Rust artifact ID. Rust hashes
`b"spiraltorch.native_lora.v1\0"` followed by the canonical record bytes;
the two digests are supposed to differ. The committed bridge confirms the
existing LoRA ID ends in `6337`, whereas the SHA-256 of its JSON bytes
ends in `863f`. The corrected check verifies both archived values independently,
and tests reject changed identity, LoRA bytes or geometry bytes. Only
weight-free checks of the correction were run. The first failure is retained
(log SHA-256 `04aa159f8b912e2f63873f902e431ebc0bec5965c6685a2e6a1eed0d14ee3733`,
receipt `148879f2ce976c2799ce5022160fbb0c77622f838b99b1e2cac9f086f782f884`).

Native geometry can attach to attention, MLP or block outputs by itself.
With an active LoRA the current Rust contract instead requires a single
colocated geometry at that LoRA's attention output. No placement was changed,
no guard was loosened, and no placement comparison or training was run.
