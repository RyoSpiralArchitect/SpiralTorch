"""Render a controller workflow; never edit or dispatch the frozen workflow.

The entire rendered document is compared at controller admission. Unchanged
execution steps are copied from the pinned source, not rediscovered by searching
for reassuring fragments of YAML in an otherwise arbitrary workflow.
"""

EXPRESSION = "$" + "{{"


def expression(text):
    return text.replace("@{", EXPRESSION)


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError("frozen workflow structure differs")
    return text.replace(before, after, 1)


def replace_step(text, name, replacement):
    marker = "      - name: " + name + "\n"
    if text.count(marker) != 1:
        raise ValueError("frozen workflow step differs")
    start = text.index(marker)
    end = text.find("\n      - ", start + len(marker))
    end = len(text) if end < 0 else end + 1
    return text[:start] + expression(replacement) + text[end:]


SOURCE_CHECKOUT = expression("""      - uses: actions/checkout@fbc6f3992d24b796d5a048ff273f7fcc4a7b6c09 # v5
        with:
          ref: @{ inputs.source_commit }}
          path: source
          persist-credentials: false
""")

ADMIT = """      - name: Reject repeated or unbound invocations before accessing input secrets
        working-directory: controller
        env:
          GH_TOKEN: @{ github.token }}
        run: |
          python -I -S -B tools/byte_release_hosted_v2.py admit-run --source ../source \\
            --output "$EVIDENCE_ROOT/hosted-context.json" > "$EVIDENCE_ROOT/admission.log" 2>&1
"""

PREPARE = """      - name: Fetch and decrypt the pinned object without exposing credentials to execution
        working-directory: controller
        env:
          INPUT_IDENTITY: @{ secrets[format('ST_BYTE_RELEASE_CANDIDATE_{0}_INPUT_IDENTITY', inputs.candidate)] }}
          INPUT_OBJECT_URL: @{ secrets[format('ST_BYTE_RELEASE_CANDIDATE_{0}_INPUT_URL', inputs.candidate)] }}
        run: |
          umask 077
          test -n "$INPUT_IDENTITY"
          test -n "$INPUT_OBJECT_URL"
          key="$TRANSPORT_ROOT/input-key.txt"
          trap 'rm -f "$key"' EXIT
          printf '%s\\n' "$INPUT_IDENTITY" > "$key"
          unset INPUT_IDENTITY
          {
            python -I -S -B tools/byte_release_hosted_v2.py fetch-input \\
              --output "$TRANSPORT_ROOT/input.zip.age"
            unset INPUT_OBJECT_URL
            "$TRANSPORT_ROOT/age/age" --decrypt --identity "$key" \\
              --output "$TRANSPORT_ROOT/input.zip" "$TRANSPORT_ROOT/input.zip.age"
            rm -f "$key"
            python -I -S -B tools/byte_release_hosted_v2.py prepare-input --source ../source \\
              --input "$TRANSPORT_ROOT/input.zip" --output "$TRANSPORT_ROOT/input"
          } > "$EVIDENCE_ROOT/input-preflight.log" 2>&1
"""

PACK = """      - name: Encrypt the first outcome including incomplete evidence
        if: @{ always() && env.TRANSPORT_ROOT != '' }}
        working-directory: controller
        run: |
          umask 077
          test -x "$TRANSPORT_ROOT/age/age"
          python -I -S -B tools/byte_release_hosted_v2.py pack-result \\
            --evidence "$EVIDENCE_ROOT" --input "$TRANSPORT_ROOT/input" --candidate "$CANDIDATE" \\
            --output "$TRANSPORT_ROOT/result.zip" --admission-sha256 "$ADMISSION_SHA256" \\
            > "$TRANSPORT_ROOT/packing.json"
          mkdir -m 700 "$TRANSPORT_ROOT/public"
          "$TRANSPORT_ROOT/age/age" --encrypt --recipient "$RESULT_RECIPIENT" \\
            --output "$TRANSPORT_ROOT/public/result.zip.age" "$TRANSPORT_ROOT/result.zip"
          cd "$TRANSPORT_ROOT/public"
          shasum -a 256 result.zip.age > SHA256SUMS
"""


def github_prepare():
    text = replace_once(PREPARE, "Fetch and decrypt the pinned object without exposing credentials to execution",
                        "Fetch and decrypt the pinned unpublished GitHub asset")
    text = replace_once(text,
        "          INPUT_OBJECT_URL: @{ secrets[format('ST_BYTE_RELEASE_CANDIDATE_{0}_INPUT_URL', inputs.candidate)] }}\n",
        "          GH_TOKEN: @{ github.token }}\n")
    text = replace_once(text, '          test -n "$INPUT_OBJECT_URL"\n', '          test -n "$GH_TOKEN"\n')
    return replace_once(text, "            unset INPUT_OBJECT_URL\n", "            unset GH_TOKEN GITHUB_TOKEN\n")


def render(baseline, *, github_asset=False):
    if type(github_asset) is not bool:
        raise ValueError("workflow transport selector must be boolean")
    text = replace_once(baseline, "Digest of the single committed age ciphertext",
                        "Digest of the separate pinned age ciphertext")
    text = replace_once(text, "      result_recipient:\n", """      transport_amendment:
        description: "Externally approved transport-only amendment JSON; never a credential"
        type: string
        required: true
      transport_amendment_sha256:
        description: "Canonical digest of the externally approved amendment"
        type: string
        required: true
      result_recipient:
""")
    old_env = expression("      RESULT_RECIPIENT: @{ inputs.result_recipient }}\n")
    text = replace_once(text, old_env, old_env + expression(
        "      TRANSPORT_AMENDMENT: @{ inputs.transport_amendment }}\n"
        "      TRANSPORT_AMENDMENT_SHA256: @{ inputs.transport_amendment_sha256 }}\n"))
    text = replace_once(text, SOURCE_CHECKOUT, "")
    marker = "      - name: Reject repeated or unbound invocations before accessing input secrets\n"
    text = replace_once(text, marker, SOURCE_CHECKOUT + marker)
    text = replace_step(text, "Reject repeated or unbound invocations before accessing input secrets", ADMIT)
    text = replace_step(text, "Decrypt the packet without exposing the input identity to execution",
                        github_prepare() if github_asset else PREPARE)
    return replace_step(text, "Encrypt the first outcome including incomplete evidence", PACK)
