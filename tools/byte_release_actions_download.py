"""Read one externally pinned Actions ciphertext with a scoped read-only token.

No credential acquisition, retries, decryption, dispatch or release promotion.
The caller owns a single deadline spanning metadata, transfer and extraction.
"""
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

import byte_release_actions_input as a
import byte_release_github_asset as g


def validate_redirect(value):
    if (type(value) is not str or len(value) > 8192
            or any(ord(char) < 33 or ord(char) > 126 for char in value)):
        raise g.AssetTransferError("invalid artifact redirect")
    parsed = urlsplit(value)
    path = unquote(parsed.path)
    # Only GitHub's documented artifact endpoints; never forward API credentials.
    host = parsed.netloc
    if (parsed.scheme != "https" or parsed.username is not None or parsed.password is not None
            or not (host == "results-receiver.actions.githubusercontent.com"
                    or re.fullmatch(r"[a-z0-9]{3,24}\.blob\.core\.windows\.net", host))
            or parsed.fragment or not path.startswith("/") or not path[1:]
            or "\\" in path or any(ord(char) < 33 or ord(char) > 126 for char in path)
            or any(part in ("", ".", "..") for part in path[1:].split("/"))):
        raise g.AssetTransferError("artifact redirect is outside the permitted endpoints")
    return value


def download(pin, token, archive, output, opener=None):
    a.validate_pin(pin)
    if (type(token) is not str or not 0 < len(token) <= 4096
            or any(ord(char) < 33 or ord(char) > 126 for char in token)):
        raise g.AssetAccessError("a scoped read-only workflow token is required")
    archive, output = Path(archive), Path(output)
    if archive.absolute() == output.absolute():
        raise ValueError("archive and ciphertext outputs must differ")
    for path in (archive, output):
        if path.exists() or path.is_symlink():
            raise FileExistsError("first Actions transfer output already exists")
        if any(parent.is_symlink() for parent in path.parents):
            raise ValueError("Actions output must not follow links")
    opener = opener or g.urllib.request.build_opener(g.urllib.request.ProxyHandler({}), g._NoRedirect(), g._RawResponses())
    root = f"{g.API_ROOT}/repos/{a.t.REPO}/actions"
    run_url = root + f"/runs/{pin['bridge']['run_id']}"
    artifact_url = root + f"/artifacts/{pin['artifact']['id']}"
    binary_url = artifact_url + "/zip"
    expected = {key: pin["artifact"][key] for key in ("bytes", "sha256")}
    try:
        run = g._metadata(opener, run_url, token)
        artifact = g._metadata(opener, artifact_url, token)
        a.verify_metadata(pin, run, artifact)
        redirect = None
        # The Actions API negotiates JSON even though its redirect serves a ZIP.
        with g._request(opener, binary_url, token) as response:
            g._response(response, binary_url, (200, 302))
            if response.status == 200:
                g._copy_response(response, binary_url, archive, expected)
            else:
                redirect = validate_redirect(response.headers.get("Location"))
        if redirect is not None:
            with g._request(opener, redirect, binary=True) as response:
                g._copy_response(response, redirect, archive, expected)
        # A completed run can still be rerun or an artifact deleted mid-transfer.
        run_after = g._metadata(opener, run_url, token)
        artifact_after = g._metadata(opener, artifact_url, token)
        a.verify_metadata(pin, run_after, artifact_after)
        return a.unpack_input(pin, run_after, artifact_after, archive, output)
    except g.AssetTransferError:
        raise
    except Exception:
        # URL/HTTP exceptions can include credentials or private response bodies.
        raise g.AssetTransferError("Actions input transfer failed; retain the first outcome; never retry") from None
