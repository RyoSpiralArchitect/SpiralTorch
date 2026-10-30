"""GET-only reader for one pinned, unpublished GitHub release asset.

No creation, publication, token acquisition, permission escalation or retry is
implemented here. The caller owns the shared deadline and token scope: read-only
for a final GPU consumer, or the explicitly approved CPU-only bridge exception.
"""
from pathlib import Path
import re
import urllib.request
from urllib.parse import urlsplit, unquote

import byte_release_transport_v2 as t

API_ROOT = "https://api.github.com"
API_VERSION = "2026-03-10"
ASSET_HOST = "release-assets.githubusercontent.com"
USER_AGENT = "SpiralTorch-byte-release-transport"


class AssetTransferError(ValueError):
    pass


class AssetAccessError(AssetTransferError):
    pass


def validate_locator(value):
    t.exact_keys(value, ("repository", "repository_id", "release_id", "asset_id", "asset_name", "release_tag"))
    if value["repository"] != t.REPO:
        raise ValueError("GitHub asset repository differs")
    for key in ("repository_id", "release_id", "asset_id"):
        if type(value[key]) is not int or not 0 < value[key] <= 2**63 - 1:
            raise ValueError("invalid GitHub object ID")
    for key in ("asset_name", "release_tag"):
        name = value[key]
        if (type(name) is not str or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,179}", name)
                or ".." in name or name.endswith((".", ".lock"))):
            raise ValueError("invalid pinned GitHub object name")
    if not value["asset_name"].endswith(".age"):
        raise ValueError("GitHub input must be an encrypted asset")
    return value


class _RawResponses(urllib.request.HTTPErrorProcessor):
    def http_response(self, request, response):
        return response

    https_response = http_response


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise AssetTransferError("automatic redirects are forbidden")


def _request(opener, url, token=None, binary=False):
    headers = {"Accept": "application/octet-stream" if binary else "application/vnd.github+json",
               "Accept-Encoding": "identity", "User-Agent": USER_AGENT}
    if token is not None:
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.netloc != "api.github.com":
            raise AssetTransferError("authentication cannot leave the API host")
        headers.update(Authorization="Bearer " + token, **{"X-GitHub-Api-Version": API_VERSION})
    return opener.open(urllib.request.Request(url, headers=headers, method="GET"), timeout=30)


def _response(response, url, allowed=(200,)):
    if response.status in (401, 403, 404):
        raise AssetAccessError("GitHub read access is unavailable; do not change credentials or retry")
    if (response.status not in allowed or response.geturl() != url
            or response.headers.get("Content-Encoding", "identity").lower() != "identity"):
        raise AssetTransferError("unexpected GitHub transfer response")


def _metadata(opener, url, token):
    with _request(opener, url, token) as response:
        _response(response, url)
        media = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if media not in ("application/json", "application/vnd.github+json"):
            raise AssetTransferError("GitHub metadata is not JSON")
        length = response.headers.get("Content-Length")
        if length is not None and (not re.fullmatch(r"[1-9][0-9]{0,5}", length) or int(length) > t.MAX_RECORD):
            raise AssetTransferError("GitHub metadata exceeds its bound")
        raw = response.read(t.MAX_RECORD + 1)
        if length is not None and len(raw) != int(length):
            raise AssetTransferError("GitHub metadata length differs")
        value = t.parse_record(raw)
        if type(value) is not dict:
            raise AssetTransferError("GitHub metadata is not a record")
        return value


def _asset_identity(row, locator, expected):
    url = f"{API_ROOT}/repos/{t.REPO}/releases/assets/{locator['asset_id']}"
    if (type(row) is not dict or type(row.get("id")) is not int or row["id"] != locator["asset_id"]
            or row.get("name") != locator["asset_name"] or row.get("state") != "uploaded"
            or type(row.get("size")) is not int or row["size"] != expected["bytes"]
            or row.get("digest") != "sha256:" + expected["sha256"] or row.get("url") != url):
        raise AssetTransferError("GitHub asset identity differs")


def _release_identity(row, locator, expected):
    url = f"{API_ROOT}/repos/{t.REPO}/releases/{locator['release_id']}"
    if (type(row.get("id")) is not int or row["id"] != locator["release_id"]
            or row.get("draft") is not True or "published_at" not in row or row["published_at"] is not None
            or row.get("tag_name") != locator["release_tag"] or row.get("url") != url
            or type(row.get("assets")) is not list):
        raise AssetTransferError("release is not the pinned unpublished draft")
    seen, matches = set(), []
    for asset in row["assets"]:
        identifier = asset.get("id") if type(asset) is dict else None
        if type(identifier) is not int or identifier <= 0 or identifier in seen:
            raise AssetTransferError("ambiguous GitHub release asset roster")
        seen.add(identifier)
        if identifier == locator["asset_id"]:
            matches.append(asset)
    if len(matches) != 1:
        raise AssetTransferError("pinned asset is absent from the draft")
    _asset_identity(matches[0], locator, expected)


def validate_redirect(value):
    if (type(value) is not str or len(value) > 8192
            or any(ord(char) < 33 or ord(char) > 126 for char in value)):
        raise AssetTransferError("invalid release asset redirect")
    parsed = urlsplit(value)
    decoded_path = unquote(parsed.path)
    if (parsed.scheme != "https" or parsed.netloc != ASSET_HOST
            or not parsed.path.startswith("/") or not parsed.path[1:]
            or parsed.fragment or parsed.username is not None or parsed.password is not None
            or "\\" in decoded_path or any(ord(char) < 33 or ord(char) > 126 for char in decoded_path)
            or any(part in ("", ".", "..") for part in decoded_path[1:].split("/"))):
        raise AssetTransferError("redirect is outside the fixed release asset host")
    return value


def _copy_response(response, url, output, expected):
    _response(response, url)
    length = response.headers.get("Content-Length")
    if length is not None and length != str(expected["bytes"]):
        raise AssetTransferError("GitHub ciphertext length differs")
    return t.copy_verified(response, output, expected)


def download(descriptor, token, output, opener=None):
    t.validate_descriptor(descriptor)
    if descriptor["schema"] != t.GITHUB_DESCRIPTOR_SCHEMA:
        raise ValueError("GitHub reader requires the GitHub descriptor")
    if (type(token) is not str or not 0 < len(token) <= 4096
            or any(ord(char) < 33 or ord(char) > 126 for char in token)):
        raise AssetAccessError("a scoped workflow token is required")
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("first transfer output already exists")
    if any(parent.is_symlink() for parent in output.parents):
        raise ValueError("transfer output must not follow a link")
    locator, expected = descriptor["object"], descriptor["ciphertext"]
    opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect(), _RawResponses())
    repository_url = f"{API_ROOT}/repos/{t.REPO}"
    release_url = repository_url + f"/releases/{locator['release_id']}"
    asset_url = repository_url + f"/releases/assets/{locator['asset_id']}"
    try:
        repository = _metadata(opener, repository_url, token)
        if (type(repository.get("id")) is not int or repository["id"] != locator["repository_id"]
                or repository.get("full_name") != t.REPO or repository.get("archived") is not False):
            raise AssetTransferError("GitHub repository identity differs")
        _release_identity(_metadata(opener, release_url, token), locator, expected)
        _asset_identity(_metadata(opener, asset_url, token), locator, expected)
        redirect = None
        with _request(opener, asset_url, token, binary=True) as response:
            _response(response, asset_url, (200, 302))
            if response.status == 200:
                _copy_response(response, asset_url, output, expected)
            else:
                redirect = validate_redirect(response.headers.get("Location"))
        if redirect is not None:
            # A fresh unauthenticated request, not generic redirect following.
            with _request(opener, redirect, binary=True) as response:
                _copy_response(response, redirect, output, expected)
        _release_identity(_metadata(opener, release_url, token), locator, expected)
        return dict(status="verified_ciphertext", repository_id=locator["repository_id"],
                    release_id=locator["release_id"], asset_id=locator["asset_id"],
                    bytes=expected["bytes"], sha256=expected["sha256"],
                    path="single_cdn_redirect" if redirect else "direct_api", release_ready=False)
    except AssetTransferError:
        raise
    except Exception:
        # HTTP/URL errors may contain credentials, signed URLs or response bodies.
        raise AssetTransferError("GitHub asset transfer failed; retain the first outcome; never retry") from None
