#!/usr/bin/env python3
"""Unit coverage for the PyPI/GitHub Release digest verifier."""

from __future__ import annotations

import importlib.util
from contextlib import ExitStack
import io
import json
from pathlib import Path
import ssl
import sys
import tempfile
import unittest
from unittest import mock
from urllib.error import HTTPError, URLError


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "security" / "verify_pypi_release.py"
SPEC = importlib.util.spec_from_file_location("verify_pypi_release", SCRIPT)
assert SPEC and SPEC.loader
verify_pypi_release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify_pypi_release)


class VerifyPyPIReleaseTests(unittest.TestCase):
    def test_parse_sha256_lines_normalizes_case_and_ignores_blank_lines(self) -> None:
        digest = "A" * 64
        parsed = verify_pypi_release.parse_sha256_lines(
            f"\n{digest}  spiraltorch-0.4.10.whl\n"
        )

        self.assertEqual(parsed, {"spiraltorch-0.4.10.whl": "a" * 64})

    def test_parse_sha256_lines_rejects_duplicates(self) -> None:
        digest = "b" * 64
        text = f"{digest}  wheel.whl\n{digest}  wheel.whl\n"

        with self.assertRaisesRegex(verify_pypi_release.VerifyError, "Duplicate"):
            verify_pypi_release.parse_sha256_lines(text)

    def test_compare_digests_reports_missing_extra_and_mismatched(self) -> None:
        release = {"linux.whl": "1" * 64, "mac.whl": "2" * 64}
        pypi = {"linux.whl": "3" * 64, "win.whl": "4" * 64}

        with self.assertRaisesRegex(verify_pypi_release.VerifyError, "mismatched"):
            verify_pypi_release.compare_digests(release, pypi)

    def test_pypi_wheel_digests_filters_non_wheels(self) -> None:
        payload = {
            "urls": [
                {
                    "filename": "spiraltorch-0.4.10.whl",
                    "digests": {"sha256": "ABCDEF" + "0" * 58},
                },
                {
                    "filename": "spiraltorch-0.4.10.tar.gz",
                    "digests": {"sha256": "f" * 64},
                },
            ],
        }

        with mock.patch.object(
            verify_pypi_release, "download_json", return_value=payload
        ):
            self.assertEqual(
                verify_pypi_release.pypi_wheel_digests("spiraltorch", "0.4.10"),
                {"spiraltorch-0.4.10.whl": "abcdef" + "0" * 58},
            )

    def test_wait_for_pypi_wheels_polls_until_expected_count(self) -> None:
        calls = [
            {"linux.whl": "1" * 64},
            {"linux.whl": "1" * 64, "mac.whl": "2" * 64},
        ]

        with mock.patch.object(
            verify_pypi_release, "pypi_wheel_digests", side_effect=calls
        ):
            with mock.patch.object(verify_pypi_release.time, "sleep") as sleep:
                result = verify_pypi_release.wait_for_pypi_wheels(
                    "spiraltorch",
                    "0.4.10",
                    expected_wheels=2,
                    timeout=30,
                    poll_interval=0.01,
                )

        self.assertEqual(result, {"linux.whl": "1" * 64, "mac.whl": "2" * 64})
        sleep.assert_called_once_with(0.01)

    def test_wait_for_pypi_wheels_polls_until_latest_version(self) -> None:
        wheels = {"spiraltorch-0.4.11.whl": "1" * 64}
        with mock.patch.object(
            verify_pypi_release, "pypi_wheel_digests", return_value=wheels
        ):
            with mock.patch.object(
                verify_pypi_release,
                "pypi_latest_version",
                side_effect=["0.4.10", "0.4.11"],
            ):
                with mock.patch.object(verify_pypi_release.time, "sleep") as sleep:
                    result = verify_pypi_release.wait_for_pypi_wheels(
                        "spiraltorch",
                        "0.4.11",
                        expected_wheels=1,
                        require_latest=True,
                        timeout=30,
                        poll_interval=0.01,
                    )

        self.assertEqual(result, wheels)
        sleep.assert_called_once_with(0.01)

    def test_main_passes_require_latest_to_publication_wait(self) -> None:
        wheels = {"spiraltorch-0.4.11.whl": "1" * 64}
        argv = [
            "verify_pypi_release.py",
            "--version",
            "0.4.11",
            "--expected-wheels",
            "1",
            "--require-latest",
        ]

        with mock.patch.object(sys, "argv", argv):
            with mock.patch.object(
                verify_pypi_release,
                "github_release_wheel_digests",
                return_value=wheels,
            ):
                with mock.patch.object(
                    verify_pypi_release,
                    "wait_for_pypi_wheels",
                    return_value=wheels,
                ) as wait:
                    self.assertEqual(verify_pypi_release.main(), 0)

        wait.assert_called_once_with(
            "spiraltorch",
            "0.4.11",
            expected_wheels=1,
            require_latest=True,
            require_simple_index=False,
            expected_digests=wheels,
            timeout=240.0,
            poll_interval=10.0,
        )


class PublicationReadinessTests(unittest.TestCase):
    def setUp(self):
        self.api = verify_pypi_release
        self.expected = {"linux.whl": "1" * 64, "mac.whl": "2" * 64}
        self.now = 0.0
        self.sleeps = []
        self.reads = {}

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds

    def wait(self, *, wheels=None, simple=None, latest=None, **options):
        arguments = dict(
            expected_wheels=2,
            expected_digests=self.expected,
            require_latest=True,
            require_simple_index=True,
            timeout=10.0,
            poll_interval=1.0,
        )
        arguments.update(options)
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    self.api.time, "monotonic", side_effect=lambda: self.now
                )
            )
            stack.enter_context(
                mock.patch.object(self.api.time, "sleep", side_effect=self.sleep)
            )
            for name, responses, default in [
                ("pypi_wheel_digests", wheels, self.expected),
                ("pypi_latest_version", latest, "0.4.24"),
                ("pypi_simple_wheel_digests", simple, self.expected),
            ]:
                self.reads[name] = stack.enter_context(
                    mock.patch.object(
                        self.api,
                        name,
                        side_effect=responses,
                        return_value=default,
                    )
                )
            return self.api.wait_for_pypi_wheels("spiraltorch", "0.4.24", **arguments)

    def test_json_ready_does_not_bypass_stale_or_partial_simple_index(self):
        self.assertEqual(
            self.wait(simple=[{}, {"linux.whl": "1" * 64}, self.expected]),
            self.expected,
        )
        self.assertEqual(self.sleeps, [1.0, 1.0])
        self.assertEqual(self.reads["pypi_simple_wheel_digests"].call_count, 3)

    def test_initial_404_or_temporary_read_failure_is_retried(self):
        self.assertEqual(
            self.wait(wheels=[self.api.PublicationPending("HTTP 404"), self.expected]),
            self.expected,
        )
        self.assertEqual(self.sleeps, [1.0])

    def test_simple_index_temporary_failure_is_retried(self):
        self.assertEqual(
            self.wait(simple=[self.api.PublicationPending("HTTP 503"), self.expected]),
            self.expected,
        )
        self.assertEqual(self.sleeps, [1.0])

    def test_latest_metadata_failure_and_delay_are_retried(self):
        self.assertEqual(
            self.wait(
                latest=[self.api.PublicationPending("HTTP 429"), "0.4.23", "0.4.24"]
            ),
            self.expected,
        )
        self.assertEqual(self.sleeps, [1.0, 1.0])
        self.assertEqual(self.reads["pypi_simple_wheel_digests"].call_count, 1)

    def test_malformed_or_integrity_failure_is_not_retried(self):
        for surface in ["wheels", "simple", "latest"]:
            with self.subTest(surface=surface):
                with self.assertRaisesRegex(self.api.VerifyError, "terminal"):
                    self.wait(
                        **{surface: self.api.VerifyError("terminal integrity failure")}
                    )
                self.assertEqual(self.sleeps, [])

    def test_partial_release_cannot_hide_a_hash_mismatch_or_extra_wheel(self):
        for wheels in [{"linux.whl": "9" * 64}, {"extra.whl": "1" * 64}]:
            with self.subTest(wheels=wheels):
                with self.assertRaisesRegex(
                    self.api.VerifyError, "Unexpected or mismatched"
                ):
                    self.wait(wheels=[wheels])
                self.assertEqual(self.sleeps, [])
                self.reads["pypi_latest_version"].assert_not_called()

    def test_persistent_unavailability_expires_without_a_final_out_of_budget_request(
        self,
    ):
        with self.assertRaisesRegex(
            self.api.VerifyError, "Timed out.*HTTP 404|Timed out.*budget"
        ):
            self.wait(
                wheels=self.api.PublicationPending("HTTP 404"),
                timeout=2.5,
                poll_interval=2,
            )
        self.assertEqual(self.sleeps, [2, 0.5])
        self.assertEqual(self.reads["pypi_wheel_digests"].call_count, 2)
        self.assertEqual(self.now, 2.5)

    def test_remaining_budget_caps_each_network_timeout(self):
        def read_wheels(*args, **kwargs):
            self.assertEqual(kwargs["timeout"], 5)
            self.now = 4
            return self.expected

        self.wait(wheels=read_wheels, timeout=5)
        self.assertEqual(
            self.reads["pypi_latest_version"].call_args.kwargs["timeout"], 1
        )
        self.assertEqual(
            self.reads["pypi_simple_wheel_digests"].call_args.kwargs["timeout"], 1
        )

    def test_slow_response_cannot_succeed_after_deadline(self):
        def read_simple(*args, **kwargs):
            self.now = 11
            return self.expected

        with self.assertRaisesRegex(self.api.VerifyError, "Timed out"):
            self.wait(simple=read_simple)
        self.assertEqual(self.sleeps, [])

    def test_zero_wheels_without_expected_count_is_not_success(self):
        with self.assertRaisesRegex(self.api.VerifyError, "Timed out"):
            self.wait(
                wheels=lambda *a, **k: {},
                expected_wheels=None,
                expected_digests=None,
                require_simple_index=False,
                require_latest=False,
                timeout=1,
            )
        self.assertEqual(self.now, 1)

    def test_invalid_budgets_fail_before_network(self):
        for key in ["timeout", "poll_interval"]:
            for value in [0, -1, float("nan"), float("inf")]:
                with self.subTest(key=key, value=value):
                    with self.assertRaisesRegex(
                        self.api.VerifyError, "positive finite"
                    ):
                        self.wait(**{key: value})
                    self.reads["pypi_wheel_digests"].assert_not_called()

    def test_invalid_expected_count_fails_before_network(self):
        for value in [0, -1, True, 2.5]:
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    self.api.VerifyError, "expected_wheels must be positive"
                ):
                    self.wait(expected_wheels=value)
                self.reads["pypi_wheel_digests"].assert_not_called()

    def test_simple_gate_cannot_run_without_reference_hashes(self):
        with self.assertRaisesRegex(self.api.VerifyError, "requires the expected"):
            self.wait(expected_digests=None)

    def test_empty_or_wrong_sized_reference_is_rejected(self):
        for expected in [{}, {"linux.whl": "1" * 64}]:
            with self.subTest(expected=expected):
                with self.assertRaises(self.api.VerifyError):
                    self.wait(expected_digests=expected)

    def test_bad_reference_digest_is_rejected(self):
        with self.assertRaisesRegex(self.api.VerifyError, "Malformed sha256"):
            self.wait(expected_digests={"linux.whl": "oops", "mac.whl": "2" * 64})

    def test_optional_simple_and_latest_gates_stay_optional(self):
        self.wait(require_simple_index=False, require_latest=False)
        self.reads["pypi_simple_wheel_digests"].assert_not_called()
        self.reads["pypi_latest_version"].assert_not_called()

    def simple_payload(self, **overrides):
        payload = {
            "meta": {"api-version": "1.4"},
            "name": "spiraltorch",
            "files": [
                {"filename": name, "hashes": {"sha256": digest}, "yanked": False}
                for name, digest in self.expected.items()
            ],
        }
        payload.update(overrides)
        return payload

    def test_simple_parser_normalizes_names_and_ignores_other_versions(self):
        payload = self.simple_payload(name="spiral-torch")
        payload["files"].append({"filename": "old.whl", "hashes": {}, "yanked": True})
        with mock.patch.object(
            self.api, "download_json", return_value=payload
        ) as download:
            self.assertEqual(
                self.api.pypi_simple_wheel_digests("Spiral.Torch", self.expected),
                self.expected,
            )
        download.assert_called_once_with(
            "https://pypi.org/simple/spiral-torch/", simple=True, timeout=30.0
        )

    def test_simple_parser_requires_reference_hash_and_rejects_yanked_wheels(self):
        for field, values in [
            ("hashes", [{}, [], {"sha256": "bad"}, {"sha256": "0" * 64}]),
            ("yanked", [True, "withdrawn", "", None, 0]),
        ]:
            for value in values:
                with self.subTest(field=field, value=value):
                    payload = self.simple_payload()
                    payload["files"][0][field] = value
                    with mock.patch.object(
                        self.api, "download_json", return_value=payload
                    ):
                        with self.assertRaises(self.api.VerifyError):
                            self.api.pypi_simple_wheel_digests(
                                "spiraltorch", self.expected
                            )

    def test_simple_parser_rejects_duplicate_or_malformed_file_entries(self):
        for files in [
            None,
            [None],
            [{"filename": None}],
            self.simple_payload()["files"] * 2,
        ]:
            with self.subTest(files=files):
                with mock.patch.object(
                    self.api,
                    "download_json",
                    return_value=self.simple_payload(files=files),
                ):
                    with self.assertRaises(self.api.VerifyError):
                        self.api.pypi_simple_wheel_digests("spiraltorch", self.expected)

    def test_simple_parser_rejects_wrong_project_or_api_major_version(self):
        for values in [
            {"name": "other"},
            {"meta": None},
            {"meta": {"api-version": "2.0"}},
            {"meta": {"api-version": "invalid"}},
        ]:
            with self.subTest(values=values):
                with mock.patch.object(
                    self.api,
                    "download_json",
                    return_value=self.simple_payload(**values),
                ):
                    with self.assertRaises(self.api.VerifyError):
                        self.api.pypi_simple_wheel_digests("spiraltorch", self.expected)

    def test_json_parser_rejects_bad_digests(self):
        for value in [None, "", "x" * 64, "a" * 63]:
            with self.subTest(value=value):
                payload = {
                    "urls": [{"filename": "linux.whl", "digests": {"sha256": value}}]
                }
                with mock.patch.object(self.api, "download_json", return_value=payload):
                    with self.assertRaisesRegex(
                        self.api.VerifyError, "Malformed sha256"
                    ):
                        self.api.pypi_wheel_digests("spiraltorch", "0.4.24")

    def test_package_validation_prevents_path_injection(self):
        for name in ["", "../other", "other/name", "other?query", "bad\nname", "-bad"]:
            with self.subTest(name=name):
                with self.assertRaises(self.api.VerifyError):
                    self.api.normalize_package(name)

    def test_release_and_latest_malformed_shapes_are_terminal(self):
        for value in [{}, {"urls": [None]}, {"urls": [{"filename": None}]}]:
            with self.subTest(value=value):
                with mock.patch.object(self.api, "download_json", return_value=value):
                    with self.assertRaises(self.api.VerifyError):
                        self.api.pypi_wheel_digests("spiraltorch", "0.4.24")
        for payload in [{}, {"info": []}, {"info": {}}, {"info": {"version": 24}}]:
            with self.subTest(payload=payload):
                with mock.patch.object(self.api, "download_json", return_value=payload):
                    with self.assertRaises(self.api.VerifyError):
                        self.api.pypi_latest_version("spiraltorch")
        with mock.patch.object(
            self.api,
            "download_json",
            return_value=self.simple_payload(meta={"api-version": 1.4}),
        ):
            with self.assertRaises(self.api.VerifyError):
                self.api.pypi_simple_wheel_digests("spiraltorch", self.expected)

    def test_http_transient_and_terminal_errors_are_distinct(self):
        for status in [400, 401, 403, 404, 408, 429, 500, 502, 503, 504]:
            with self.subTest(status=status):
                error = HTTPError(
                    "https://pypi.org/simple/spiraltorch/",
                    status,
                    "test",
                    {},
                    io.BytesIO(),
                )
                with mock.patch.object(self.api, "urlopen", side_effect=error):
                    with self.assertRaises(self.api.VerifyError) as raised:
                        self.api.download_json("https://pypi.org/simple/spiraltorch/")
                self.assertEqual(
                    isinstance(raised.exception, self.api.PublicationPending),
                    status in {404, 408, 429} or status >= 500,
                )

    def test_network_timeout_retries_but_tls_validation_does_not(self):
        for error, transient in [
            (TimeoutError(), True),
            (URLError("connection reset"), True),
            (URLError(ssl.SSLCertVerificationError("certificate rejected")), False),
        ]:
            with self.subTest(error=error):
                with mock.patch.object(self.api, "urlopen", side_effect=error):
                    with self.assertRaises(self.api.VerifyError) as raised:
                        self.api.download_json("https://pypi.org/simple/spiraltorch/")
                self.assertEqual(
                    isinstance(raised.exception, self.api.PublicationPending), transient
                )

    def test_download_checks_media_type_json_shape_and_request_budget(self):
        payload = self.simple_payload()
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.headers = {
            "Content-Type": self.api.SIMPLE_JSON_TYPE + "; charset=UTF-8"
        }
        response.read.return_value = json.dumps(payload).encode()
        with mock.patch.object(self.api, "urlopen", return_value=response) as open_url:
            self.assertEqual(
                self.api.download_json(
                    "https://pypi.org/simple/spiraltorch/", simple=True, timeout=0.5
                ),
                payload,
            )
        request = open_url.call_args.args[0]
        self.assertEqual(request.get_header("Accept"), self.api.SIMPLE_ACCEPT)
        self.assertEqual(request.get_header("Cache-control"), "max-age=0")
        self.assertEqual(open_url.call_args.kwargs["timeout"], 0.5)
        for content_type, body in [
            ("text/html", b"{}"),
            (self.api.SIMPLE_JSON_TYPE, b"[1]"),
            (self.api.SIMPLE_JSON_TYPE, b"{malformed"),
        ]:
            with self.subTest(content_type=content_type, body=body):
                response.headers = {"Content-Type": content_type}
                response.read.return_value = body
                with mock.patch.object(self.api, "urlopen", return_value=response):
                    with self.assertRaises(self.api.VerifyError) as raised:
                        self.api.download_json(
                            "https://pypi.org/simple/spiraltorch/", simple=True
                        )
                self.assertNotIsInstance(raised.exception, self.api.PublicationPending)

    def test_cli_simple_flag_is_forwarded_with_manifest_hashes(self):
        with mock.patch.object(
            sys, "argv", ["verify", "--version", "0.4.24", "--require-simple-index"]
        ):
            with mock.patch.object(
                self.api, "github_release_wheel_digests", return_value=self.expected
            ):
                with mock.patch.object(
                    self.api, "wait_for_pypi_wheels", return_value=self.expected
                ) as wait:
                    self.assertEqual(self.api.main(), 0)
        self.assertTrue(wait.call_args.kwargs["require_simple_index"])
        self.assertEqual(wait.call_args.kwargs["expected_digests"], self.expected)

    def test_real_parsers_wait_until_the_simple_response_catches_up(self):
        simple_reads = 0

        def respond(request, **kwargs):
            nonlocal simple_reads
            self.assertGreater(kwargs["timeout"], 0)
            if "/simple/" in request.full_url:
                simple_reads += 1
                payload = self.simple_payload()
                if simple_reads == 1:
                    payload["files"] = []
                content_type = self.api.SIMPLE_JSON_TYPE
            elif request.full_url.endswith("/0.4.24/json"):
                payload = {
                    "urls": [
                        {"filename": name, "digests": {"sha256": digest}}
                        for name, digest in self.expected.items()
                    ]
                }
                content_type = "application/json"
            else:
                payload = {"info": {"version": "0.4.24"}}
                content_type = "application/json"
            response = mock.MagicMock()
            response.__enter__.return_value = response
            response.headers = {"Content-Type": content_type}
            response.read.return_value = json.dumps(payload).encode()
            return response

        with mock.patch.object(
            self.api.time, "monotonic", side_effect=lambda: self.now
        ):
            with mock.patch.object(self.api.time, "sleep", side_effect=self.sleep):
                with mock.patch.object(
                    self.api, "urlopen", side_effect=respond
                ) as reads:
                    result = self.api.wait_for_pypi_wheels(
                        "spiraltorch",
                        "0.4.24",
                        expected_wheels=2,
                        expected_digests=self.expected,
                        require_latest=True,
                        require_simple_index=True,
                        timeout=10,
                        poll_interval=1,
                    )
        self.assertEqual(result, self.expected)
        self.assertEqual(self.sleeps, [1])
        self.assertEqual(reads.call_count, 6)

    def test_requirements_pin_exact_version_and_all_approved_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requirements.txt"
            self.api.write_pip_requirements(
                path, "Spiral.Torch", "0.4.24", self.expected
            )
            self.assertEqual(
                path.read_text(),
                f"spiral-torch==0.4.24 --hash=sha256:{'1' * 64} --hash=sha256:{'2' * 64}\n",
            )
            for version in [
                "",
                "0.4.24\nother",
                "0.4.24 --extra-index-url=x",
                "file://x",
            ]:
                with self.subTest(version=version):
                    before = path.read_bytes()
                    with self.assertRaises(self.api.VerifyError):
                        self.api.write_pip_requirements(
                            path, "spiraltorch", version, self.expected
                        )
                    self.assertEqual(path.read_bytes(), before)

    def test_failed_readiness_cannot_write_install_requirements(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requirements.txt"
            argv = [
                "verify",
                "--version",
                "0.4.24",
                "--require-simple-index",
                "--pip-requirements",
                str(path),
            ]
            with mock.patch.object(sys, "argv", argv):
                with mock.patch.object(
                    self.api, "github_release_wheel_digests", return_value=self.expected
                ):
                    with mock.patch.object(
                        self.api,
                        "wait_for_pypi_wheels",
                        side_effect=self.api.VerifyError("not ready"),
                    ):
                        with self.assertRaises(self.api.VerifyError):
                            self.api.main()
            self.assertFalse(path.exists())

    def test_invalid_cli_budgets_fail_before_manifest_download(self):
        for flag, value in [
            ("--timeout", "nan"),
            ("--timeout", "0"),
            ("--poll-interval", "inf"),
            ("--poll-interval", "-1"),
            ("--expected-wheels", "0"),
        ]:
            with self.subTest(flag=flag, value=value):
                with mock.patch.object(
                    sys, "argv", ["verify", "--version", "0.4.24", flag, value]
                ):
                    with mock.patch.object(sys, "stderr", io.StringIO()):
                        with mock.patch.object(
                            self.api, "github_release_wheel_digests"
                        ) as download:
                            with self.assertRaises(SystemExit) as raised:
                                self.api.main()
                self.assertEqual(raised.exception.code, 2)
                download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
