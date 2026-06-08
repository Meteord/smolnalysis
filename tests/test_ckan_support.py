from __future__ import annotations

import json
import sys
import urllib.error
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import MagicMock, patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from ckan_support import ckan_api_base, normalize_ckan_base_url, validate_ckan_endpoint


class CkanSupportTests(TestCase):
    def test_normalizes_munich_endpoint(self) -> None:
        base_url = normalize_ckan_base_url("https://opendata.muenchen.de/")
        self.assertEqual(base_url, "https://opendata.muenchen.de/")
        self.assertEqual(ckan_api_base(base_url), "https://opendata.muenchen.de/api/3/action")

    def test_rejects_invalid_urls(self) -> None:
        for value in ["", "ftp://example.com", "https://user@example.com", "https://example.com?q=1", "https://example.com#x"]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    normalize_ckan_base_url(value)

    @patch("ckan_support._assert_public_host")
    @patch("ckan_support.urllib.request.urlopen")
    def test_successful_connection(self, urlopen: MagicMock, _public_host: MagicMock) -> None:
        urlopen.side_effect = [
            _response({"success": True, "result": True}),
            _response({"success": True, "result": {"count": 42}}),
        ]

        status = validate_ckan_endpoint("https://opendata.muenchen.de/")

        self.assertTrue(status.ok)
        self.assertEqual(status.dataset_count, 42)
        self.assertEqual(status.api_base, "https://opendata.muenchen.de/api/3/action")

    @patch("ckan_support._assert_public_host")
    @patch("ckan_support.urllib.request.urlopen")
    def test_ckan_error_response(self, urlopen: MagicMock, _public_host: MagicMock) -> None:
        urlopen.return_value = _response({"success": False, "error": {"message": "nope"}})

        status = validate_ckan_endpoint("https://opendata.muenchen.de/")

        self.assertFalse(status.ok)
        self.assertIn("site_read", status.message)

    @patch("ckan_support._assert_public_host")
    @patch("ckan_support.urllib.request.urlopen")
    def test_network_error_is_friendly(self, urlopen: MagicMock, _public_host: MagicMock) -> None:
        urlopen.side_effect = urllib.error.URLError("timeout")

        status = validate_ckan_endpoint("https://opendata.muenchen.de/")

        self.assertFalse(status.ok)
        self.assertEqual(status.message, "Could not reach the CKAN endpoint.")


def _response(payload: dict):
    response = MagicMock()
    response.__enter__.return_value = response
    response.read.return_value = json.dumps(payload).encode("utf-8")
    return response


if __name__ == "__main__":
    main()
