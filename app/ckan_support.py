from __future__ import annotations

import ipaddress
import json
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any


DEFAULT_CKAN_ENDPOINT = "https://opendata.muenchen.de/"
CKAN_TIMEOUT_SECONDS = 5
ALLOW_LOCAL_CKAN_ENV = "SMOLNALYSIS_ALLOW_LOCAL_CKAN"


@dataclass
class CkanConnectionStatus:
    ok: bool
    base_url: str
    api_base: str
    message: str
    dataset_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CkanEndpointError(ValueError):
    pass


def normalize_ckan_base_url(raw_url: str) -> str:
    value = raw_url.strip()
    if not value:
        raise CkanEndpointError("Enter a CKAN endpoint URL.")

    parsed = urllib.parse.urlsplit(value)
    if not parsed.scheme:
        parsed = urllib.parse.urlsplit(f"https://{value}")

    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise CkanEndpointError("CKAN endpoint must use http or https.")
    if not parsed.netloc:
        raise CkanEndpointError("CKAN endpoint must include a host.")
    if parsed.username or parsed.password:
        raise CkanEndpointError("CKAN endpoint must not include credentials.")
    if parsed.query or parsed.fragment:
        raise CkanEndpointError("CKAN endpoint must not include query parameters or fragments.")

    host = (parsed.hostname or "").lower()
    port = f":{parsed.port}" if parsed.port else ""
    path = parsed.path.rstrip("/")
    if path.endswith("/api/3/action"):
        path = path[: -len("/api/3/action")]
    elif path.endswith("/api/action"):
        path = path[: -len("/api/action")]

    normalized = urllib.parse.urlunsplit((scheme, f"{host}{port}", path or "", "", ""))
    return f"{normalized}/"


def ckan_api_base(base_url: str) -> str:
    return urllib.parse.urljoin(base_url, "api/3/action")


def _local_addresses_allowed() -> bool:
    return os.environ.get(ALLOW_LOCAL_CKAN_ENV, "").casefold() in {"1", "true", "yes", "on"}


def _address_is_private(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _assert_public_host(base_url: str) -> None:
    if _local_addresses_allowed():
        return

    parsed = urllib.parse.urlsplit(base_url)
    host = parsed.hostname
    if not host:
        raise CkanEndpointError("CKAN endpoint must include a host.")

    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        try:
            resolved = socket.getaddrinfo(host, parsed.port, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise CkanEndpointError("Could not resolve the CKAN endpoint host.") from exc
        addresses = {item[4][0] for item in resolved}
    else:
        addresses = {str(ip)}

    if any(_address_is_private(address) for address in addresses):
        raise CkanEndpointError("CKAN endpoint must resolve to a public address.")


def _read_ckan_action(api_base: str, action: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    query = urllib.parse.urlencode(params or {})
    url = f"{api_base}/{action}"
    if query:
        url = f"{url}?{query}"

    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=CKAN_TIMEOUT_SECONDS) as response:
        body = response.read().decode("utf-8")
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise CkanEndpointError("CKAN returned an unexpected response.")
    return payload


def read_ckan_action(base_url: str, action: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    normalized = normalize_ckan_base_url(base_url)
    _assert_public_host(normalized)
    payload = _read_ckan_action(ckan_api_base(normalized), action, params)
    if payload.get("success") is not True:
        raise CkanEndpointError(f"CKAN action failed: {action}")
    result = payload.get("result")
    return result if isinstance(result, dict) else {"value": result}


def package_search(base_url: str, query: str, rows: int = 5, start: int = 0, fq: str = "") -> dict[str, Any]:
    clean_rows = max(1, min(int(rows), 10))
    clean_start = max(0, int(start))
    params: dict[str, Any] = {"q": query.strip() or "open data", "rows": clean_rows, "start": clean_start}
    if fq.strip():
        params["fq"] = fq.strip()
    return read_ckan_action(base_url, "package_search", params)


def package_show(base_url: str, package_id: str) -> dict[str, Any]:
    return read_ckan_action(base_url, "package_show", {"id": package_id})


def tag_search(base_url: str, query: str, rows: int = 10) -> dict[str, Any]:
    return read_ckan_action(base_url, "tag_search", {"query": query.strip(), "limit": max(1, min(int(rows), 25))})


def group_list(base_url: str, rows: int = 10) -> dict[str, Any]:
    return read_ckan_action(base_url, "group_list", {"all_fields": True, "limit": max(1, min(int(rows), 25))})


def organization_list(base_url: str, rows: int = 10) -> dict[str, Any]:
    return read_ckan_action(base_url, "organization_list", {"all_fields": True, "limit": max(1, min(int(rows), 25))})


def validate_ckan_endpoint(raw_url: str) -> CkanConnectionStatus:
    try:
        base_url = normalize_ckan_base_url(raw_url)
        _assert_public_host(base_url)
        api_base = ckan_api_base(base_url)

        site_read = _read_ckan_action(api_base, "site_read")
        if site_read.get("success") is not True:
            return CkanConnectionStatus(False, base_url, api_base, "CKAN site_read check failed.")

        package_search = _read_ckan_action(api_base, "package_search", {"rows": 0})
        if package_search.get("success") is not True:
            return CkanConnectionStatus(False, base_url, api_base, "CKAN package_search check failed.")

        result = package_search.get("result") or {}
        dataset_count = result.get("count") if isinstance(result, dict) else None
        return CkanConnectionStatus(
            True,
            base_url,
            api_base,
            f"Connected to CKAN endpoint. {dataset_count:,} datasets found." if isinstance(dataset_count, int) else "Connected to CKAN endpoint.",
            dataset_count if isinstance(dataset_count, int) else None,
        )
    except CkanEndpointError as exc:
        base_url = ""
        api_base = ""
        try:
            base_url = normalize_ckan_base_url(raw_url)
            api_base = ckan_api_base(base_url)
        except CkanEndpointError:
            pass
        return CkanConnectionStatus(False, base_url, api_base, str(exc))
    except (TimeoutError, urllib.error.URLError, OSError):
        base_url = normalize_ckan_base_url(raw_url)
        return CkanConnectionStatus(False, base_url, ckan_api_base(base_url), "Could not reach the CKAN endpoint.")
    except json.JSONDecodeError:
        base_url = normalize_ckan_base_url(raw_url)
        return CkanConnectionStatus(False, base_url, ckan_api_base(base_url), "CKAN endpoint did not return JSON.")


def default_ckan_status() -> CkanConnectionStatus:
    base_url = normalize_ckan_base_url(DEFAULT_CKAN_ENDPOINT)
    return CkanConnectionStatus(False, base_url, ckan_api_base(base_url), "Not connected.")
