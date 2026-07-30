"""Score-source ingestion: turn a document that DEFINES a composite score into plain text.

The composite/derived-variable builder (:mod:`ddharmon.harmonization.composite`) requires the score's
definition to come from a real document — a paper, a supplement table, a repo that implements the index —
never from a model's recollection of it. This module is that front door: every adapter returns a
:class:`ScoreSource` carrying the extracted text plus provenance (what it came from and a sha256 of what we
read), so a derived spec can always name its source.

Adapters::

    from_text(text)             # pasted methods section / component table
    from_pdf(path_or_bytes)     # uploaded paper or supplement (needs ddharmon[sources])
    from_url(url_or_doi)        # fetched HTML page, PDF, DOI, or GitHub repo
    fetch_source(ref)           # dispatch on what `ref` looks like

Fetching is deliberately bounded: http(s) only, every redirect hop re-validated, private/loopback/link-local
address space refused, a byte cap, and a timeout. The builder is reachable from a hosted UI, so a
user-supplied URL must not become an SSRF primitive.

Metadata-only discipline is unaffected — these are *documents describing a score*, never participant data.
"""

from __future__ import annotations

import hashlib
import html
import io
import ipaddress
import json
import re
import socket
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from ftfy import fix_text

_TIMEOUT = 10.0
_MAX_BYTES = 5 * 1024 * 1024  # 5 MB — papers and READMEs are far smaller; a bigger body is a mistake
_MAX_REDIRECTS = 5  # doi.org -> publisher -> article can legitimately take a few hops
_MAX_REPO_FILES = 5
_USER_AGENT = "ddharmon/2.0 (+https://ddharmon.io) score-source fetcher"

_DOI_RE = re.compile(r"^(?:doi:)?(10\.\d{4,9}/\S+)$", re.IGNORECASE)
_GITHUB_RE = re.compile(r"^https?://(?:www\.)?github\.com/([^/\s]+)/([^/\s#?]+?)(?:\.git)?(?:/tree/([^/\s#?]+))?/?$")
_SCRIPT_RE = re.compile(r"<(script|style|head)\b.*?</\1>", re.DOTALL | re.IGNORECASE)
_BREAK_RE = re.compile(r"</?(br|p|div|tr|li|h[1-6]|table|section)\b[^>]*>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_BLANKS_RE = re.compile(r"[ \t]*\n[ \t]*")
_MULTI_NL_RE = re.compile(r"\n{3,}")
_REPO_DOC_RE = re.compile(r"(^|/)(readme|methods?|scoring|index|definitions?)[^/]*\.(md|rst|txt)$", re.IGNORECASE)
_BLOCKED_HOSTNAMES = frozenset({"localhost", "localhost.localdomain", "metadata", "metadata.google.internal"})


@dataclass
class ScoreSource:
    """A document that defines a composite score, reduced to text plus where it came from.

    ``kind`` is ``paste | pdf | url | repo``. ``provenance`` is human-facing (a URL, a filename, an
    ``owner/repo@ref``); ``sha256`` fingerprints the extracted text so a spec can be tied to exactly the
    bytes that produced it. ``parts`` names the constituent documents when several were concatenated
    (a repo's README + docs), and is empty for a single-document source.
    """

    text: str
    kind: str
    provenance: str = ""
    sha256: str = ""
    parts: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.sha256:
            self.sha256 = hashlib.sha256(self.text.encode("utf-8", "replace")).hexdigest()

    @property
    def n_chars(self) -> int:
        return len(self.text)


# --- text / PDF / HTML extraction ------------------------------------------------------------


def from_text(text: str, *, provenance: str = "pasted text") -> ScoreSource:
    """Wrap pasted text (a methods section, a component table) as a :class:`ScoreSource`."""
    cleaned = _normalize(text)
    if not cleaned:
        raise ValueError("empty score source: nothing to read a score definition from")
    return ScoreSource(text=cleaned, kind="paste", provenance=provenance)


def pdf_to_text(data: bytes) -> str:
    """Extract text from PDF bytes via ``pypdf``.

    Raises ``ImportError`` naming the extra when ``pypdf`` is absent — the PDF path is optional so the core
    install stays lean — and ``ValueError`` when ``data`` is not a PDF at all. That second case is the common
    one in the wild: publishers answer a ``.pdf`` URL with an HTML interstitial or access-check page, and a
    pypdf stack trace would blame the wrong thing.
    """
    if not data.lstrip()[:5].startswith(b"%PDF"):
        raise ValueError("not a PDF (no %PDF header) — the server likely returned an HTML/access-check page")
    try:
        from pypdf import PdfReader  # optional dep — imported only when the PDF path is used
    except ImportError as exc:
        raise ImportError("reading PDF score sources needs pypdf — install `ddharmon[sources]`") from exc

    reader = PdfReader(io.BytesIO(data))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def from_pdf(source: bytes | str | Path, *, provenance: str = "") -> ScoreSource:
    """Extract a score definition from an uploaded/on-disk PDF (``bytes`` or a path)."""
    if isinstance(source, bytes):
        data, name = source, provenance or "uploaded.pdf"
    else:
        path = Path(source)
        data, name = path.read_bytes(), provenance or path.name
    text = _normalize(pdf_to_text(data))
    if not text:
        raise ValueError(f"no extractable text in {name} — a scanned/image-only PDF needs OCR first")
    return ScoreSource(text=text, kind="pdf", provenance=name)


def html_to_text(markup: str) -> str:
    """Reduce an HTML page to readable text: drop script/style/head, keep block breaks, strip tags.

    Deliberately dependency-free (no bs4). Good enough for a paper's methods section or a docs page, which
    is all the definition-extraction pass needs.
    """
    t = _SCRIPT_RE.sub(" ", markup)
    t = _BREAK_RE.sub("\n", t)
    t = _TAG_RE.sub(" ", t)
    return _normalize(html.unescape(t))


def _normalize(text: str) -> str:
    """Fix mojibake, collapse horizontal whitespace, and cap consecutive blank lines."""
    t = fix_text(str(text or ""))
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t ]+", " ", t)
    t = _BLANKS_RE.sub("\n", t)
    return _MULTI_NL_RE.sub("\n\n", t).strip()


def _decode(body: bytes, content_type: str, name: str = "") -> str:
    """Turn a fetched body into text, routing on content type (and filename) — PDF, HTML, or plain.

    A body advertised as PDF that carries no ``%PDF`` header falls through to the HTML/plain path instead of
    failing: a publisher answering a ``.pdf`` URL with an interstitial page is routine, and the page itself
    may still carry the definition.
    """
    ctype = (content_type or "").lower()
    if ("pdf" in ctype or name.lower().endswith(".pdf")) and body.lstrip()[:5].startswith(b"%PDF"):
        return _normalize(pdf_to_text(body))
    text = body.decode("utf-8", errors="replace")
    if text.lstrip()[:200].lower().startswith(("<!doctype html", "<html")):
        return html_to_text(text)
    if "html" in ctype or "xml" in ctype or name.lower().endswith((".html", ".htm")):
        return html_to_text(text)
    return _normalize(text)


# --- bounded fetching -----------------------------------------------------------------------


def _resolve(host: str) -> list[str]:
    """Resolve ``host`` to its IP strings. Separate function so tests can stub DNS."""
    return [str(info[4][0]) for info in socket.getaddrinfo(host, None)]


def _check_url(url: str) -> None:
    """Reject a URL we must not fetch: non-http(s), no host, or one resolving into non-public address space.

    Called for the initial URL *and* every redirect hop, because a public host can redirect to
    ``169.254.169.254`` (cloud metadata) or ``127.0.0.1``.
    """
    parsed = httpx.URL(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"refusing to fetch {parsed.scheme or '(no scheme)'}:// — only http(s) is allowed")
    host = parsed.host
    if not host:
        raise ValueError(f"refusing to fetch {url!r}: no host")
    if host.lower() in _BLOCKED_HOSTNAMES:
        raise ValueError(f"refusing to fetch {host!r}: local address space")
    try:
        addresses = _resolve(host)
    except OSError as exc:
        raise ValueError(f"cannot resolve {host!r}: {exc}") from exc
    for addr in addresses:
        ip = ipaddress.ip_address(addr.split("%", 1)[0])  # strip any scope id
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise ValueError(f"refusing to fetch {host!r}: resolves to non-public address {ip}")


def _fetch(
    url: str,
    *,
    client: httpx.Client,
    timeout: float = _TIMEOUT,
    max_bytes: int = _MAX_BYTES,
    max_redirects: int = _MAX_REDIRECTS,
) -> tuple[bytes, str, str]:
    """GET ``url`` with hop-by-hop validation and a hard byte cap. Returns ``(body, content_type, final_url)``.

    Redirects are followed manually (``follow_redirects=False``) so each hop passes :func:`_check_url`.
    """
    current = url
    for _ in range(max_redirects + 1):
        _check_url(current)
        with client.stream(
            "GET",
            current,
            timeout=timeout,
            follow_redirects=False,
            headers={"User-Agent": _USER_AGENT, "Accept": "*/*"},
        ) as response:
            if response.status_code in (301, 302, 303, 307, 308):
                location = response.headers.get("location")
                if not location:
                    raise ValueError(f"redirect from {current} without a Location header")
                current = str(httpx.URL(current).join(location))
                continue
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_bytes():
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"{current} exceeds the {max_bytes // 1024 // 1024} MB score-source cap")
                chunks.append(chunk)
            return b"".join(chunks), content_type, current
    raise ValueError(f"too many redirects (>{max_redirects}) starting at {url}")


def _pick_repo_paths(paths: list[str], include: str | None, max_files: int) -> list[str]:
    """Choose which repo files describe the score: root README first, then ``include`` matches, then docs.

    ``include`` is a caller-supplied regex over paths (e.g. ``"frailty|fi_lab"``) so a repo whose index
    lives in code, not prose, can still be read.
    """
    picked: list[str] = []

    def add(path: str) -> None:
        if path not in picked and len(picked) < max_files:
            picked.append(path)

    for path in paths:
        if "/" not in path and path.lower().startswith("readme"):
            add(path)
    if include:
        pattern = re.compile(include, re.IGNORECASE)
        for path in paths:
            if pattern.search(path):
                add(path)
    for path in paths:
        if _REPO_DOC_RE.search(path):
            add(path)
    return picked


def _from_github_repo(
    owner: str,
    repo: str,
    ref: str | None,
    *,
    client: httpx.Client,
    include: str | None,
    max_files: int,
    timeout: float,
    max_bytes: int,
    max_redirects: int,
) -> ScoreSource:
    """Read a repo's score definition from its README plus up to ``max_files`` doc/code files.

    Uses the public GitHub API (unauthenticated) to list the tree, then fetches the selected blobs from
    ``raw.githubusercontent.com``. Each part is prefixed with its path so the extraction pass — and a human
    reading the provenance — can tell which file said what.
    """

    def get(url: str) -> tuple[bytes, str, str]:
        return _fetch(url, client=client, timeout=timeout, max_bytes=max_bytes, max_redirects=max_redirects)

    def get_json(url: str) -> dict:
        body, _, _ = get(url)
        return json.loads(body.decode("utf-8", errors="replace"))

    api = f"https://api.github.com/repos/{owner}/{repo}"
    if not ref:
        ref = str(get_json(api).get("default_branch") or "main")
    tree = get_json(f"{api}/git/trees/{ref}?recursive=1")
    paths = [str(e.get("path")) for e in tree.get("tree", []) if e.get("type") == "blob"]
    picked = _pick_repo_paths(paths, include, max_files)
    if not picked:
        raise ValueError(f"no README or documentation found in {owner}/{repo}@{ref} to read a definition from")

    sections: list[str] = []
    for path in picked:
        body, ctype, _ = get(f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}")
        text = _decode(body, ctype, path)
        if text:
            sections.append(f"===== {path} =====\n{text}")
    if not sections:
        raise ValueError(f"no extractable text in {owner}/{repo}@{ref}")
    return ScoreSource(
        text="\n\n".join(sections),
        kind="repo",
        provenance=f"github.com/{owner}/{repo}@{ref}",
        parts=picked,
    )


def from_url(
    ref: str,
    *,
    client: httpx.Client | None = None,
    include: str | None = None,
    max_files: int = _MAX_REPO_FILES,
    timeout: float = _TIMEOUT,
    max_bytes: int = _MAX_BYTES,
    max_redirects: int = _MAX_REDIRECTS,
) -> ScoreSource:
    """Fetch a score definition from a URL, a bare DOI, or a GitHub repo.

    ``ref`` may be an ``http(s)`` URL (HTML page or PDF), a bare DOI (``10.1007/s11357-017-9993-7``, resolved
    via doi.org), or a GitHub repo URL (README + docs, see :func:`_pick_repo_paths`). ``include`` narrows the
    repo file selection. Pass ``client`` to inject an ``httpx.Client`` (tests use ``httpx.MockTransport``).

    Bounded by design: http(s) only, each redirect hop re-validated against non-public address space,
    ``timeout`` seconds, and ``max_bytes`` per document.
    """
    target = str(ref or "").strip()
    if not target:
        raise ValueError("empty score-source reference")

    doi = _DOI_RE.match(target)
    if doi:
        target = f"https://doi.org/{doi.group(1)}"
    elif "://" not in target:
        target = f"https://{target}"

    owned = client is None
    http = client or httpx.Client(follow_redirects=False)
    try:
        repo_match = _GITHUB_RE.match(target)
        if repo_match:
            owner, repo, tree_ref = repo_match.groups()
            return _from_github_repo(
                owner,
                repo,
                tree_ref,
                client=http,
                include=include,
                max_files=max_files,
                timeout=timeout,
                max_bytes=max_bytes,
                max_redirects=max_redirects,
            )
        body, content_type, final_url = _fetch(
            target, client=http, timeout=timeout, max_bytes=max_bytes, max_redirects=max_redirects
        )
        text = _decode(body, content_type, final_url)
        if not text:
            raise ValueError(f"no extractable text at {final_url}")
        return ScoreSource(text=text, kind="url", provenance=final_url)
    finally:
        if owned:
            http.close()


def fetch_source(ref: str | bytes | Path, **kwargs) -> ScoreSource:
    """Dispatch on what ``ref`` looks like: PDF bytes/path → :func:`from_pdf`, URL/DOI/repo →
    :func:`from_url`, anything else → :func:`from_text`.

    ``kwargs`` are forwarded to the chosen adapter, so a caller that does not know the source kind up front
    (a CLI flag, a UI field) still gets the right one.
    """
    if isinstance(ref, bytes):
        return from_pdf(ref, **kwargs)
    if isinstance(ref, Path):
        return from_pdf(ref, **kwargs) if ref.suffix.lower() == ".pdf" else from_text(ref.read_text(), **kwargs)

    text = str(ref).strip()
    if _DOI_RE.match(text) or re.match(r"^https?://", text):
        return from_url(text, **kwargs)
    candidate = Path(text)
    if len(text) < 4096 and "\n" not in text and candidate.is_file():
        if candidate.suffix.lower() == ".pdf":
            return from_pdf(candidate, **kwargs)
        return from_text(candidate.read_text(encoding="utf-8", errors="replace"), provenance=candidate.name)
    return from_text(text, **kwargs)
