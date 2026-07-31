"""Score-source ingestion — paste / PDF / URL / DOI / GitHub-repo adapters and their fetch bounds.

No real network: every fetch test drives ``from_url`` through an ``httpx.MockTransport`` and stubs DNS
(``_resolve``) so the SSRF guard can be exercised without leaving the machine.
"""

from __future__ import annotations

import io
import json

import httpx
import pytest

from ddharmon.harmonization import score_sources as ss
from ddharmon.harmonization.score_sources import ScoreSource, from_docx, from_text, from_url, html_to_text

_ZIP_HEADER = b"PK\x03\x04" + b"\x00" * 8  # enough to pass the container check and reach the parser import


@pytest.fixture(autouse=True)
def _public_dns(monkeypatch):
    """Resolve every host to a public address by default; individual tests override."""
    monkeypatch.setattr(ss, "_resolve", lambda host: ["93.184.216.34"])


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)


# --- paste ------------------------------------------------------------------------------------


def test_from_text_normalizes_and_fingerprints():
    """Pasted text is whitespace-normalized and fingerprinted so a spec can name its exact source."""
    src = from_text("  Fried   phenotype:\r\n\r\n\r\n  5 criteria \t here  ")
    assert src.kind == "paste"
    assert src.text == "Fried phenotype:\n\n5 criteria here"
    assert len(src.sha256) == 64
    assert src.n_chars == len(src.text)


def test_from_text_rejects_empty():
    with pytest.raises(ValueError, match="empty score source"):
        from_text("   \n  ")


def test_sha256_is_content_addressed():
    """Same text → same fingerprint regardless of adapter; different text → different fingerprint."""
    assert ScoreSource(text="abc", kind="paste").sha256 == ScoreSource(text="abc", kind="url").sha256
    assert ScoreSource(text="abc", kind="paste").sha256 != ScoreSource(text="abd", kind="paste").sha256


# --- HTML extraction --------------------------------------------------------------------------


def test_html_to_text_drops_script_style_and_keeps_block_breaks():
    markup = (
        "<html><head><title>x</title></head><body><script>var a=1;</script>"
        "<style>p{color:red}</style><h1>FI-Lab</h1><p>32 deficits</p>"
        "<table><tr><td>Hemoglobin</td><td>&lt;130 g/L</td></tr></table></body></html>"
    )
    text = html_to_text(markup)
    assert "var a=1" not in text and "color:red" not in text  # script/style bodies dropped
    assert "<script" not in text and "<p>" not in text and "<td>" not in text  # tags stripped
    assert "FI-Lab" in text and "32 deficits" in text
    assert "Hemoglobin" in text and "<130 g/L" in text  # &lt; unescaped back to a real "<"
    assert "FI-Lab\n\n32 deficits" in text  # block-level tags became line breaks


# --- fetch bounds / SSRF guard ----------------------------------------------------------------


def test_from_url_refuses_non_http_scheme():
    with pytest.raises(ValueError, match="only http\\(s\\)"):
        from_url("file:///etc/passwd")


@pytest.mark.parametrize("addr", ["127.0.0.1", "10.0.0.5", "169.254.169.254", "::1"])
def test_from_url_refuses_non_public_addresses(monkeypatch, addr):
    """A public-looking hostname that resolves into private/loopback/link-local space is refused."""
    monkeypatch.setattr(ss, "_resolve", lambda host: [addr])
    with pytest.raises(ValueError, match="non-public address"):
        from_url("https://internal.example.com/score")


def test_from_url_refuses_localhost_by_name(monkeypatch):
    monkeypatch.setattr(ss, "_resolve", lambda host: pytest.fail("must not resolve a blocked hostname"))
    with pytest.raises(ValueError, match="local address space"):
        from_url("http://localhost:8000/score")


def test_from_url_revalidates_each_redirect_hop(monkeypatch):
    """A public host redirecting to cloud metadata is caught on the hop, not just the first URL."""
    resolutions = {"paper.example.com": ["93.184.216.34"], "169.254.169.254": ["169.254.169.254"]}
    monkeypatch.setattr(ss, "_resolve", lambda host: resolutions[host])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/meta-data/"})

    with _client(handler) as client, pytest.raises(ValueError, match="non-public address"):
        from_url("https://paper.example.com/score", client=client)


def test_from_url_caps_body_size():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/plain"}, content=b"x" * 4096)

    with _client(handler) as client, pytest.raises(ValueError, match="score-source cap"):
        from_url("https://paper.example.com/huge", client=client, max_bytes=1024)


def test_from_url_stops_after_max_redirects():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://paper.example.com/next"})

    with _client(handler) as client, pytest.raises(ValueError, match="too many redirects"):
        from_url("https://paper.example.com/start", client=client, max_redirects=2)


# --- fetch failures are user-actionable ValueErrors, never raw httpx ---------------------------


def test_publisher_403_names_the_recovery_not_a_raw_httpx_error():
    """The real smoke-test failure: a DOI resolves to a Cloudflare-protected journal that refuses the fetch.

    Raw `httpx.HTTPStatusError` escaped the hosted route (which maps only ValueError to a 4xx) and became an
    opaque 500. This is the single most likely outcome of the URL/DOI path, so it must arrive as guidance.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "doi.org":
            return httpx.Response(302, headers={"location": "https://academic.oup.com/article/74/4/582"})
        return httpx.Response(403, headers={"content-type": "text/html"}, content=b"<html>Just a moment...</html>")

    with _client(handler) as client, pytest.raises(ValueError) as exc:
        from_url("10.1093/gerona/gly094", client=client)
    message = str(exc.value)
    assert "403" in message
    assert "academic.oup.com" in message  # the URL that actually refused, post-redirect
    assert "upload the PDF" in message


@pytest.mark.parametrize(
    ("status", "expected"),
    [(404, "was not found"), (429, "rate-limiting"), (500, "server error"), (418, "418")],
)
def test_error_statuses_explain_themselves(status, expected):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status)

    with _client(handler) as client, pytest.raises(ValueError, match=expected):
        from_url("https://paper.example.com/score", client=client)


def test_github_rate_limit_is_named_as_such():
    """GitHub answers an over-quota unauthenticated API call with 403 — 'paywalled publisher' would mislead."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"message": "API rate limit exceeded"})

    with _client(handler) as client, pytest.raises(ValueError, match="rate limit"):
        from_url("https://github.com/acme/frailty", client=client)


def test_timeout_is_a_value_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    with _client(handler) as client, pytest.raises(ValueError, match="timed out after"):
        from_url("https://paper.example.com/slow", client=client, timeout=3)


def test_transport_failure_is_a_value_error():
    """DNS/TLS/connection-reset failures reach the caller as a readable ValueError too."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with _client(handler) as client, pytest.raises(ValueError, match="could not fetch"):
        from_url("https://paper.example.com/down", client=client)


def test_from_url_fetches_html_and_records_final_url():
    """Provenance is the URL we actually read (post-redirect), not the one the user typed."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/doi":
            return httpx.Response(302, headers={"location": "https://paper.example.com/article"})
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            content=b"<html><body><h1>FI-Lab</h1><p>proportion of deficits</p></body></html>",
        )

    with _client(handler) as client:
        src = from_url("https://paper.example.com/doi", client=client)
    assert src.kind == "url"
    assert src.provenance == "https://paper.example.com/article"
    assert "proportion of deficits" in src.text


def test_bare_doi_resolves_through_doi_org():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, headers={"content-type": "text/plain"}, content=b"FI-Lab: 32 deficits")

    with _client(handler) as client:
        src = from_url("10.1007/s11357-017-9993-7", client=client)
    assert seen == ["https://doi.org/10.1007/s11357-017-9993-7"]
    assert src.text == "FI-Lab: 32 deficits"


# --- PDF --------------------------------------------------------------------------------------


def test_pdf_path_reports_the_extra_when_pypdf_missing(monkeypatch):
    """The PDF path is optional — the error must name the extra to install, not a bare ImportError."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pypdf":
            raise ImportError("No module named 'pypdf'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"ddharmon\[sources\]"):
        ss.pdf_to_text(b"%PDF-1.4")


def test_pdf_bytes_that_are_not_a_pdf_are_named_as_such():
    """The real case: a publisher answers a `.pdf` URL with an HTML access page. Blaming pypdf would mislead."""
    with pytest.raises(ValueError, match="not a PDF"):
        ss.pdf_to_text(b"<html><body>Access check</body></html>")


def test_a_pdf_url_serving_html_falls_back_to_html_extraction():
    """PMC does exactly this. The interstitial may still carry the definition, so read it as HTML."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/pdf"},
            content=b"<!doctype html><html><body><h1>FI-Lab</h1><p>32 deficits</p></body></html>",
        )

    with _client(handler) as client:
        src = from_url("https://paper.example.com/article.pdf", client=client)
    assert "32 deficits" in src.text and "<h1>" not in src.text


def test_url_pdf_routes_through_pdf_extraction(monkeypatch):
    """A fetched application/pdf body goes to the PDF extractor, not the HTML/plain path."""
    monkeypatch.setattr(ss, "pdf_to_text", lambda data: "Fried phenotype: weight loss, exhaustion")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF-1.4 ...")

    with _client(handler) as client:
        src = from_url("https://paper.example.com/fried.pdf", client=client)
    assert src.text.startswith("Fried phenotype")


# --- Word (.docx) -----------------------------------------------------------------------------


def _docx_bytes(build) -> bytes:
    """Build a real .docx in memory with python-docx and return its bytes."""
    docx = pytest.importorskip("docx")
    buffer = io.BytesIO()
    document = docx.Document()
    build(document)
    document.save(buffer)
    return buffer.getvalue()


def test_docx_extraction_keeps_table_cells():
    """THE case this adapter exists for: a supplement whose item table is a table, not prose.

    `Document.paragraphs` omits table content entirely, so a naive reader returns the surrounding prose and
    silently drops the item list — the one thing the caller came for.
    """

    def build(document):
        document.add_paragraph("Supplementary Table 1. Items of the frailty index.")
        table = document.add_table(rows=3, cols=2)
        table.cell(0, 0).text = "Deficit"
        table.cell(0, 1).text = "Cut-point"
        table.cell(1, 0).text = "Hemoglobin"
        table.cell(1, 1).text = "<130 g/L"
        table.cell(2, 0).text = "Grip strength"
        table.cell(2, 1).text = "lowest quintile"

    src = from_docx(_docx_bytes(build), provenance="supplement.docx")
    assert src.kind == "docx"
    assert src.provenance == "supplement.docx"
    assert "Supplementary Table 1" in src.text  # prose kept
    assert "Hemoglobin | <130 g/L" in src.text  # ...and the table's rows survived with their structure
    assert "Grip strength | lowest quintile" in src.text


def test_docx_extraction_preserves_document_order():
    """Paragraphs and tables interleave as the document orders them, so a table stays under its heading."""

    def build(document):
        document.add_paragraph("Section A")
        table_a = document.add_table(rows=1, cols=1)
        table_a.cell(0, 0).text = "belongs to A"
        document.add_paragraph("Section B")
        table_b = document.add_table(rows=1, cols=1)
        table_b.cell(0, 0).text = "belongs to B"

    text = from_docx(_docx_bytes(build)).text
    assert text.index("Section A") < text.index("belongs to A") < text.index("Section B") < text.index("belongs to B")


def test_legacy_doc_is_named_as_a_different_format():
    """A binary .doc needs re-saving, not a better parser — say so instead of blaming the zip container."""
    with pytest.raises(ValueError, match="legacy binary .doc"):
        ss.docx_to_text(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1rest of an OLE2 file")


def test_non_docx_bytes_are_rejected_before_the_parser():
    with pytest.raises(ValueError, match="not a .docx"):
        ss.docx_to_text(b"{\\rtf1\\ansi an RTF file}")


def test_docx_path_reports_the_extra_when_python_docx_missing(monkeypatch):
    """The Word path is optional — the error must name the extra to install."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "docx" or name.startswith("docx."):
            raise ImportError("No module named 'docx'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"ddharmon\[sources\]"):
        ss.docx_to_text(_ZIP_HEADER)


def test_a_fetched_docx_supplement_is_extracted():
    """Supplements are often served straight off a publisher's static host as a .docx."""

    def build(document):
        table = document.add_table(rows=1, cols=2)
        table.cell(0, 0).text = "Weight loss"
        table.cell(0, 1).text = ">5% in a year"

    body = _docx_bytes(build)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": ss._DOCX_CTYPE}, content=body)

    with _client(handler) as client:
        src = from_url("https://paper.example.com/supplement.docx", client=client)
    assert "Weight loss | >5% in a year" in src.text


def test_a_docx_url_serving_html_falls_back_to_html_extraction():
    """Same guard as the PDF path: an access-check page must not reach the docx parser."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": ss._DOCX_CTYPE},
            content=b"<!doctype html><html><body><p>Sign in to continue</p></body></html>",
        )

    with _client(handler) as client:
        src = from_url("https://paper.example.com/supplement.docx", client=client)
    assert "Sign in to continue" in src.text


# --- GitHub repo ------------------------------------------------------------------------------


def test_github_repo_reads_readme_plus_included_files():
    """A repo source concatenates README + `include`-matched files, each labeled with its path."""
    tree = {
        "tree": [
            {"path": "README.md", "type": "blob"},
            {"path": "src/frailty_index.py", "type": "blob"},
            {"path": "src/unrelated.py", "type": "blob"},
            {"path": "docs/methods.md", "type": "blob"},
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == "https://api.github.com/repos/acme/frailty":
            return httpx.Response(200, json={"default_branch": "trunk"})
        if url.startswith("https://api.github.com/repos/acme/frailty/git/trees/trunk"):
            return httpx.Response(200, json=tree)
        if url.endswith("/trunk/README.md"):
            return httpx.Response(200, headers={"content-type": "text/plain"}, content=b"FI implementation")
        if url.endswith("/trunk/src/frailty_index.py"):
            return httpx.Response(200, headers={"content-type": "text/plain"}, content=b"deficits / total")
        if url.endswith("/trunk/docs/methods.md"):
            return httpx.Response(200, headers={"content-type": "text/plain"}, content=b"32 lab deficits")
        return httpx.Response(404)

    with _client(handler) as client:
        src = from_url("https://github.com/acme/frailty", client=client, include=r"frailty_index")
    assert src.kind == "repo"
    assert src.provenance == "github.com/acme/frailty@trunk"
    assert src.parts == ["README.md", "src/frailty_index.py", "docs/methods.md"]
    assert "===== src/frailty_index.py =====" in src.text
    assert "deficits / total" in src.text and "src/unrelated.py" not in src.text


def test_github_repo_honours_explicit_ref_and_file_cap():
    tree = {"tree": [{"path": f"docs/methods{i}.md", "type": "blob"} for i in range(6)]}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.startswith("https://api.github.com/repos/acme/frailty/git/trees/v2"):
            return httpx.Response(200, json=tree)
        if url == "https://api.github.com/repos/acme/frailty":
            pytest.fail("must not query the default branch when a /tree/<ref> was given")
        return httpx.Response(200, headers={"content-type": "text/plain"}, content=b"body")

    with _client(handler) as client:
        src = from_url("https://github.com/acme/frailty/tree/v2", client=client, max_files=2)
    assert src.provenance == "github.com/acme/frailty@v2"
    assert len(src.parts) == 2


def test_github_repo_without_docs_is_an_honest_error():
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == "https://api.github.com/repos/acme/frailty":
            return httpx.Response(200, json={"default_branch": "main"})
        return httpx.Response(200, json={"tree": [{"path": "src/a.py", "type": "blob"}]})

    with _client(handler) as client, pytest.raises(ValueError, match="no README or documentation"):
        from_url("https://github.com/acme/frailty", client=client)


# --- dispatch ---------------------------------------------------------------------------------


def test_fetch_source_dispatch(tmp_path, monkeypatch):
    """`fetch_source` routes on the shape of the reference: file path, URL, or literal text."""
    txt = tmp_path / "fried.txt"
    txt.write_text("Fried phenotype: 5 criteria", encoding="utf-8")
    assert ss.fetch_source(str(txt)).kind == "paste"
    assert ss.fetch_source(str(txt)).provenance == "fried.txt"
    assert ss.fetch_source("A score defined inline\nwith two lines").kind == "paste"

    monkeypatch.setattr(ss, "from_url", lambda ref, **kw: ScoreSource(text=f"fetched {ref}", kind="url"))
    assert ss.fetch_source("https://example.com/x").text == "fetched https://example.com/x"
    assert ss.fetch_source("10.1007/s11357-017-9993-7").kind == "url"


def test_fetch_source_pdf_bytes_and_path(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "pdf_to_text", lambda data: "deficit accumulation")
    pdf = tmp_path / "fi.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    assert ss.fetch_source(pdf).kind == "pdf"
    assert ss.fetch_source(b"%PDF-1.4").text == "deficit accumulation"


def test_fetch_source_routes_word_by_magic_and_by_suffix(tmp_path):
    """Bytes are routed by magic number — an upload may arrive with no usable filename at all."""

    def build(document):
        document.add_paragraph("Fried phenotype: 5 criteria")

    data = _docx_bytes(build)
    assert ss.fetch_source(data).kind == "docx"  # bytes: zip magic, no filename involved

    path = tmp_path / "supplement.docx"
    path.write_bytes(data)
    assert ss.fetch_source(path).kind == "docx"  # Path
    assert ss.fetch_source(str(path)).kind == "docx"  # path as a string


def test_legacy_doc_bytes_are_routed_to_the_word_reader_not_the_pdf_one():
    """Otherwise an uploaded .doc is blamed on the wrong format: "not a PDF" instead of "re-save as .docx"."""
    with pytest.raises(ValueError, match="legacy binary .doc"):
        ss.fetch_source(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1 an old Word file")


def test_json_tree_response_is_parsed_not_stripped():
    """Guard the repo path's JSON handling: an application/json body must not go through the HTML stripper."""
    payload = json.dumps({"default_branch": "main"}).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == "https://api.github.com/repos/acme/frailty":
            return httpx.Response(200, headers={"content-type": "application/json"}, content=payload)
        if "git/trees" in url:
            return httpx.Response(200, json={"tree": [{"path": "README.md", "type": "blob"}]})
        return httpx.Response(200, headers={"content-type": "text/plain"}, content=b"readme body")

    with _client(handler) as client:
        src = from_url("https://github.com/acme/frailty", client=client)
    assert "readme body" in src.text
