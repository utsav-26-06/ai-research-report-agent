"""
Citation Management Module (TASK-014).
Handles deduplication of citations across findings and builds the final reference list.
"""

from urllib.parse import urlparse
from app.models.report import Citation, Finding


def clean_title_or_fallback(title: str | None, url: str, domain: str = "") -> str:
    """Ensure a clean, human-readable title; derive from URL/domain if missing or generic."""
    if title:
        t = title.strip()
        if t and t.lower() not in ("untitled", "untitled document", "none", "null", "unknown"):
            return t

    parsed = urlparse(url)
    dom = domain.replace("www.", "").strip() if domain else parsed.netloc.replace("www.", "").strip()

    common_exts = (".html", ".htm", ".php", ".asp", ".aspx", ".jsp", ".pdf")
    path = parsed.path.strip("/")
    if path:
        slug = path.split("/")[-1]
        for ext in common_exts:
            if slug.lower().endswith(ext):
                slug = slug[:-len(ext)]
                break
        slug_clean = slug.replace("-", " ").replace("_", " ").strip()
        if len(slug_clean) > 3:
            return f"{slug_clean.title()} ({dom})" if dom else slug_clean.title()

    return f"{dom.capitalize()} Article" if dom else "Web Source"


class CitationManager:
    """
    Manages citations across a research session.
    Deduplicates sources based on URL and assigns global sequential markers [1], [2], etc.
    """

    def __init__(self):
        # Maps URL to the assigned global marker (e.g. "https://example.com" -> "[1]")
        self._url_to_marker: dict[str, str] = {}
        # Stores the first Citation object encountered for each unique URL
        self._unique_citations: list[Citation] = []

    def process_findings(self, findings: list[Finding]) -> list[Finding]:
        """
        Update findings with globally deduplicated citations and append markers to claims.
        """
        for finding in findings:
            unique_markers_for_finding = set()
            
            for cit in finding.citations:
                # Ensure citation has a clean title
                cit.title = clean_title_or_fallback(cit.title, cit.url, cit.domain)

                if cit.url not in self._url_to_marker:
                    # Assign a new sequential marker
                    marker = f"[{len(self._url_to_marker) + 1}]"
                    self._url_to_marker[cit.url] = marker
                    
                    # Store a copy as the canonical reference
                    canonical_cit = cit.model_copy()
                    canonical_cit.marker = marker
                    self._unique_citations.append(canonical_cit)
                
                # Update the citation's marker to the global one
                cit.marker = self._url_to_marker[cit.url]
                unique_markers_for_finding.add(cit.marker)

            # Append markers to the end of the claim if there are citations
            if unique_markers_for_finding:
                sorted_markers = sorted(
                    list(unique_markers_for_finding), 
                    key=lambda x: int(x.strip("[]"))
                )
                marker_suffix = " ".join(sorted_markers)
                
                if not finding.claim.endswith(marker_suffix):
                    finding.claim = f"{finding.claim.strip()} {marker_suffix}"
                    
        return findings

    @property
    def all_citations(self) -> list[Citation]:
        """Return the deduplicated list of all citations."""
        return self._unique_citations

    def get_reference_list(self) -> str:
        """
        Generate a formatted reference list (APA-ish style).
        Format: [1] Title. Domain. Retrieved from URL
        """
        lines = []
        for cit in self._unique_citations:
            title = clean_title_or_fallback(cit.title, cit.url, cit.domain)
            domain = cit.domain.strip() if cit.domain else "Unknown Domain"
            
            line = f"{cit.marker} {title}. {domain}. Retrieved from {cit.url}"
            lines.append(line)
            
        return "\n".join(lines)
