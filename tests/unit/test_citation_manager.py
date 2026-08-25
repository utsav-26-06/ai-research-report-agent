"""
Unit tests for CitationManager (TASK-014).
"""

from app.generation.citation_manager import CitationManager
from app.models.report import Citation, Finding


def _make_citation(url: str, title: str = "Test", domain: str = "test.com") -> Citation:
    return Citation(
        marker="[x]",
        source_id="src1",
        chunk_id="chk1",
        url=url,
        title=title,
        domain=domain,
        excerpt="text"
    )


def test_deduplication():
    manager = CitationManager()
    
    # Same URL should get same marker
    finding1 = Finding(claim="Claim 1", citations=[_make_citation("http://a.com")])
    finding2 = Finding(claim="Claim 2", citations=[_make_citation("http://a.com")])
    finding3 = Finding(claim="Claim 3", citations=[_make_citation("http://b.com")])
    
    findings = manager.process_findings([finding1, finding2, finding3])
    
    # 2 unique citations overall
    assert len(manager.all_citations) == 2
    
    # Check markers
    assert findings[0].citations[0].marker == "[1]"
    assert findings[1].citations[0].marker == "[1]"
    assert findings[2].citations[0].marker == "[2]"
    
    # Check claims are updated with markers
    assert findings[0].claim == "Claim 1 [1]"
    assert findings[1].claim == "Claim 2 [1]"
    assert findings[2].claim == "Claim 3 [2]"


def test_multiple_citations_in_one_finding():
    manager = CitationManager()
    
    finding = Finding(
        claim="Multi claim",
        citations=[
            _make_citation("http://b.com"),
            _make_citation("http://a.com")
        ]
    )
    
    manager.process_findings([finding])
    
    assert finding.citations[0].marker == "[1]"
    assert finding.citations[1].marker == "[2]"
    # Sorted order in the claim
    assert finding.claim == "Multi claim [1] [2]"


def test_reference_list_format():
    manager = CitationManager()
    finding = Finding(
        claim="Claim text longer than 5 chars",
        citations=[
            _make_citation("http://test.com", title="My Doc", domain="example.org")
        ]
    )
    
    manager.process_findings([finding])
    
    ref_list = manager.get_reference_list()
    assert ref_list == "[1] My Doc. example.org. Retrieved from http://test.com"


def test_empty_citations():
    manager = CitationManager()
    finding = Finding(claim="No evidence", citations=[])
    manager.process_findings([finding])
    
    assert finding.claim == "No evidence"
    assert len(manager.all_citations) == 0
    assert manager.get_reference_list() == ""

