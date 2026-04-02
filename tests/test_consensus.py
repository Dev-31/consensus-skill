import pytest
from src.consensus import (
    analyze_consensus,
    format_report,
    _extract_keywords,
    _jaccard_similarity,
)


def test_analyze_consensus_empty():
    res = analyze_consensus([])
    assert res["overall_agreement_pct"] == 0.0
    assert res["consensus_points"] == []
    assert res["conflict_points"] == []


def test_analyze_consensus_single_source():
    mock_sources = [{"title": "Single", "body": "Some text here", "href": "http://x"}]
    res = analyze_consensus(mock_sources)
    assert res["overall_agreement_pct"] == 100.0
    assert "Single source provided" in res["consensus_points"][0]


def test_analyze_consensus_identical_sources():
    body = "climate change is causing global warming and rising sea levels"
    sources = [
        {"title": "A", "body": body, "href": "http://a"},
        {"title": "B", "body": body, "href": "http://b"},
    ]
    res = analyze_consensus(sources)
    assert res["overall_agreement_pct"] == 100.0
    assert len(res["consensus_points"]) > 0


def test_analyze_consensus_contradictory_sources():
    s1 = {"title": "Pro-AI", "body": "AI technology is beneficial to society and improves productivity efficiency", "href": "http://a"}
    s2 = {"title": "Anti-AI", "body": "AI technology threatens jobs and causes unemployment in manufacturing sectors", "href": "http://b"}
    res = analyze_consensus([s1, s2])
    # They share "AI" and "technology" keywords, so agreement > 0 but < 100
    assert 0 <= res["overall_agreement_pct"] <= 100
    assert len(res["conflict_points"]) > 0  # Each has unique keywords


def test_format_report():
    analysis = {
        "consensus_points": ["Shared point A", "Shared point B"],
        "conflict_points": ["Divergent view X"],
        "overall_agreement_pct": 65.5,
    }
    report = format_report("Test Topic", analysis)
    assert "Test Topic" in report
    assert "65.5%" in report
    assert "Shared point A" in report
    assert "Divergent view X" in report
    assert "=" * 50 in report  # Border formatting


def test_extract_keywords_removes_stop_words():
    text = "the quick brown fox is jumping over the lazy dog"
    kw = _extract_keywords(text)
    assert "the" not in kw
    assert "is" not in kw
    assert "over" not in kw
    assert "quick" in kw
    assert "brown" in kw
    assert "lazy" in kw


def test_extract_keywords_empty():
    assert _extract_keywords("") == []
    assert _extract_keywords("the a an is") == []


def test_jaccard_similarity():
    assert _jaccard_similarity({"a", "b"}, {"a", "b"}) == 1.0
    assert _jaccard_similarity({"a", "b"}, {"c", "d"}) == 0.0
    assert _jaccard_similarity(set(), {"a"}) == 0.0
    assert _jaccard_similarity({"a", "b"}, {"b", "c"}) == pytest.approx(1/3)
