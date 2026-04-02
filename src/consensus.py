"""
consensus-skill: Research a topic, find where sources agree and disagree.
Uses real text overlap analysis — no LLM calls needed.
"""
import re
from collections import Counter
from duckduckgo_search import DDGS

# Common stop words to exclude from analysis
STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "don't", "doesn", "didn", "won", "wouldn", "couldn", "shouldn",
    "isn", "aren", "wasn", "weren", "hasn", "haven", "hadn", "that",
    "this", "these", "those", "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them", "my", "your", "his", "its", "our",
    "their", "what", "which", "who", "whom", "and", "but", "or", "if",
    "while", "because", "until", "about", "also", "like", "new", "one",
    "two", "first", "get", "got", "make", "made", "many", "much"
}


def _extract_keywords(text: str, top_n: int = 20) -> list[str]:
    """Extract keywords from text after removing stop words and normalizing."""
    if not text:
        return []
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    filtered = [w for w in words if w not in STOP_WORDS]
    return [word for word, _ in Counter(filtered).most_common(top_n)]


def _jaccard_similarity(set_a: set, set_b: set) -> float:
    """Calculate Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def search_topic(topic: str, max_results: int = 5) -> list[dict]:
    """Search the web for a topic and return source dictionaries."""
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(topic, max_results=max_results):
            results.append({
                "title": r.get("title", ""),
                "body": r.get("body", ""),
                "href": r.get("href", ""),
            })
    return results


def analyze_consensus(sources: list[dict]) -> dict:
    """
    Analyze multiple sources for consensus and conflict points.

    Works by:
    1. Extracting keywords from each source body
    2. Finding shared keywords (consensus topics)
    3. Finding source-specific keywords (potential conflicts/divergence)
    4. Calculating overall agreement via average pairwise Jaccard similarity
    """
    if not sources:
        return {"consensus_points": [], "conflict_points": [], "overall_agreement_pct": 0.0}

    if len(sources) == 1:
        return {
            "consensus_points": [f"Single source provided: {sources[0].get('title', '')}"],
            "conflict_points": ["Need multiple sources to identify conflicts"],
            "overall_agreement_pct": 100.0,
        }

    # Extract keywords per source
    source_keywords = {}
    source_bodies = {}
    for i, src in enumerate(sources):
        body = src.get("body", "")
        source_bodies[i] = body
        source_keywords[i] = set(_extract_keywords(body))

    # Find consensus: keywords appearing in 50%+ of sources
    all_keywords = set()
    for kw_set in source_keywords.values():
        all_keywords.update(kw_set)

    keyword_frequency = Counter()
    for kw_set in source_keywords.values():
        for kw in kw_set:
            keyword_frequency[kw] += 1

    consensus_threshold = max(2, len(sources) * 0.5)
    consensus_keywords = sorted(
        [kw for kw, freq in keyword_frequency.items() if freq >= consensus_threshold],
        key=lambda x: keyword_frequency[x],
        reverse=True,
    )

    # Group consensus keywords into meaningful phrases from most frequent
    consensus_points = []
    for kw in consensus_keywords[:5]:
        # Find a source that mentions this keyword and extract a short context snippet
        for src in sources:
            body = src.get("body", "")
            idx = body.lower().find(kw)
            if idx >= 0:
                start = max(0, idx - 30)
                end = min(len(body), idx + len(kw) + 40)
                snippet = body[start:end].strip()
                if snippet and len(snippet) > 10:
                    count = keyword_frequency[kw]
                    consensus_points.append(f"[{count}/{len(sources)} sources] ...{snippet}...")
                    break

    # Find conflict/divergence: keywords unique to one source
    conflict_points = []
    for i, src in enumerate(sources):
        unique_kw = source_keywords[i] - set().union(
            *(source_keywords[j] for j in source_keywords if j != i)
        )
        top_unique = sorted(unique_kw, key=lambda x: keyword_frequency[x])[:3]
        if top_unique:
            title = src.get("title", f"Source {i+1}")
            conflict_points.append(
                f"'{title}' focuses on unique angles: {', '.join(top_unique)}"
            )

    # Overall agreement: average pairwise Jaccard similarity
    similarities = []
    ids = list(source_keywords.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            sim = _jaccard_similarity(source_keywords[ids[i]], source_keywords[ids[j]])
            similarities.append(sim)

    overall_pct = round((sum(similarities) / len(similarities)) * 100, 1) if similarities else 0.0

    return {
        "consensus_points": consensus_points if consensus_points else ["No strong consensus detected across sources"],
        "conflict_points": conflict_points if conflict_points else ["Sources align closely — no major divergence found"],
        "overall_agreement_pct": overall_pct,
    }


def format_report(topic: str, analysis: dict) -> str:
    """Format the consensus analysis as a clean CLI report."""
    lines = [
        f"{'=' * 50}",
        f"  Consensus Report: {topic}",
        f"{'=' * 50}",
        f"  Overall Agreement: {analysis['overall_agreement_pct']}%",
        f"{'=' * 50}",
        "",
        "  CONSENSUS (shared across sources):",
    ]
    for p in analysis["consensus_points"]:
        lines.append(f"  • {p}")

    lines.append("")
    lines.append("  CONFLICT/DIVERGENCE (source-specific angles):")
    for p in analysis["conflict_points"]:
        lines.append(f"  • {p}")

    lines.append("")
    lines.append(f"{'=' * 50}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])
        sources = search_topic(topic)
        if not sources:
            print(f"No results found for: {topic}")
            sys.exit(1)
        analysis = analyze_consensus(sources)
        print(format_report(topic, analysis))
    else:
        print("Usage: python -m src.consensus <topic>")
        print("Example: python -m src.consensus 'AI regulation 2026'")
