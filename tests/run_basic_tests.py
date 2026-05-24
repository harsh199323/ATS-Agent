from ats_orchestrator import extract_keywords, compute_match_scores


def run():
    print("Running basic tests...")
    text = "Python and data engineering with ML and AI for production."
    result = extract_keywords(text, n=10)
    assert "and" not in result["keywords"]
    assert "python" in result["keywords"]

    jd = "We need Python, Docker, and FastAPI experience with 3+ years."
    resume = "Built Python services with FastAPI. 4 years experience."
    scores = compute_match_scores(jd, resume)
    assert "overall_score" in scores
    assert "missing_skills" in scores
    assert isinstance(scores["top_matches"], list)
    print("All basic tests passed.")


if __name__ == "__main__":
    run()
