"""Smoke tests: every public module imports without errors."""


def test_top_level():
    import replicant
    assert hasattr(replicant, "__version__")
    assert hasattr(replicant, "PersonalityFactory")
    assert hasattr(replicant, "PaperComparison")
    assert hasattr(replicant, "build_personality")
    assert hasattr(replicant, "sample_personalities")


def test_personalities():
    from replicant.personalities import (
        PersonalityFactory,
        PROFILES,
        POPULATION_NORMS,
        DOMAINS,
        SENTENCES,
        build_description,
        build_personality,
        sample_personalities,
        score_to_weight,
    )
    assert "extraversion" in DOMAINS
    assert "cooperative" in PROFILES
    assert "agreeableness" in POPULATION_NORMS


def test_personalities_validation():
    from replicant.personalities import (
        run_level_validation,
        run_continuous_validation,
        MINI_IPIP,
        build_mini_ipip_survey,
        score_results,
        level_from_score,
    )
    assert len(MINI_IPIP) == 20
    assert callable(run_level_validation)
    assert callable(run_continuous_validation)


def test_otree():
    from replicant.otree import (
        OTreeClient, PageData, FormField,
        LLMBot, FormController, run_bots,
        HybridSession, OTreeSession, OTreeExporter,
        parse, translate,
    )
    assert all(callable(f) for f in [
        run_bots, parse, translate,
    ])
    # OTreeExporter should be instantiable without a live server
    exporter = OTreeExporter("http://localhost:8000", rest_key="test")
    assert exporter.server_url == "http://localhost:8000"
    assert exporter.rest_key == "test"


def test_analysis_cost():
    from replicant.analysis import (
        estimate_cost, print_estimate, get_pricing, MODEL_PRICING,
    )
    in_, out_ = get_pricing("stepfun/step-3.5-flash")
    assert in_ > 0 and out_ > 0


def test_analysis_stats():
    from replicant.analysis import (
        mann_whitney, chi_square, cohen_d, sig_marker,
        compare_means, compare_proportions, print_comparison_header,
    )
    assert all(callable(f) for f in [
        mann_whitney, chi_square, cohen_d,
        compare_means, compare_proportions, print_comparison_header,
    ])


def test_analysis_comparison():
    from replicant.analysis import PaperComparison
    comp = PaperComparison("Test")
    comp.add_finding("metric", 5.0, "test")
    comp.compare("metric", 4.5)
    assert "metric" in comp.findings


def test_preflight():
    from replicant.preflight import (
        check_api_key, check_model, check_otree_server, PreflightError,
    )
    assert callable(check_api_key)
    assert callable(check_model)
    assert callable(check_otree_server)
