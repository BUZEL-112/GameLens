"""
Parity Test: core_ml.features.build_user_vector

Verifies that the shared build_user_vector() function produces the exact same
output as the original _get_user_features() logic that was previously inlined
inside training/pipeline.py stage3_process_users.

If someone modifies the shared function in a way that breaks mathematical parity,
this test will catch it before the change reaches production — where degraded
recommendations would be the only signal.

Run with:
    cd game_recommender
    python -m pytest tests/test_feature_parity.py -v
"""

import numpy as np
import pandas as pd
import pytest

from core_ml.features import build_user_vector, normalize_user_matrix, safe_parse


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def toy_item_content_matrix() -> np.ndarray:
    """5 items, 4-dim content vectors (simplified version of the 320-dim matrix)."""
    np.random.seed(42)
    return np.random.rand(5, 4).astype(np.float32)


@pytest.fixture
def toy_item_vocab() -> dict[str, int]:
    return {
        "Counter-Strike": 0,
        "Dota 2": 1,
        "Portal": 2,
        "Team Fortress 2": 3,
        "Half-Life": 4,
    }


@pytest.fixture
def toy_history() -> pd.DataFrame:
    return pd.DataFrame({
        "item_name": ["Counter-Strike", "Dota 2", "Portal"],
        "interaction": [3.5, 2.0, 1.0],
    })


# ── Reference implementation (original inline logic from pipeline.py) ─────────

def _reference_get_user_features(
    user_history: pd.DataFrame,
    item_content_matrix: np.ndarray,
    item_vocab: dict[str, int],
) -> np.ndarray:
    """
    Verbatim copy of the original _get_user_features() inner function that was
    inlined in stage3_process_users before the refactor.

    This function is the ground truth for the parity test. It must not be
    modified — if the shared function needs to change, update this reference
    to match and document why the behavior changed.
    """
    feat_dim = item_content_matrix.shape[1]

    valid = [
        (item_vocab[name], weight)
        for name, weight in zip(user_history["item_name"], user_history["interaction"])
        if name in item_vocab and weight > 0
    ]
    if not valid:
        return np.zeros(feat_dim + 2, dtype=np.float32)

    indices, weights = zip(*valid)
    weights_arr = np.array(weights, dtype=np.float32)
    vecs = item_content_matrix[list(indices)]

    history_vec = (vecs * weights_arr[:, None]).sum(axis=0) / (weights_arr.sum() + 1e-9)
    total_interaction = float(weights_arr.sum())
    num_games = float(len(valid))

    return np.concatenate([history_vec, [total_interaction, num_games]]).astype(np.float32)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestBuildUserVectorParity:

    def test_matches_reference_for_normal_history(
        self, toy_history, toy_item_content_matrix, toy_item_vocab
    ):
        """Core parity check: shared function must match the original inline logic."""
        reference = _reference_get_user_features(
            toy_history, toy_item_content_matrix, toy_item_vocab
        )
        result = build_user_vector(
            toy_history, toy_item_content_matrix, toy_item_vocab
        )
        np.testing.assert_allclose(result, reference, rtol=1e-6,
            err_msg="build_user_vector output diverged from the reference implementation")

    def test_output_shape(self, toy_history, toy_item_content_matrix, toy_item_vocab):
        """Output must be (content_dim + 2,) = (4 + 2,) = (6,) for our 4-dim fixture."""
        result = build_user_vector(toy_history, toy_item_content_matrix, toy_item_vocab)
        expected_dim = toy_item_content_matrix.shape[1] + 2
        assert result.shape == (expected_dim,), (
            f"Expected shape ({expected_dim},), got {result.shape}"
        )

    def test_output_dtype(self, toy_history, toy_item_content_matrix, toy_item_vocab):
        """Output must be float32 to match the model tower input dtype."""
        result = build_user_vector(toy_history, toy_item_content_matrix, toy_item_vocab)
        assert result.dtype == np.float32

    def test_empty_history_returns_zeros(self, toy_item_content_matrix, toy_item_vocab):
        """A user with no valid interactions must return a zero vector, not crash."""
        empty = pd.DataFrame({"item_name": [], "interaction": []})
        result = build_user_vector(empty, toy_item_content_matrix, toy_item_vocab)
        expected_dim = toy_item_content_matrix.shape[1] + 2
        assert result.shape == (expected_dim,)
        np.testing.assert_array_equal(result, np.zeros(expected_dim, dtype=np.float32))

    def test_unknown_items_are_ignored(self, toy_item_content_matrix, toy_item_vocab):
        """Items not in item_vocab must be silently ignored, not cause a KeyError."""
        history_with_unknown = pd.DataFrame({
            "item_name": ["Counter-Strike", "Unknown Game XYZ"],
            "interaction": [2.0, 5.0],
        })
        # Should not raise; Unknown Game XYZ is filtered
        result = build_user_vector(
            history_with_unknown, toy_item_content_matrix, toy_item_vocab
        )
        # Result should equal what we get with only Counter-Strike
        history_known_only = pd.DataFrame({
            "item_name": ["Counter-Strike"],
            "interaction": [2.0],
        })
        expected = build_user_vector(
            history_known_only, toy_item_content_matrix, toy_item_vocab
        )
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_zero_interaction_items_are_ignored(
        self, toy_item_content_matrix, toy_item_vocab
    ):
        """Items with interaction weight <= 0 must not contribute to the vector."""
        history_with_zero = pd.DataFrame({
            "item_name": ["Counter-Strike", "Dota 2"],
            "interaction": [2.0, 0.0],
        })
        history_positive_only = pd.DataFrame({
            "item_name": ["Counter-Strike"],
            "interaction": [2.0],
        })
        result = build_user_vector(
            history_with_zero, toy_item_content_matrix, toy_item_vocab
        )
        expected = build_user_vector(
            history_positive_only, toy_item_content_matrix, toy_item_vocab
        )
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_statistical_tail_values(
        self, toy_history, toy_item_content_matrix, toy_item_vocab
    ):
        """The last two dimensions must encode total_interaction and num_games."""
        result = build_user_vector(toy_history, toy_item_content_matrix, toy_item_vocab)
        # toy_history has interactions [3.5, 2.0, 1.0]
        expected_total = 3.5 + 2.0 + 1.0
        expected_ngames = 3.0
        assert result[-2] == pytest.approx(expected_total, rel=1e-5)
        assert result[-1] == pytest.approx(expected_ngames, rel=1e-5)


class TestNormalizeUserMatrix:

    def test_normalizes_last_two_dimensions(self, toy_item_content_matrix, toy_item_vocab):
        """After normalization, the max of each statistical dimension should be ~1.0."""
        raw = {
            "user_a": np.array([0.1, 0.2, 0.3, 0.4, 10.0, 5.0], dtype=np.float32),
            "user_b": np.array([0.5, 0.1, 0.2, 0.8, 5.0, 2.0], dtype=np.float32),
        }
        normalized = normalize_user_matrix(raw)
        max_total = max(v[-2] for v in normalized.values())
        max_ngames = max(v[-1] for v in normalized.values())
        assert max_total == pytest.approx(1.0, rel=1e-5)
        assert max_ngames == pytest.approx(1.0, rel=1e-5)

    def test_content_dims_unchanged(self):
        """The first N content dimensions must not be touched by normalization."""
        raw = {
            "user_a": np.array([0.1, 0.2, 0.3, 0.4, 10.0, 5.0], dtype=np.float32),
        }
        normalized = normalize_user_matrix(raw)
        np.testing.assert_allclose(
            normalized["user_a"][:4],
            raw["user_a"][:4],
            rtol=1e-6,
        )

    def test_empty_dict_returns_empty(self):
        assert normalize_user_matrix({}) == {}


class TestSafeParse:

    def test_list_passthrough(self):
        assert safe_parse(["Action", "RPG"]) == ["Action", "RPG"]

    def test_stringified_list(self):
        assert safe_parse("['Action', 'RPG']") == ["Action", "RPG"]

    def test_nan_returns_empty(self):
        assert safe_parse(float("nan")) == []

    def test_malformed_string_returns_empty(self):
        assert safe_parse("{not: valid}") == []

    def test_html_entity_normalization(self):
        result = safe_parse("['Action &amp; Adventure']")
        assert result == ["Action & Adventure"]

    def test_plain_string_fallback(self):
        # A plain string that can't be evaluated as a list becomes a single-element list
        result = safe_parse("Action")
        assert result == ["Action"]
