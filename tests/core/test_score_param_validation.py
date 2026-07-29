"""Tests for the `score` parameter validation in UncertaintyQuantifier.__init__."""

import pytest
from utrace import UncertaintyQuantifier


def test_score_aps_rejected_at_construction():
    """score='aps' must raise ValueError at construction, not fail later during calibration."""
    with pytest.raises(ValueError) as exc_info:
        # type: ignore[arg-type]  # deliberately invalid: tests runtime rejection
        UncertaintyQuantifier(score='aps')
    message = str(exc_info.value)
    assert 'aps' in message
    assert 'lac' in message

def test_score_lac_matches_default():
    assert UncertaintyQuantifier(score='lac').score_ is UncertaintyQuantifier().score_

def test_score_unknown_value_rejected():
    """A clearly invalid score value must also raise ValueError, not silently fall back."""
    with pytest.raises(ValueError) as exc_info:
        # type: ignore[arg-type]  # deliberately invalid: tests runtime rejection
        UncertaintyQuantifier(score='nonexistent')
    message = str(exc_info.value)
    assert 'nonexistent' in message
    assert 'lac' in message
