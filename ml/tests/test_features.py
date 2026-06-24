"""Tests for feature extraction functions."""

import math
import numpy as np
from ml.features import (
    compute_entropy,
    compute_log_len,
    compute_digit_ratio,
    compute_upper_ratio,
    compute_lower_ratio,
    compute_special_ratio,
    compute_max_consecutive_repeat,
    has_base64_padding,
    has_colon,
    has_slash,
    has_dot,
    has_hyphen,
    has_underscore,
    has_equals,
    is_hex_only,
    count_keywords,
    is_assignment,
    has_quotes,
    compute_all_features,
    FEATURE_COUNT,
    FEATURE_NAMES,
    RULE_TYPE_NAMES,
    rule_type_onehot,
)


class TestEntropy:
    """Shannon entropy of a string."""

    def test_uniform_string_has_zero_entropy(self):
        assert compute_entropy("aaaaa") == 0.0

    def test_empty_string_has_zero_entropy(self):
        assert compute_entropy("") == 0.0

    def test_maximum_entropy_for_two_chars(self):
        result = compute_entropy("ab")
        assert abs(result - 1.0) < 1e-10

    def test_high_entropy_random_string(self):
        result = compute_entropy("aB3$xY9#qR!z")
        assert result > 3.0


class TestLogLen:
    """Log of string length."""

    def test_log_len_positive(self):
        result = compute_log_len("hello")
        assert result == math.log(len("hello"))

    def test_log_len_zero_len(self):
        result = compute_log_len("")
        assert result == 0.0

    def test_log_len_grows_slowly(self):
        short = compute_log_len("abc")
        long_ = compute_log_len("abcdefghijklmnopqrstuvwxyz")
        assert long_ > short


class TestDigitRatio:
    """Proportion of [0-9] characters."""

    def test_all_digits(self):
        assert compute_digit_ratio("12345") == 1.0

    def test_no_digits(self):
        assert compute_digit_ratio("abcde") == 0.0

    def test_half_digits(self):
        assert compute_digit_ratio("a1b2c") == 0.4

    def test_empty(self):
        assert compute_digit_ratio("") == 0.0


class TestUpperRatio:
    """Proportion of [A-Z] characters."""

    def test_all_upper(self):
        assert compute_upper_ratio("ABCDE") == 1.0

    def test_no_upper(self):
        assert compute_upper_ratio("abcde") == 0.0

    def test_mixed(self):
        assert compute_upper_ratio("AbCdE") == 0.6

    def test_empty(self):
        assert compute_upper_ratio("") == 0.0


class TestLowerRatio:
    """Proportion of [a-z] characters."""

    def test_all_lower(self):
        assert compute_lower_ratio("abcde") == 1.0

    def test_no_lower(self):
        assert compute_lower_ratio("ABCDE") == 0.0

    def test_mixed(self):
        assert compute_lower_ratio("AbCdE") == 0.4  # 'b', 'd' = 2/5

    def test_empty(self):
        assert compute_lower_ratio("") == 0.0


class TestSpecialRatio:
    """Proportion of non-alphanumeric characters."""

    def test_all_special(self):
        assert compute_special_ratio("!@#$%") == 1.0

    def test_no_special(self):
        assert compute_special_ratio("abc123") == 0.0

    def test_mixed(self):
        result = compute_special_ratio("a!b@c")
        assert abs(result - 0.4) < 1e-10

    def test_empty(self):
        assert compute_special_ratio("") == 0.0


class TestMaxConsecutiveRepeat:
    """Longest run of identical characters."""

    def test_no_repeats(self):
        assert compute_max_consecutive_repeat("abcdef") == 1

    def test_single_run(self):
        assert compute_max_consecutive_repeat("abbbbc") == 4

    def test_multiple_runs(self):
        assert compute_max_consecutive_repeat("aabbbccccc") == 5

    def test_empty(self):
        assert compute_max_consecutive_repeat("") == 0

    def test_all_same(self):
        assert compute_max_consecutive_repeat("xxxxx") == 5


class TestHasBase64Padding:
    """Ends with '=' or '=='."""

    def test_no_padding(self):
        assert has_base64_padding("abc123") is False

    def test_single_padding(self):
        assert has_base64_padding("abc=") is True

    def test_double_padding(self):
        assert has_base64_padding("abc==") is True

    def test_empty(self):
        assert has_base64_padding("") is False


class TestBinaryFlags:
    """Simple presence checks."""

    def test_has_colon(self):
        assert has_colon("key:value") is True
        assert has_colon("no colon") is False

    def test_has_slash(self):
        assert has_slash("http://example.com") is True
        assert has_slash("no slash") is False

    def test_has_dot(self):
        assert has_dot("example.com") is True
        assert has_dot("nope") is False

    def test_has_hyphen(self):
        assert has_hyphen("some-token") is True
        assert has_hyphen("nope") is False

    def test_has_underscore(self):
        assert has_underscore("my_key") is True
        assert has_underscore("nope") is False

    def test_has_equals(self):
        assert has_equals("x=1") is True
        assert has_equals("nope") is False


class TestIsHexOnly:
    """Only contains [0-9a-fA-F]."""

    def test_hex_only(self):
        assert is_hex_only("DEADbeef1234") is True

    def test_not_hex(self):
        assert is_hex_only("DEADbeef1234z") is False

    def test_empty(self):
        assert is_hex_only("") is True  # vacuously true

    def test_with_dashes(self):
        assert is_hex_only("DEAD-BEEF") is False  # dash not hex


class TestKeywordCount:
    """Count credential-related keywords in a line."""

    def test_no_keywords(self):
        assert count_keywords("x = get_data()") == 0

    def test_password_keyword(self):
        assert count_keywords("password = 'secret'") == 2  # 'password' + 'secret'

    def test_multiple_keywords(self):
        assert count_keywords("db_password = api_token") == 4  # 'db_', 'password', 'api', 'token'

    def test_apikey_compound(self):
        assert count_keywords("apikey = 'xxx'") >= 1


class TestIsAssignment:
    """Line contains '=' assignment."""

    def test_assignment(self):
        assert is_assignment("x = 1") is True

    def test_no_assignment(self):
        assert is_assignment("print('hello')") is False

    def test_equals_in_string(self):
        assert is_assignment("'=' inside") is True  # still has =


class TestHasQuotes:
    """Value appears in quotes."""

    def test_double_quotes(self):
        assert has_quotes('"secret"') is True

    def test_single_quotes(self):
        assert has_quotes("'secret'") is True

    def test_no_quotes(self):
        assert has_quotes("hello") is False

    def test_mixed(self):
        assert has_quotes('mixed"') is True


class TestComputeAllFeatures:
    """Integration: compute_all_features returns consistent array."""

    def test_returns_correct_length(self):
        features = compute_all_features("some_value", "password = 'abc123'", ".env")
        assert len(features) == FEATURE_COUNT

    def test_returns_numpy_array(self):
        features = compute_all_features("abc123", "x=abc123", ".py")
        assert isinstance(features, np.ndarray)

    def test_feature_count_matches_names(self):
        assert FEATURE_COUNT == len(FEATURE_NAMES)

    def test_different_inputs_different_features(self):
        a = compute_all_features("12345", "x=12345", ".py")
        b = compute_all_features("abcde", "x=abcde", ".py")
        # Different values should produce different feature vectors
        assert not np.array_equal(a, b)

    def test_entropy_feature_is_first(self):
        features = compute_all_features("aaaaa", "x = 'aaaaa'", ".py")
        assert features[0] == 0.0  # entropy of uniform string

    def test_empty_value(self):
        features = compute_all_features("", "", "")
        assert len(features) == FEATURE_COUNT

    def test_rule_type_api_key_onehot(self):
        hot = rule_type_onehot("api_key")
        assert len(hot) == 8
        assert hot[0] == 1.0
        assert all(v == 0.0 for i, v in enumerate(hot) if i != 0)

    def test_rule_type_password_onehot(self):
        hot = rule_type_onehot("password")
        assert hot[2] == 1.0

    def test_rule_type_empty_onehot(self):
        hot = rule_type_onehot("")
        assert all(v == 0.0 for v in hot)

    def test_rule_type_unknown_onehot(self):
        hot = rule_type_onehot("bogus_type")
        assert all(v == 0.0 for v in hot)

    def test_rule_type_in_all_features(self):
        feats = compute_all_features("abc123", "x=abc123", ".py", "api_key")
        assert feats[21] == 1.0  # api_key at index 21
        assert feats[22] == 0.0  # token
        assert feats[23] == 0.0  # password

    def test_rule_type_auth_in_all_features(self):
        feats = compute_all_features("abc123", "x=abc123", ".py", "auth")
        assert feats[21] == 0.0  # api_key
        assert feats[24] == 1.0  # auth at index 24

    def test_rule_type_default_is_empty(self):
        """Default rule_type='' should produce all zeros."""
        feats = compute_all_features("abc123", "x=abc123", ".py")
        for i in range(21, 29):
            assert feats[i] == 0.0, f"Expected feats[{i}] == 0, got {feats[i]}"
