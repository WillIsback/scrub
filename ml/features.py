"""Feature extraction functions for credential classification.

All functions are pure: take a string, return a float or bool.
This allows the same features to be computed in Python (training)
and Rust (inference).

Feature numbering must stay consistent between Python and Rust.
Add new features at the end only — never reorder or remove.
"""

import math
import re
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Feature catalog
# ---------------------------------------------------------------------------
# When adding features: append to ALL feature functions, update FEATURE_COUNT
# and FEATURE_NAMES. Never reorder or delete — that shifts indices.
# ---------------------------------------------------------------------------

FEATURE_COUNT = 24

FEATURE_NAMES = [
    "entropy",
    "log_len",
    "digit_ratio",
    "upper_ratio",
    "lower_ratio",
    "special_ratio",
    "max_consecutive_repeat",
    "has_base64_padding",
    "has_colon",
    "has_slash",
    "has_dot",
    "has_hyphen",
    "has_underscore",
    "has_equals",
    "is_hex_only",
    "line_log_len",
    "keyword_count",
    "is_assignment",
    "has_quotes",
    "is_source_file",
    "is_config_file",
    # 3 reserved for rule-type one-hot or future expansion
    "reserved_01",
    "reserved_02",
    "reserved_03",
]

# ---------------------------------------------------------------------------
# Value-level features
# ---------------------------------------------------------------------------


def compute_entropy(s: str) -> float:
    """Shannon entropy in bits."""
    if not s:
        return 0.0
    length = len(s)
    freq: dict[str, int] = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    entropy = 0.0
    for count in freq.values():
        p = count / length
        entropy -= p * math.log2(p)
    return entropy


def compute_log_len(s: str) -> float:
    """Natural log of string length (0.0 for empty)."""
    return math.log(len(s)) if s else 0.0


def compute_digit_ratio(s: str) -> float:
    """Proportion of [0-9] characters."""
    if not s:
        return 0.0
    return sum(1 for ch in s if ch.isdigit()) / len(s)


def compute_upper_ratio(s: str) -> float:
    """Proportion of [A-Z] characters."""
    if not s:
        return 0.0
    return sum(1 for ch in s if ch.isupper()) / len(s)


def compute_lower_ratio(s: str) -> float:
    """Proportion of [a-z] characters."""
    if not s:
        return 0.0
    return sum(1 for ch in s if ch.islower()) / len(s)


def compute_special_ratio(s: str) -> float:
    """Proportion of non-alphanumeric characters."""
    if not s:
        return 0.0
    return sum(1 for ch in s if not ch.isalnum()) / len(s)


def compute_max_consecutive_repeat(s: str) -> int:
    """Longest run of identical characters."""
    if not s:
        return 0
    max_run = 1
    current_run = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    return max_run


def has_base64_padding(s: str) -> bool:
    """Ends with '=' or '==' (base64 padding)."""
    return s.endswith("=") or s.endswith("==")


def has_colon(s: str) -> bool:
    return ":" in s


def has_slash(s: str) -> bool:
    return "/" in s


def has_dot(s: str) -> bool:
    return "." in s


def has_hyphen(s: str) -> bool:
    return "-" in s


def has_underscore(s: str) -> bool:
    return "_" in s


def has_equals(s: str) -> bool:
    return "=" in s


def is_hex_only(s: str) -> bool:
    """Only contains characters [0-9a-fA-F]."""
    if not s:
        return True
    return bool(re.fullmatch(r"[0-9a-fA-F]+", s))


# ---------------------------------------------------------------------------
# Line-level features
# ---------------------------------------------------------------------------

KEYWORDS = [
    "password", "passwd", "pwd",
    "secret",
    "token",
    "api_key", "apikey", "api",
    "auth",
    "credential", "creds",
    "login",
    "db_", "database",
    "aws", "amazon",
    "ssh",
    "-----begin",  # PEM key start (lowercased)
]


def count_keywords(line: str) -> int:
    """Count credential-related keywords in a line (case-insensitive)."""
    lower = line.lower()
    return sum(1 for kw in KEYWORDS if kw in lower)


def is_assignment(line: str) -> bool:
    """Line contains '='."""
    return "=" in line


def has_quotes(line: str) -> bool:
    """Line contains single or double quotes."""
    return "'" in line or '"' in line


# ---------------------------------------------------------------------------
# File-level features
# ---------------------------------------------------------------------------

SOURCE_EXTENSIONS = {".py", ".js", ".ts", ".rs", ".go", ".java", ".c", ".cpp",
                     ".h", ".hpp", ".rb", ".php", ".swift", ".kt", ".scala",
                     ".pl", ".pm", ".sh", ".bash", ".zsh", ".ps1", ".r"}

CONFIG_EXTENSIONS = {".env", ".cfg", ".ini", ".conf", ".toml", ".yaml", ".yml",
                     ".json", ".xml", ".properties", ".config", ".cnf"}


def _file_type_flags(filename: str) -> tuple[bool, bool]:
    """Return (is_source, is_config) based on file extension."""
    if not filename:
        return False, False
    _, _, ext = filename.rpartition(".")
    ext = "." + ext.lower() if ext else ""
    return ext in SOURCE_EXTENSIONS, ext in CONFIG_EXTENSIONS


# ---------------------------------------------------------------------------
# Aggregate feature computation
# ---------------------------------------------------------------------------

# Type alias for the feature vector: a tuple of 24 floats
FeatureVector = tuple[float, ...]


def compute_all_features(
    value: str,
    line: str,
    filename: str,
) -> np.ndarray:
    """Compute all 24 features, return as float64 numpy array.

    Parameters
    ----------
    value : str
        The matched credential value (e.g. the captured regex group).
    line : str
        The full line of source code containing the value.
    filename : str
        The file path (used only for extension detection).
    """
    is_src, is_cfg = _file_type_flags(filename)

    features = [
        compute_entropy(value),             # 0
        compute_log_len(value),             # 1
        compute_digit_ratio(value),         # 2
        compute_upper_ratio(value),         # 3
        compute_lower_ratio(value),         # 4
        compute_special_ratio(value),       # 5
        float(compute_max_consecutive_repeat(value)),  # 6
        float(has_base64_padding(value)),   # 7
        float(has_colon(value)),            # 8
        float(has_slash(value)),            # 9
        float(has_dot(value)),              # 10
        float(has_hyphen(value)),           # 11
        float(has_underscore(value)),       # 12
        float(has_equals(value)),           # 13
        float(is_hex_only(value)),          # 14
        compute_log_len(line),              # 15
        float(count_keywords(line)),        # 16
        float(is_assignment(line)),         # 17
        float(has_quotes(line)),            # 18
        float(is_src),                      # 19
        float(is_cfg),                      # 20
        0.0,                                # 21 reserved
        0.0,                                # 22 reserved
        0.0,                                # 23 reserved
    ]
    assert len(features) == FEATURE_COUNT
    return np.array(features, dtype=np.float64)
