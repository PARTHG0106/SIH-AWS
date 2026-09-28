"""Software-only HTTP fixtures; these bytes are never admitted as observations."""
import hashlib
import json
from unittest.mock import patch

import pytest

from awsad.data.acquisition import acquire


class Response:
    status = 200
    headers = {"Content-Length": "12", "Content-Type": "text/plain"}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def read(self, limit):
        return b"TEST FIXTURE"[:limit]

    def geturl(self):
        return "https://example.invalid/test.txt"


def test_cached_acquisition_preserves_time_and_detects_tampering(tmp_path):
    url = "https://example.invalid/test.txt"
    with patch("urllib.request.urlopen", return_value=Response()) as request:
        original = acquire(url, tmp_path)
        repeated = acquire(url, tmp_path)
        assert original == repeated
        assert request.call_count == 1
    assert original["sha256"] == hashlib.sha256(b"TEST FIXTURE").hexdigest()
    (tmp_path / original["file"]).write_bytes(b"MODIFIED")
    with pytest.raises(ValueError, match="no longer match"):
        acquire(url, tmp_path)


def test_cache_path_escape_is_rejected(tmp_path):
    url = "https://example.invalid/test.txt"
    key = hashlib.sha256(url.encode()).hexdigest()
    (tmp_path / (key + ".acquisition.json")).write_text(json.dumps({"file": "../outside.txt"}))
    with pytest.raises(ValueError, match="outside"):
        acquire(url, tmp_path)


def test_size_limit_is_enforced_before_persisting(tmp_path):
    with patch("urllib.request.urlopen", return_value=Response()):
        with pytest.raises(ValueError, match="size limit"):
            acquire("https://example.invalid/test.txt", tmp_path, max_bytes=5)
    assert list(tmp_path.iterdir()) == []
