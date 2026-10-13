"""Tests for run-provenance helpers."""

import pytest

from bhc_experiment import provenance
from bhc_experiment.provenance import read_launch_record


class TestReadLaunchRecord:
    """Parse the KEY=VALUE launch record serve_local_llm.sh writes per server start."""

    def test_reads_keyvalue_record(self, tmp_path):
        """A record file for the alias parses into its fields."""
        (tmp_path / "launch-record-gemma-4-e4b.txt").write_text(
            "flash_attn=off\n"
            "ctx_size=131072\n"
            "n_gpu_layers=999\n"
            "cache_type_k=f16\n"
            "cache_type_v=f16\n"
            "ubatch_size=2048\n"
            "model_file=google_gemma-4-E4B-it-Q4_K_M.gguf\n"
            "server_build=9535\n"
        )
        record = read_launch_record(tmp_path, "gemma-4-e4b")
        assert record == {
            "flash_attn": "off",
            "ctx_size": "131072",
            "n_gpu_layers": "999",
            "cache_type_k": "f16",
            "cache_type_v": "f16",
            "ubatch_size": "2048",
            "model_file": "google_gemma-4-E4B-it-Q4_K_M.gguf",
            "server_build": "9535",
        }

    def test_skips_blank_and_comment_and_malformed_lines(self, tmp_path):
        """Blank lines, comments, and lines without '=' are skipped without error."""
        (tmp_path / "launch-record-llama-3.2-3b.txt").write_text(
            "# written by serve_local_llm.sh\n"
            "flash_attn=on\n"
            "\n"
            "this line has no equals sign\n"
            "ctx_size=131072\n"
        )
        record = read_launch_record(tmp_path, "llama-3.2-3b")
        assert record == {"flash_attn": "on", "ctx_size": "131072"}


class TestGitCommitHash:
    def test_is_the_full_hash(self):
        # A short hash can become ambiguous as the history grows, and the
        # commit is the only provenance a run records.
        commit = provenance.git_commit_hash()
        assert commit is not None
        assert len(commit) == 40


class TestRequireCleanTree:
    def test_dirty_tree_exits(self, monkeypatch):
        monkeypatch.delenv("BHC_SKIP_GIT_CHECK", raising=False)
        monkeypatch.setattr(provenance, "_git_status_porcelain", lambda: " M src/x.py")
        with pytest.raises(SystemExit, match="Uncommitted changes"):
            provenance.require_clean_tree()

    def test_clean_tree_passes(self, monkeypatch):
        monkeypatch.delenv("BHC_SKIP_GIT_CHECK", raising=False)
        monkeypatch.setattr(provenance, "_git_status_porcelain", lambda: "")
        provenance.require_clean_tree()

    def test_bypass_env_skips_the_check(self, monkeypatch):
        monkeypatch.setenv("BHC_SKIP_GIT_CHECK", "1")
        monkeypatch.setattr(provenance, "_git_status_porcelain", lambda: " M src/x.py")
        provenance.require_clean_tree()
