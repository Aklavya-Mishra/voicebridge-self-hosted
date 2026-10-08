"""Settings for live next-best-action hints."""
from app.config import Settings


def test_hints_are_off_in_code_and_on_from_env(monkeypatch):
    assert Settings().nba_enabled is False
    monkeypatch.delenv("NBA_ENABLED", raising=False)
    assert Settings.from_env().nba_enabled is True


def test_hint_settings_from_env(monkeypatch):
    monkeypatch.setenv("NBA_ENABLED", "false")
    monkeypatch.setenv("NBA_MODEL", "qwen2.5:3b")
    monkeypatch.setenv("NBA_NUM_CTX", "2048")
    monkeypatch.setenv("NBA_TIMEOUT_S", "12.5")
    monkeypatch.setenv("NBA_ABSENT_AFTER", "6")
    s = Settings.from_env()
    assert (s.nba_enabled, s.nba_model, s.nba_num_ctx, s.nba_timeout_s, s.nba_absent_after) == (
        False, "qwen2.5:3b", 2048, 12.5, 6,
    )
