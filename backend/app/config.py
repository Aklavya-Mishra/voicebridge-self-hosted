"""Runtime settings, read from environment variables (and an optional .env file)."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

DEV_JWT_SECRET = "dev-secret-change-me"


def _bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    jwt_secret: str = DEV_JWT_SECRET
    jwt_ttl_seconds: int = 12 * 60 * 60
    cors_origins: tuple[str, ...] = ("http://localhost:5173",)
    max_participants: int = 2
    db_path: str | None = None  # SQLite file for saved calls; None keeps everything in memory

    sample_rate: int = 16000
    vad_provider: str = "energy"  # energy | silero (torch package) | sherpa (Silero model through sherpa-onnx, no torch)
    vad_energy_threshold_dbfs: float = -45.0
    end_silence_ms: int = 3000  # silence that ends a chat bubble (Deepgram and local)
    max_utterance_ms: int = 15000
    min_speech_ms: int = 250
    preroll_ms: int = 200
    partials_enabled: bool = True
    partial_interval_ms: int = 1500

    stt_provider: str = "whisper"  # deepgram (cloud) | sherpa (local Parakeet) | whisper (local faster-whisper) | mock
    stt_fallback: str = "whisper"  # deepgram only: local provider used when Deepgram is unreachable
    stt_language: str | None = None
    stt_concurrency: int = 4
    deepgram_api_key: str | None = None
    deepgram_model: str = "nova-3"
    deepgram_language: str = "multi"  # multi = English + Hindi (and other languages) code-switching; or en, hi...
    deepgram_url: str = "wss://api.deepgram.com/v1/listen"
    deepgram_endpointing_ms: int = 300  # silence that ends a sentence
    deepgram_connect_timeout_s: float = 3.0
    gemini_api_key: str | None = None
    # End-of-call analysis (summary + PII). Provider order: Gemini (GEMINI_API_KEY), Anthropic
    # (ANTHROPIC_API_KEY), then the local model through Ollama, which needs no key.
    call_analysis: bool = False  # from_env turns it on
    pii_masking: bool = True  # mask long numbers and emails as each line is finalised
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-opus-5-5"
    llm_model: str = "qwen2.5:7b"  # local Ollama model
    ollama_url: str = "http://127.0.0.1:11434"
    # How long Ollama keeps the model in memory after a request (Ollama's default is 5m). Unloading also drops the
    # cached hint prompt, which takes ~20 s on CPU to read again.
    ollama_keep_alive: str = "60m"
    llm_num_ctx: int = 16384  # tokens of context: about an hour of call
    llm_cpu_only: bool = True
    gemini_model: str = "gemini-2.5-flash-lite"
    # Live next-best-action hints (app.nba): local Ollama only, never a cloud model.
    nba_enabled: bool = False  # from_env turns it on
    nba_model: str = "qwen2.5:7b"  # set from the CPU benchmark (tools/bench_nba.py)
    # Same context as llm_num_ctx by default: one model loaded at two sizes makes Ollama reload it on every switch, and
    # Ollama 0.40.4 was seen hanging a 16384-token summary request forever while the 4096-token hint model was loaded.
    nba_num_ctx: int = 16384
    nba_context_lines: int = 8  # transcript lines sent per turn
    # A slower turn is dropped. It includes waiting behind a warm-up (app.nba.service.warm, ~20 s on CPU).
    nba_timeout_s: float = 60.0
    nba_phrase_match: int = 88  # fuzzy score (0-100) for phrase rules
    nba_dedupe: int = 90  # similarity above which a new hint repeats an earlier one
    nba_absent_after: int = 4  # customer lines before "fact missing" rules may fire
    # Jira tickets drafted from the call (app.tickets): on only when URL, email, token and project key are set.
    jira_url: str | None = None  # e.g. https://yourcompany.atlassian.net
    jira_email: str | None = None
    jira_api_token: str | None = None
    jira_project_key: str | None = None  # e.g. SCRUM
    jira_issue_type: str = "Task"
    # sherpa-onnx (local, CPU, English): model files from scripts/download_models.py, under sherpa_model_dir.
    sherpa_model_dir: str = "models"
    sherpa_asr_model: str = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"  # folder with encoder/decoder/joiner/tokens
    sherpa_num_threads: int = 2  # per transcription; STT_CONCURRENCY of them run at once
    sherpa_vad_model: str = "silero_vad.onnx"
    sherpa_segmentation_model: str = "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"  # in-room: who spoke when
    sherpa_embedding_model: str = "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx"  # in-room: voice prints (CAM++ mixed voices up in tests)
    # In-room live: voice-print similarity (0-1) at or above which a line joins a known voice; 0.4-0.6 all labelled the
    # tools/make_test_audio.py meeting right (same person >= 0.74, two people <= 0.23 with ERes2Net).
    sherpa_speaker_threshold: float = 0.5
    whisper_model: str = "base"  # tiny | base | small | medium | large-v3 (bigger = more accurate, slower)
    whisper_device: str = "cpu"  # cpu | cuda
    whisper_compute_type: str = "int8"  # int8 on CPU; float16 on GPU
    whisper_beam_size: int = 1
    whisper_cpu_threads: int = 1  # per worker; 1 measured fastest per call on CPU
    whisper_num_workers: int = 2  # parallel transcriptions

    max_import_bytes: int = 100 * 1024 * 1024
    max_chunk_bytes: int = 64000

    @property
    def tickets_enabled(self) -> bool:
        return bool(self.jira_url and self.jira_email and self.jira_api_token and self.jira_project_key)

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.getenv
        return cls(
            jwt_secret=env("JWT_SECRET", cls.jwt_secret),
            jwt_ttl_seconds=int(env("JWT_TTL_SECONDS", cls.jwt_ttl_seconds)),
            cors_origins=tuple(
                o.strip() for o in env("CORS_ORIGINS", ",".join(cls.cors_origins)).split(",") if o.strip()
            ),
            max_participants=int(env("MAX_PARTICIPANTS", cls.max_participants)),
            db_path=env("DB_PATH", "data/voicebridge.db") or None,
            vad_provider=env("VAD_PROVIDER", cls.vad_provider).lower(),
            vad_energy_threshold_dbfs=float(env("VAD_ENERGY_THRESHOLD_DBFS", cls.vad_energy_threshold_dbfs)),
            end_silence_ms=int(env("END_SILENCE_MS", cls.end_silence_ms)),
            max_utterance_ms=int(env("MAX_UTTERANCE_MS", cls.max_utterance_ms)),
            min_speech_ms=int(env("MIN_SPEECH_MS", cls.min_speech_ms)),
            preroll_ms=int(env("PREROLL_MS", cls.preroll_ms)),
            partials_enabled=_bool(env("PARTIALS_ENABLED", str(cls.partials_enabled))),
            partial_interval_ms=int(env("PARTIAL_INTERVAL_MS", cls.partial_interval_ms)),
            stt_provider=env("STT_PROVIDER", cls.stt_provider).lower(),
            stt_fallback=env("STT_FALLBACK", cls.stt_fallback).lower(),
            stt_language=env("STT_LANGUAGE") or None,
            stt_concurrency=int(env("STT_CONCURRENCY", cls.stt_concurrency)),
            deepgram_api_key=env("DEEPGRAM_API_KEY") or None,
            deepgram_model=env("DEEPGRAM_MODEL", cls.deepgram_model),
            deepgram_language=env("DEEPGRAM_LANGUAGE", cls.deepgram_language),
            deepgram_url=env("DEEPGRAM_URL", cls.deepgram_url),
            deepgram_endpointing_ms=int(env("DEEPGRAM_ENDPOINTING_MS", cls.deepgram_endpointing_ms)),
            deepgram_connect_timeout_s=float(env("DEEPGRAM_CONNECT_TIMEOUT_S", cls.deepgram_connect_timeout_s)),
            gemini_api_key=env("GEMINI_API_KEY") or None,
            call_analysis=_bool(env("CALL_ANALYSIS", "true")),
            pii_masking=_bool(env("PII_MASKING", "true")),
            anthropic_api_key=env("ANTHROPIC_API_KEY") or None,
            anthropic_model=env("ANTHROPIC_MODEL", cls.anthropic_model),
            llm_model=env("LLM_MODEL", cls.llm_model),
            ollama_url=env("OLLAMA_URL", cls.ollama_url),
            ollama_keep_alive=env("OLLAMA_KEEP_ALIVE", cls.ollama_keep_alive),
            llm_num_ctx=int(env("LLM_NUM_CTX", cls.llm_num_ctx)),
            llm_cpu_only=_bool(env("LLM_CPU_ONLY", "true")),
            gemini_model=env("GEMINI_MODEL", cls.gemini_model),
            nba_enabled=_bool(env("NBA_ENABLED", "true")),
            nba_model=env("NBA_MODEL", cls.nba_model),
            nba_num_ctx=int(env("NBA_NUM_CTX") or env("LLM_NUM_CTX") or cls.nba_num_ctx),
            nba_context_lines=int(env("NBA_CONTEXT_LINES", cls.nba_context_lines)),
            nba_timeout_s=float(env("NBA_TIMEOUT_S", cls.nba_timeout_s)),
            nba_phrase_match=int(env("NBA_PHRASE_MATCH", cls.nba_phrase_match)),
            nba_dedupe=int(env("NBA_DEDUPE", cls.nba_dedupe)),
            nba_absent_after=int(env("NBA_ABSENT_AFTER", cls.nba_absent_after)),
            jira_url=(env("JIRA_URL") or "").rstrip("/") or None,
            jira_email=env("JIRA_EMAIL") or None,
            jira_api_token=env("JIRA_API_TOKEN") or None,
            jira_project_key=env("JIRA_PROJECT_KEY") or None,
            jira_issue_type=env("JIRA_ISSUE_TYPE", cls.jira_issue_type),
            sherpa_model_dir=env("SHERPA_MODEL_DIR", cls.sherpa_model_dir),
            sherpa_asr_model=env("SHERPA_ASR_MODEL", cls.sherpa_asr_model),
            sherpa_num_threads=int(env("SHERPA_NUM_THREADS", cls.sherpa_num_threads)),
            sherpa_vad_model=env("SHERPA_VAD_MODEL", cls.sherpa_vad_model),
            sherpa_segmentation_model=env("SHERPA_SEGMENTATION_MODEL", cls.sherpa_segmentation_model),
            sherpa_embedding_model=env("SHERPA_EMBEDDING_MODEL", cls.sherpa_embedding_model),
            sherpa_speaker_threshold=float(env("SHERPA_SPEAKER_THRESHOLD", cls.sherpa_speaker_threshold)),
            whisper_model=env("WHISPER_MODEL", cls.whisper_model),
            whisper_device=env("WHISPER_DEVICE", cls.whisper_device),
            whisper_compute_type=env("WHISPER_COMPUTE_TYPE", cls.whisper_compute_type),
            whisper_beam_size=int(env("WHISPER_BEAM_SIZE", cls.whisper_beam_size)),
            whisper_cpu_threads=int(env("WHISPER_CPU_THREADS", cls.whisper_cpu_threads)),
            whisper_num_workers=int(env("WHISPER_NUM_WORKERS", cls.whisper_num_workers)),
            max_import_bytes=int(env("MAX_IMPORT_BYTES", cls.max_import_bytes)),
        )
