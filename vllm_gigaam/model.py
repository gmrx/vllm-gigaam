"""
Адаптер модели GigaAM-v3 для vLLM.

Оборачивает нативный инференс GigaAM (Conformer + RNN-T) в vLLM-совместимый
класс модели, реализующий SupportsTranscription + IsAttentionFree.

Стратегия:
  1. get_generation_prompt() запускает полный инференс GigaAM и сохраняет
     результат в переменную уровня процесса (процесс APIServer).
  2. nn.Module (процесс EngineCore) — минимальная заглушка, которая сразу
     выдаёт EOS, и генерация vLLM не производит реальных токенов.
  3. post_process_output() возвращает сохранённый текст транскрипции
     вместо (пустого) вывода модели.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from typing import ClassVar, Literal, Mapping

import numpy as np
import torch
from torch import nn

from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.inputs.data import PromptType, TextPrompt
from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsAttentionFree,
    SupportsTranscription,
)

logger = init_logger(__name__)

# ── Нативная модель GigaAM (ленивая загрузка, один раз на процесс) ───

_gigaam_model = None
_gigaam_device = None


def _get_gigaam(model_path: str, device: torch.device):
    """Ленивая загрузка нативной модели GigaAM (один раз на процесс)."""
    global _gigaam_model, _gigaam_device
    if _gigaam_model is None:
        import sys
        sys.path.insert(0, model_path)
        from transformers import AutoModel

        logger.info("Загрузка нативной GigaAM-v3 из %s", model_path)
        _gigaam_model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True
        )
        _gigaam_model = _gigaam_model.to(device).eval()
        _gigaam_device = device
        logger.info("GigaAM-v3 загружена на %s", device)
    return _gigaam_model


# ── Транскрипция аудио с авто-чанкингом ──────────────────────────────

_CHUNK_SECONDS = 24   # Лимит GigaAM transcribe() — 25 с
_OVERLAP_SECONDS = 1


def _transcribe_audio(audio: np.ndarray, sr: int, model_path: str,
                      device: torch.device) -> str:
    """Нативный инференс GigaAM с авто-нарезкой файлов длиннее 24 с."""
    import soundfile as sf

    sr = int(sr)
    gm = _get_gigaam(model_path, device)
    max_samples = _CHUNK_SECONDS * sr

    if len(audio) <= max_samples:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sr)
            tmp = f.name
        try:
            with torch.no_grad():
                return gm.transcribe(tmp)
        finally:
            os.unlink(tmp)

    step = (_CHUNK_SECONDS - _OVERLAP_SECONDS) * sr
    parts: list[str] = []
    offset = 0
    while offset < len(audio):
        chunk = audio[offset: offset + max_samples]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, chunk, sr)
            tmp = f.name
        try:
            with torch.no_grad():
                text = gm.transcribe(tmp)
            if text:
                parts.append(text)
        finally:
            os.unlink(tmp)
        offset += step

    return " ".join(parts)


# ── Класс модели для vLLM ───────────────────────────────────────────

# Общая переменная между get_generation_prompt() и post_process_output(),
# оба метода выполняются в процессе APIServer.
_last_transcription: str = ""

EOS_TOKEN_ID = 1  # </s> в T5/SentencePiece токенизаторе


class GigaAMForTranscription(
    nn.Module,
    SupportsTranscription,
    HasInnerState,
    IsAttentionFree,
):
    """
    vLLM-обёртка для ai-sage/GigaAM-v3 (Conformer + RNN-T).

    Реальная транскрипция происходит в ``get_generation_prompt()`` через
    нативную модель GigaAM. Сам nn.Module — минимальная заглушка,
    которая сразу выдаёт EOS, чтобы цикл генерации vLLM остановился.
    """

    supports_transcription_only: ClassVar[bool] = True
    has_inner_state: ClassVar[Literal[True]] = True
    is_attention_free: ClassVar[Literal[True]] = True

    supported_languages: ClassVar[Mapping[str, str]] = {
        "ru": "Русский",
        "en": "Английский",
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config

        hidden = getattr(config, "hidden_size", 768)
        self.vocab_size = getattr(config, "vocab_size", 1025)

        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, hidden)
        self.lm_head = ParallelLMHead(self.vocab_size, hidden)

    # ── Интерфейс nn.Module ──

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden = inputs_embeds
        elif input_ids is not None:
            hidden = self.embed_tokens(input_ids)
        else:
            raise ValueError("Необходим input_ids или inputs_embeds")
        return hidden

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Всегда возвращает логиты EOS — генерация останавливается мгновенно."""
        batch = hidden_states.shape[0]
        logits = torch.full(
            (batch, self.vocab_size),
            -1e9,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        logits[:, EOS_TOKEN_ID] = 0.0
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = set()
        for name, tensor in weights:
            loaded.add(name)

        nn.init.normal_(self.embed_tokens.weight, std=0.02)
        nn.init.normal_(self.lm_head.weight, std=0.02)
        loaded.update(["embed_tokens.weight", "lm_head.weight"])

        logger.info("GigaAM плагин: пропущено %d HF-весов (используется нативная модель)",
                     len(loaded) - 2)
        return loaded

    # ── Интерфейс SupportsTranscription ──

    @classmethod
    def get_speech_to_text_config(
        cls,
        model_config: ModelConfig,
        task_type: Literal["transcribe", "translate"],
    ) -> SpeechToTextConfig:
        return SpeechToTextConfig(
            sample_rate=16_000,
            max_audio_clip_s=600,
            min_energy_split_window_size=None,
        )

    @classmethod
    def get_generation_prompt(
        cls,
        audio: np.ndarray,
        stt_config: SpeechToTextConfig,
        model_config: ModelConfig,
        language: str | None,
        task_type: Literal["transcribe", "translate"],
        request_prompt: str,
        to_language: str | None,
    ) -> PromptType:
        model_path = model_config.model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        global _last_transcription
        text = _transcribe_audio(audio, stt_config.sample_rate, model_path, device)
        logger.info("GigaAM транскрипция: %s", text[:100])

        _last_transcription = text

        return TextPrompt(prompt="<s>")

    @classmethod
    def post_process_output(cls, text: str) -> str:
        """Возвращает предвычисленную транскрипцию вместо вывода модели."""
        global _last_transcription
        result = _last_transcription
        _last_transcription = ""
        return result

    @classmethod
    def get_num_audio_tokens(
        cls,
        audio_duration_s: float,
        stt_config: SpeechToTextConfig,
        model_config: ModelConfig,
    ) -> int | None:
        return max(1, int(audio_duration_s * 10))

    @classmethod
    def validate_language(cls, language: str | None) -> str | None:
        if language is None:
            return "ru"
        return language
