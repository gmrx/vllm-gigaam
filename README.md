# vllm-gigaam

Плагин vLLM для запуска модели распознавания речи **[ai-sage/GigaAM-v3](https://huggingface.co/ai-sage/GigaAM-v3)** через OpenAI-совместимый API `/v1/audio/transcriptions`.

GigaAM-v3 — модель автоматического распознавания речи (ASR) на русском языке от SberDevices, построенная на архитектуре Conformer + RNN-T. Она **не** является трансформером, поэтому не может работать напрямую в движке генерации vLLM. Данный плагин решает эту проблему:

1. Регистрирует кастомную архитектуру `GigaAMForTranscription` в `ModelRegistry` vLLM
2. Запускает реальный инференс GigaAM внутри `get_generation_prompt()` — до цикла генерации vLLM
3. Возвращает в vLLM минимальную заглушку, которая сразу выдаёт EOS
4. Передаёт предвычисленный текст транскрипции через `post_process_output()`

## Требования

- Python >= 3.10
- CUDA GPU
- vLLM >= 0.17
- ~1 ГБ видеопамяти для модели

## Быстрый старт

### 1. Установка плагина

```bash
git clone https://github.com/gmrx/vllm-gigaam.git
cd vllm-gigaam
pip install -e .
```

### 2. Скачивание и подготовка модели

Из корня репозитория `vllm-gigaam/` выполните:

```bash
python scripts/prepare_model.py
```

По умолчанию модель сохраняется в `./gigaam-v3-e2e-rnnt`. Можно указать другой путь:

```bash
python scripts/prepare_model.py --output-dir /путь/к/модели
```

Скрипт выполнит 4 шага автоматически:

1. **Скачивание** — загрузит `ai-sage/GigaAM-v3` (ревизия `e2e_rnnt`, ~430 МБ) из HuggingFace
2. **Конвертация весов** — преобразует `pytorch_model.bin` → `model.safetensors`
3. **Создание токенизатора** — сгенерирует HuggingFace-совместимый токенизатор из SentencePiece-модели
4. **Патч конфига** — добавит в `config.json` обязательные для vLLM поля (`architectures`, `hidden_size` и др.)

По завершении скрипт выведет готовую команду для запуска сервера.

### 3. Запуск сервера

```bash
vllm serve ./gigaam-v3-e2e-rnnt \
  --trust-remote-code \
  --max-model-len 4096 \
  --port 9000 \
  --enforce-eager
```

Для привязки к конкретному GPU:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve ./gigaam-v3-e2e-rnnt \
  --trust-remote-code \
  --max-model-len 4096 \
  --port 9000 \
  --enforce-eager
```

### 4. Транскрипция аудио

```bash
curl http://localhost:9000/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "model=./gigaam-v3-e2e-rnnt" \
  -F "language=ru"
```

Ответ:

```json
{
  "text": "Игорь, добрый день! Андрей, добрый день! ...",
  "usage": {"type": "duration", "seconds": 136}
}
```

### Python-клиент (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:9000/v1", api_key="unused")

with open("audio.mp3", "rb") as f:
    result = client.audio.transcriptions.create(
        model="./gigaam-v3-e2e-rnnt",
        file=f,
        language="ru",
    )

print(result.text)
```

## Как это работает

### Схема архитектуры

```
┌──────────────────────────────────────────────────────┐
│  vLLM API-сервер  (порт 9000)                        │
│                                                      │
│  POST /v1/audio/transcriptions                       │
│       │                                              │
│       ▼                                              │
│  ┌─────────────────────────────────────────────────┐ │
│  │  get_generation_prompt()                        │ │
│  │    1. Получает аудио-волну от vLLM              │ │
│  │    2. Сохраняет во временный .wav файл          │ │
│  │    3. Запускает нативный GigaAM model.transcribe│ │
│  │    4. Сохраняет результат в _last_transcription  │ │
│  │    5. Возвращает TextPrompt("<s>") в vLLM       │ │
│  └─────────────────────────────────────────────────┘ │
│       │                                              │
│       ▼                                              │
│  ┌─────────────────────────────────────────────────┐ │
│  │  Движок vLLM (процесс EngineCore)              │ │
│  │    - Токенизирует "<s>" → 1 токен              │ │
│  │    - forward() → embed_tokens                   │ │
│  │    - compute_logits() → всегда EOS             │ │
│  │    - Генерация останавливается мгновенно       │ │
│  └─────────────────────────────────────────────────┘ │
│       │                                              │
│       ▼                                              │
│  ┌─────────────────────────────────────────────────┐ │
│  │  post_process_output()                          │ │
│  │    Возвращает сохранённый текст транскрипции    │ │
│  └─────────────────────────────────────────────────┘ │
│       │                                              │
│       ▼                                              │
│  {"text": "результат транскрипции..."}               │
└──────────────────────────────────────────────────────┘
```

### Регистрация плагина

Плагин использует механизм `entry_points` vLLM:

```python
# setup.py
entry_points={
    "vllm.general_plugins": [
        "register_gigaam = vllm_gigaam:register",
    ],
}
```

При импорте регистрируется:
- Архитектура модели `GigaAMForTranscription`
- `MambaModelArchConfigConvertor` для типа модели `gigaam` (сигнализирует об attention-free архитектуре)

### Авто-чанкинг

Метод `transcribe()` GigaAM имеет ограничение в 25 секунд. Плагин автоматически разбивает длинные аудиофайлы на чанки по 24 секунды с перекрытием в 1 секунду и объединяет результаты. Это позволяет обойтись без `transcribe_longform()`, который требует HuggingFace-токен для VAD-пайплайна pyannote.

### Патчи config.json

Скрипт `prepare_model.py` добавляет в `config.json` следующие поля верхнего уровня (необходимые для vLLM):

| Поле | Значение | Назначение |
|---|---|---|
| `architectures` | `["GigaAMForTranscription"]` | Маппинг на зарегистрированный класс модели |
| `model_type` | `"gigaam"` | Используется для поиска конвертора конфига |
| `hidden_size` | `768` | d_model энкодера Conformer |
| `num_attention_heads` | `16` | n_heads энкодера Conformer |
| `num_hidden_layers` | `16` | n_layers энкодера Conformer |
| `intermediate_size` | `3072` | hidden_size * ff_expansion_factor |
| `vocab_size` | `1127` | SentencePiece (1024) + extra_ids (100) + спецтокены |
| `max_position_embeddings` | `5000` | Из конфига энкодера |

## Структура проекта

```
vllm-gigaam/
├── README.md
├── LICENSE
├── setup.py
├── .gitignore
├── scripts/
│   └── prepare_model.py    # Скачивание + конвертация + патч
└── vllm_gigaam/
    ├── __init__.py          # Регистрация плагина
    └── model.py             # Модель GigaAMForTranscription
```

## Поддерживаемые форматы аудио

Все форматы, поддерживаемые `librosa` / `ffmpeg`: WAV, MP3, M4A, OGG, FLAC, WEBM и др.

## Ограничения

- Запросы на транскрипцию обрабатываются последовательно (по одному) из-за использования переменной уровня процесса для передачи результатов между `get_generation_prompt` и `post_process_output`.
- Формат ответа `verbose_json` с таймстемпами не поддерживается.
- Перевод (`/v1/audio/translations`) не поддерживается — GigaAM является моделью ASR для русского языка.

## Протестировано с

- vLLM 0.17.1
- Python 3.12
- CUDA 12.x
- NVIDIA GPU с >= 1 ГБ видеопамяти

## Благодарности

- Модель **GigaAM-v3** — [SberDevices / ai-sage](https://huggingface.co/ai-sage/GigaAM-v3)
- **vLLM** — [vllm-project](https://github.com/vllm-project/vllm)

## Лицензия

MIT
