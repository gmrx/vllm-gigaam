#!/usr/bin/env python3
"""
Скачивание ai-sage/GigaAM-v3, конвертация весов в safetensors,
патч config.json для vLLM и создание HuggingFace-совместимого токенизатора.

Использование:
    python scripts/prepare_model.py [--output-dir ./gigaam-v3-e2e-rnnt]
"""

import argparse
import json
import os
import sys


def download_model(output_dir: str):
    """Скачивает GigaAM-v3 (ревизия e2e_rnnt) из HuggingFace."""
    from huggingface_hub import snapshot_download

    print("==> Скачивание ai-sage/GigaAM-v3 (ревизия: e2e_rnnt) ...")
    snapshot_download(
        repo_id="ai-sage/GigaAM-v3",
        revision="e2e_rnnt",
        local_dir=output_dir,
    )
    print(f"    Скачано в {output_dir}")


def convert_weights(output_dir: str):
    """Конвертирует pytorch_model.bin → model.safetensors."""
    import torch
    from safetensors.torch import save_file

    bin_path = os.path.join(output_dir, "pytorch_model.bin")
    sf_path = os.path.join(output_dir, "model.safetensors")

    if not os.path.exists(bin_path):
        print(f"    ПРОПУСК: {bin_path} не найден")
        return
    if os.path.exists(sf_path):
        print(f"    ПРОПУСК: {sf_path} уже существует")
        return

    print("==> Конвертация pytorch_model.bin → model.safetensors ...")
    state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
    save_file(state_dict, sf_path)
    size_mb = os.path.getsize(sf_path) / 1024 / 1024
    print(f"    Ключей: {len(state_dict)}, Размер: {size_mb:.1f} МБ")


def create_tokenizer(output_dir: str) -> int:
    """Создаёт HuggingFace-совместимые файлы токенизатора из SentencePiece-модели."""
    from transformers import T5Tokenizer

    sp_path = os.path.join(output_dir, "tokenizer.model")
    if not os.path.exists(sp_path):
        print(f"    ОШИБКА: {sp_path} не найден")
        sys.exit(1)

    print("==> Создание HuggingFace-токенизатора ...")
    tok = T5Tokenizer(vocab_file=sp_path, extra_ids=100)
    tok.save_pretrained(output_dir)
    vocab_size = tok.vocab_size + len(tok.get_added_vocab())
    print(f"    Токенизатор сохранён. Размер словаря: {vocab_size}")
    return vocab_size


CONFIG_PATCH = {
    "model_type": "gigaam",
    "architectures": ["GigaAMForTranscription"],
    "hidden_size": 768,
    "num_attention_heads": 16,
    "num_hidden_layers": 16,
    "intermediate_size": 3072,
    "max_position_embeddings": 5000,
    "auto_map": {
        "AutoConfig": "modeling_gigaam.GigaAMConfig",
        "AutoModel": "modeling_gigaam.GigaAMModel",
    },
}


def patch_config(output_dir: str, vocab_size: int):
    """Патчит config.json, добавляя обязательные для vLLM поля верхнего уровня."""
    config_path = os.path.join(output_dir, "config.json")

    print("==> Патч config.json ...")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    config.update(CONFIG_PATCH)
    config["vocab_size"] = vocab_size

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"    config.json пропатчен (vocab_size={vocab_size})")


def main():
    parser = argparse.ArgumentParser(
        description="Подготовка GigaAM-v3 для работы в vLLM"
    )
    parser.add_argument(
        "--output-dir",
        default="./gigaam-v3-e2e-rnnt",
        help="Директория для сохранения модели (по умолчанию: ./gigaam-v3-e2e-rnnt)",
    )
    args = parser.parse_args()

    download_model(args.output_dir)
    convert_weights(args.output_dir)
    vocab_size = create_tokenizer(args.output_dir)
    patch_config(args.output_dir, vocab_size)

    print()
    print("=" * 60)
    print("Готово! Запустите сервер командой:")
    print()
    print(f"  vllm serve {args.output_dir} \\")
    print("    --trust-remote-code \\")
    print("    --max-model-len 4096 \\")
    print("    --port 9000 \\")
    print("    --enforce-eager")
    print("=" * 60)


if __name__ == "__main__":
    main()
