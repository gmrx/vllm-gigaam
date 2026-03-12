def register():
    from vllm import ModelRegistry
    from vllm.transformers_utils.model_arch_config_convertor import (
        MODEL_ARCH_CONFIG_CONVERTORS,
        MambaModelArchConfigConvertor,
    )

    if "GigaAMForTranscription" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "GigaAMForTranscription",
            "vllm_gigaam.model:GigaAMForTranscription",
        )

    MODEL_ARCH_CONFIG_CONVERTORS["gigaam"] = MambaModelArchConfigConvertor
