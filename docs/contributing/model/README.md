# Target-model implementation

Nabla retains native Qwen3.8-Flash-Next and DeepSeek-V4.1-Flash implementations
and their compatible draft/MTP dependencies. The
[model registry scope](../../models/supported_models.md) determines what this
runtime loads. Generic Transformers, pooling and unrelated native model
registrations are outside that scope.

Start with the existing target implementation and trace the actual operator,
cache, checkpoint and P/D paths. Changes must preserve request outputs and
report model evaluation alongside performance measurements. The inherited
implementation guides explain the `vllm` interfaces used by the retained code:

- [Basic Model](basic.md)
- [Registering a Model](registration.md)
- [Unit Testing](tests.md)
- [Multi-Modal Support](multimodal.md): reference for conditional-generation paths.
- [Speech-to-Text Support](transcription.md): upstream reference outside native scope.

Discuss target-model changes in
[Nabla issues](https://github.com/zhangxin81/nabla-vllm/issues).
