# KServe

!!! note "Integration reference"
    This inherited tutorial describes the external integration and upstream
    artifacts. For Nabla, use a build from this checkout, a supported target
    checkpoint and NVIDIA CUDA. Validate the adapter and P/D topology separately;
    this page is not a Nabla deployment or performance certification.

vLLM can be deployed with [KServe](https://github.com/kserve/kserve) on Kubernetes for highly scalable distributed model serving.

You can use vLLM with KServe's [Hugging Face serving runtime](https://kserve.github.io/website/docs/model-serving/generative-inference/overview) or via [`LLMInferenceService` that uses llm-d](https://kserve.github.io/website/docs/model-serving/generative-inference/llmisvc/llmisvc-overview).
