# BentoML

!!! note "Integration reference"
    This inherited tutorial describes the external integration and upstream
    artifacts. For Nabla, use a build from this checkout, a supported target
    checkpoint and NVIDIA CUDA. Validate the adapter and P/D topology separately;
    this page is not a Nabla deployment or performance certification.

[BentoML](https://github.com/bentoml/BentoML) allows you to deploy a large language model (LLM) server with vLLM as the backend, which exposes OpenAI-compatible endpoints. You can serve the model locally or containerize it as an OCI-compliant image and deploy it on Kubernetes.

For details, see the tutorial [vLLM inference in the BentoML documentation](https://docs.bentoml.com/en/latest/use-cases/large-language-models/vllm.html).
