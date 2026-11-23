# NVIDIA Dynamo

!!! note "Integration reference"
    This inherited tutorial describes the external integration and upstream
    artifacts. For Nabla, use a build from this checkout, a supported target
    checkpoint and NVIDIA CUDA. Validate the adapter and P/D topology separately;
    this page is not a Nabla deployment or performance certification.

[NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) is an open-source framework for distributed LLM inference that can run vLLM on Kubernetes with flexible serving architectures (e.g. aggregated/disaggregated, optional router/planner).

For Kubernetes deployment instructions and examples (including vLLM), see the [Deploying Dynamo on Kubernetes](https://docs.nvidia.com/dynamo/kubernetes/getting-started/introduction) guide.

Background reading: InfoQ news coverage — [NVIDIA Dynamo simplifies Kubernetes deployment for LLM inference](https://www.infoq.com/news/2025/12/nvidia-dynamo-kubernetes/).
