# KubeAI

!!! note "Integration reference"
    This inherited tutorial describes the external integration and upstream
    artifacts. For Nabla, use a build from this checkout, a supported target
    checkpoint and NVIDIA CUDA. Validate the adapter and P/D topology separately;
    this page is not a Nabla deployment or performance certification.

[KubeAI](https://github.com/substratusai/kubeai) is a Kubernetes operator that enables you to deploy and manage AI models on Kubernetes. It provides a simple and scalable way to deploy vLLM in production. Functionality such as scale-from-zero, load based autoscaling, model caching, and much more is provided out of the box with zero external dependencies.

Please see the Installation Guides for environment specific instructions:

- [Any Kubernetes Cluster](https://www.kubeai.org/installation/any/)
- [AKS](https://www.kubeai.org/installation/aks/)
- [EKS](https://www.kubeai.org/installation/eks/)
- [GKE](https://www.kubeai.org/installation/gke/)

Once you have KubeAI installed, you can
[configure text generation models](https://www.kubeai.org/how-to/configure-text-generation-models/)
using vLLM.
