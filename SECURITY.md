# Security Policy

## Reporting security issues

Report issues in nabla-vllm privately through this repository's
[vulnerability submission form](https://github.com/zhangxin81/nabla-vllm/security/advisories/new).
Include the affected revision, target model, hardware, configuration and a
minimal reproducer. Keep unpatched vulnerability reports in the private form.

The repository maintainers triage reports and coordinate fixes and advisory
publication. If the affected code is also present in upstream vLLM, the
maintainers coordinate with the
[upstream vulnerability management team](https://docs.vllm.ai/en/latest/contributing/vulnerability_management.html).
Issues reproduced only in upstream vLLM can be reported through its
[private reporting form](https://github.com/vllm-project/vllm/security/advisories/new).
The retained upstream process documents that project's team and
prenotification group.

## Deployment assumptions

The runtime retains vLLM's trust boundaries for model loading, API clients,
distributed communication and shared caches. CUDA specialization retains
these deployment requirements. See the retained
[upstream security reference](docs/usage/security.md) and
[PyTorch security policy](https://github.com/pytorch/pytorch/blob/main/SECURITY.md)
for those deployment assumptions.
