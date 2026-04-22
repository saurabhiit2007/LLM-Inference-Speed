# References

## Attention Optimization

- Dao, T. et al. (2022). **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.** NeurIPS 2022. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- Dao, T. (2023). **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.** ICLR 2024. [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
- Shah, J. et al. (2024). **FlashAttention-3: Fast and Accurate Attention for Hopper GPUs.** [arXiv:2407.08608](https://arxiv.org/abs/2407.08608)
- Kwon, W. et al. (2023). **Efficient Memory Management for Large Language Model Serving with PagedAttention.** SOSP 2023. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
- Shazeer, N. (2019). **Fast Transformer Decoding: One Write-Head is All You Need (Multi-Query Attention).** [arXiv:1911.02150](https://arxiv.org/abs/1911.02150)
- Ainslie, J. et al. (2023). **GQA: Training Generalised Multi-Query Transformer Models from Multi-Head Checkpoints.** EMNLP 2023. [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)

## Decoding Strategies

- Leviathan, Y. et al. (2023). **Fast Inference from Transformers via Speculative Decoding.** ICML 2023. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
- Chen, C. et al. (2023). **Accelerating Large Language Model Decoding with Speculative Sampling.** [arXiv:2302.01318](https://arxiv.org/abs/2302.01318)
- Holtzman, A. et al. (2020). **The Curious Case of Neural Text Degeneration (nucleus sampling).** ICLR 2020. [arXiv:1904.09751](https://arxiv.org/abs/1904.09751)

## Batching

- Yu, G. et al. (2022). **Orca: A Distributed Serving System for Transformer-Based Generative Models (continuous batching).** OSDI 2022.
- Agrawal, A. et al. (2024). **Sarathi-Serve: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills.** OSDI 2024. [arXiv:2403.02310](https://arxiv.org/abs/2403.02310)

## Quantization

- Dettmers, T. et al. (2022). **LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.** NeurIPS 2022. [arXiv:2208.07339](https://arxiv.org/abs/2208.07339)
- Frantar, E. et al. (2022). **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.** ICLR 2023. [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)
- Lin, J. et al. (2023). **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.** MLSys 2024. [arXiv:2306.00978](https://arxiv.org/abs/2306.00978)
- Xiao, G. et al. (2023). **SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models.** ICML 2023. [arXiv:2211.10438](https://arxiv.org/abs/2211.10438)
- Dettmers, T. et al. (2023). **QLoRA: Efficient Finetuning of Quantized LLMs.** NeurIPS 2023. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- Gerganov, G. (2023). **llama.cpp / GGML / GGUF.** [GitHub](https://github.com/ggml-org/llama.cpp)

## Serving Frameworks

- **vLLM.** Kwon et al. (2023). [GitHub](https://github.com/vllm-project/vllm)
- **TensorRT-LLM.** NVIDIA (2023). [GitHub](https://github.com/NVIDIA/TensorRT-LLM)
- **Text Generation Inference (TGI).** Hugging Face (2023). [GitHub](https://github.com/huggingface/text-generation-inference)
- **DeepSpeed-Inference.** Aminabadi et al. (2022). **DeepSpeed Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale.** SC 2022. [arXiv:2207.00032](https://arxiv.org/abs/2207.00032)
- **Triton Inference Server.** NVIDIA (2022). [GitHub](https://github.com/triton-inference-server/server)

## Test-Time Compute Scaling

- Lightman, H. et al. (2023). **Let's Verify Step by Step (Process Reward Models).** ICLR 2024. [arXiv:2305.20050](https://arxiv.org/abs/2305.20050)
- Wang, P. et al. (2024). **Math-Shepherd: Verify and Reinforce LLMs Step-by-Step without Human Annotations.** ACL 2024. [arXiv:2312.08935](https://arxiv.org/abs/2312.08935)
- Snell, C. et al. (2024). **Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Model Parameters.** [arXiv:2408.03314](https://arxiv.org/abs/2408.03314)
- DeepSeek-AI (2025). **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.** [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
- OpenAI (2024). **OpenAI o1 System Card.** [openai.com](https://openai.com/index/openai-o1-system-card/)

## Disaggregated Prefill-Decode

- Qin, R. et al. (2024). **Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving.** [arXiv:2407.00079](https://arxiv.org/abs/2407.00079)
- Zhong, Y. et al. (2024). **DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving.** OSDI 2024. [arXiv:2401.09670](https://arxiv.org/abs/2401.09670)

## Benchmarks & Profiling

- Williams, S. et al. (2009). **Roofline: An Insightful Visual Performance Model for Multicore Architectures.** CACM 2009.
- NVIDIA (2024). **Nsight Systems / Nsight Compute.** [developer.nvidia.com](https://developer.nvidia.com/nsight-systems)
- **LM Evaluation Harness.** EleutherAI. [GitHub](https://github.com/EleutherAI/lm-evaluation-harness)
