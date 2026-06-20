# MaxText Backend Benchmarking & MoE Scaling Study

## Overview

This repository contains my solutions and experimental results for the MaxText benchmarking assignment. The objective was to understand MaxText configuration, model scaling, Mixture-of-Experts (MoE) architecture, and benchmark model training across CPU, GPU, and TPU backends using synthetic data.

The work includes:

* Running Qwen3 Dense models on CPU, GPU, and TPU
* Scaling a DeepSeek Mixture-of-Experts (MoE) model to under 1B parameters
* Benchmarking DeepSeek MoE on CPU, GPU, and TPU
* Comparing Dense and MoE architectures
* Investigating MegaBlocks sparse-kernel support

Synthetic data was used for all experiments.

---

# Environment

## Framework

* MaxText 0.2.1
* JAX 0.8.1
* jaxlib 0.8.1

## Hardware

### GPU

* NVIDIA A100 SXM4 80GB

### TPU

* Google TPU v5e, v56

### CPU

* Google Colab CPU Runtime

---

# Repository Structure

```text
.
├── notebooks/
│   ├── task2_qwen_dense.ipynb
│   ├── task3_deepseek_moe.ipynb
│
├── logs/
│   ├── qwen_0.6b_gpu/
│   ├── qwen_0.6b_tpu/
│   ├── qwen_0.6b_cpu/
│   ├── qwen_1.7b_gpu/
│   ├── qwen_1.7b_tpu/
│   ├── deepseek_moe_gpu/
│   ├── deepseek_moe_tpu/
│   └── deepseek_moe_cpu/
│
├── reports/
│   └── benchmark_summary.md
│
└── README.md
```

---

# Task 2: Dense Model Benchmarking

## Objective

Benchmark Qwen3 dense transformer models across multiple hardware backends.

Models tested:

* Qwen3 0.6B
* Qwen3 1.7B

---

## Results

| Model      | Device   | Step Time  | TFLOP/s/device | Tokens/s/device |
| ---------- | -------- | ---------- | -------------- | --------------- |
| Qwen3 0.6B | A100 GPU | 0.123 sec  | 71.002         | 16,587.16       |
| Qwen3 0.6B | TPU v5e  | 0.079 sec  | 67.725         | 6,450.719       |
| Qwen3 0.6B | CPU      | 25.469 sec | 0.075          | 20.103          |
| Qwen3 1.7B | A100 GPU | 1.882 sec  | 144.007        | 13,059.12       |
| Qwen3 1.7B | TPU v5e  | 0.079 sec  | 67.725         | 6,450.719       |

---

## Observations

### GPU

* Highest compute throughput.
* Excellent utilization at larger model sizes.
* Best overall accelerator performance.

### TPU

* Competitive performance.
* Lower step time than GPU in some configurations.
* Well suited for large-scale JAX workloads.

### CPU

* Significantly slower than accelerator hardware.
* Useful only for debugging and validation.

---

# Task 3: DeepSeek MoE Scaling Study

## Objective

Scale a DeepSeek Mixture-of-Experts model to under 1B parameters and benchmark it on:

* CPU
* GPU
* TPU

using synthetic training data.

---

# Model Modifications

The original DeepSeek architecture was reduced to remain under the 1B parameter requirement.

## Changes Made

| Parameter        | Value   |
| ---------------- | ------- |
| Hidden Size      | Reduced |
| Number of Layers | Reduced |
| Expert Count     | 16      |
| Shared Experts   | 1       |
| Top-K Routing    | 2       |
| Sequence Length  | 512     |
| Precision        | BF16    |

---

## Final Model

| Metric                   | Value       |
| ------------------------ | ----------- |
| Total Parameters         | 308 Million |
| Active Experts per Token | 2           |
| Total Experts            | 16          |
| Shared Experts           | 1           |
| Training Steps           | 50          |

---

# DeepSeek MoE Results

## GPU (A100)

| Metric          | Value      |
| --------------- | ---------- |
| Step Time       | 0.040 sec  |
| TFLOP/s/device  | 14.464     |
| Tokens/s/device | 12,944.329 |
| Final Loss      | 2.083      |

---

## TPU v6e

| Metric          | Value      |
| --------------- | ---------- |
| Step Time       | 0.021 sec  |
| TFLOP/s/device  | 27.647     |
| Tokens/s/device | 24,741.471 |
| Final Loss      | 2.095      |

---

## CPU

| Metric          | Value     |
| --------------- | --------- |
| Step Time       | 8.018 sec |
| TFLOP/s/device  | 0.071     |
| Tokens/s/device | 63.853    |
| Final Loss      | 2.085     |

---

# MoE vs Dense Comparison

| Feature                   | Dense (Qwen)         | MoE (DeepSeek)     |
| ------------------------- | -------------------- | ------------------ |
| Architecture              | Standard Transformer | Mixture of Experts |
| Parameters Used per Token | All Parameters       | Top-K Experts Only |
| Routing                   | None                 | Learned Router     |
| Compute Cost              | Higher               | Lower              |
| Parameter Efficiency      | Lower                | Higher             |
| Scalability               | Limited              | Better             |

---

# Why MoE Behaves Differently

Dense transformers execute every feed-forward network for every token.

MoE models introduce a routing network that activates only a subset of experts for each token.

Benefits:

* Higher model capacity
* Lower compute cost per token
* Better parameter efficiency

Trade-offs:

* Expert routing overhead
* Load balancing complexity
* Additional communication costs in distributed settings

---

# MegaBlocks Investigation

An attempt was made to run the DeepSeek MoE model using MegaBlocks sparse kernels.

Training failed with:

```text
NotImplementedError:
dynamic grid bounds not supported in the Triton backend
```

Root cause:

* MegaBlocks uses Triton/Pallas kernels.
* Current JAX/Triton backend does not support the required dynamic grid bounds.

Workaround:

* Disable MegaBlocks sparse kernels.
* Run MoE using the standard implementation.

After disabling MegaBlocks, training completed successfully on:

* GPU
* TPU
* CPU

---

# Key Findings

1. TPU achieved the highest DeepSeek MoE throughput:

   * ~24.7K tokens/sec

2. GPU delivered strong accelerator performance:

   * ~12.9K tokens/sec

3. CPU remained useful only for functional validation.

4. MoE enabled scaling model capacity while keeping active compute low through Top-K routing.

5. MegaBlocks support is currently blocked by Triton backend limitations in this environment.

---

# Conclusion

This study demonstrates successful deployment and benchmarking of both dense and Mixture-of-Experts transformer architectures using MaxText.

The DeepSeek MoE model was scaled to 308M parameters and trained successfully across CPU, GPU, and TPU backends. Results show that accelerator hardware significantly outperforms CPU execution, while MoE architectures provide a more compute-efficient alternative to dense transformers through sparse expert activation.
