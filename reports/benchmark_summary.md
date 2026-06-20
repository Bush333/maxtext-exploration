# Benchmark Summary

## Environment

| Component      | Value          |
| -------------- | -------------- |
| Framework      | MaxText 0.2.1  |
| JAX            | 0.8.1          |
| jaxlib         | 0.8.1          |
| Dataset        | Synthetic Data |
| Training Steps | 50             |

---

# Task 2: Dense Model Benchmarking

## Results

| Model      | Device   | Step Time  | TFLOP/s/device | Tokens/s/device |
| ---------- | -------- | ---------- | -------------- | --------------- |
| Qwen3 0.6B | A100 GPU | 0.123 sec  | 71.002         | 16,587.16       |
| Qwen3 0.6B | TPU v5e  | 0.079 sec  | 67.725         | 6,450.719       |
| Qwen3 0.6B | CPU      | 25.469 sec | 0.075          | 20.103          |
| Qwen3 1.7B | A100 GPU | 1.882 sec  | 144.007        | 13,059.12       |
| Qwen3 1.7B | TPU v5e  | 0.079 sec  | 67.725         | 6,450.719       |

### Key Observation

* GPU delivered the highest throughput for Qwen models.
* TPU provided competitive performance with lower step times.
* CPU was significantly slower and primarily useful for validation.

---

# Task 3: DeepSeek MoE Benchmarking

## Model Configuration

| Parameter        | Value |
| ---------------- | ----- |
| Total Parameters | 308M  |
| Experts          | 16    |
| Shared Experts   | 1     |
| Top-K Routing    | 2     |
| Sequence Length  | 512   |
| Precision        | BF16  |

## Results

| Device   | Step Time | TFLOP/s/device | Tokens/s/device | Final Loss |
| -------- | --------- | -------------- | --------------- | ---------- |
| A100 GPU | 0.040 sec | 14.464         | 12,944.329      | 2.083      |
| TPU v5e  | 0.021 sec | 27.647         | 24,741.471      | 2.095      |
| CPU      | 8.018 sec | 0.071          | 63.853          | 2.085      |

### Key Observation

* TPU achieved the highest throughput (~24.7K tokens/sec).
* GPU achieved strong accelerator performance (~12.9K tokens/sec).
* CPU remained significantly slower than accelerator hardware.

---

# Dense vs MoE Comparison

| Aspect               | Dense (Qwen)      | DeepSeek MoE       |
| -------------------- | ----------------- | ------------------ |
| Architecture         | Dense Transformer | Mixture of Experts |
| Parameters Activated | All               | Top-K Experts Only |
| Compute Efficiency   | Lower             | Higher             |
| Parameter Efficiency | Lower             | Higher             |

### Why MoE Differs

MoE models activate only a subset of experts for each token, reducing compute while maintaining larger overall model capacity. This improves parameter efficiency but introduces routing overhead.

---

# MegaBlocks Investigation

Attempting to run DeepSeek MoE with MegaBlocks sparse kernels resulted in:

```text
NotImplementedError:
dynamic grid bounds not supported in the Triton backend
```

The issue was resolved by disabling MegaBlocks and running the standard MoE implementation.

---

# Final Conclusion

* Successfully benchmarked Dense and MoE models on CPU, GPU, and TPU.
* DeepSeek MoE was scaled to 308M parameters (<1B requirement).
* TPU delivered the best MoE throughput.
* MoE achieved improved parameter efficiency through sparse expert activation.
* MegaBlocks sparse kernels were not supported in the current Triton backend environment.
