# YOLOv8 GPU Inference Optimization

Benchmarking and optimizing YOLOv8 object detection inference on an NVIDIA T4 GPU using **FP16 precision**, **batching**, and **TensorRT**, and profiling the pipeline to find the remaining bottleneck.

**Result:** 2.77× higher throughput (90.6 → 251.4 images/sec) and 29% lower latency (11.04 → 7.83 ms/image), with **no loss in detection accuracy**.

---

## Goal

A trained model is only useful in production if it runs fast enough. This project starts from a standard PyTorch YOLOv8 baseline and applies GPU optimizations one at a time, measuring:

- **Latency:** milliseconds per image (lower is better)
- **Throughput:** images processed per second (higher is better)
- **GPU memory:** peak memory used
- **Accuracy:** whether detections change after optimization

## Setup

| Item | Details |
| --- | --- |
| Model | YOLOv8n (nano), pretrained on COCO |
| GPU | NVIDIA Tesla T4 (15 GB), Google Colab |
| Frameworks | PyTorch, Ultralytics, ONNX, TensorRT |
| Test data | 64 copies of a 1080×810 image (identical workload for fair comparison) |

## Method

Each configuration is measured with a custom benchmark function that:

1. Runs **2 untimed warm-up rounds**, so one-time GPU setup costs don't distort the results.
2. Calls `torch.cuda.synchronize()` before and after timing, because GPU work runs asynchronously and timing without syncing is inaccurate.
3. Processes all 64 images **3 times** and averages, for stable numbers.

Timings are **end-to-end**: preprocessing, inference, and postprocessing.

Optimizations tested:

1. **Baseline:** PyTorch, FP32, batch size 1
2. **FP16 (half precision):** 16-bit floats, which use the T4's Tensor Cores and halve memory traffic
3. **Batching:** batch sizes 1, 8, 16, 32, 64, so the GPU processes many images in parallel
4. **TensorRT:** ONNX export, then a TensorRT FP16 engine with dynamic batch size (1–16)

---

## Results

### Summary

| Setup | ms/image | images/sec | vs baseline |
| --- | --- | --- | --- |
| PyTorch FP32, batch 1 (baseline) | 11.04 | 90.6 | 1.00× |
| PyTorch FP16, batch 1 | 9.56 | 104.6 | 1.15× |
| **TensorRT FP16, batch 1** (lowest latency) | **7.83** | **127.7** | **1.41×** |
| PyTorch FP32, batch 16 | 5.09 | 196.5 | 2.17× |
| TensorRT FP16, batch 8 | 4.35 | 229.6 | 2.54× |
| **PyTorch FP16, batch 16** (highest throughput) | **3.98** | **251.4** | **2.77×** |

### Batch size scaling (PyTorch)

| Precision | Batch | ms/image | images/sec | ms/batch | Peak GPU memory |
| --- | --- | --- | --- | --- | --- |
| FP32 | 1 | 10.55 | 94.7 | 10.6 | 42 MB |
| FP32 | 8 | 5.19 | 192.8 | 41.5 | 232 MB |
| FP32 | 16 | 5.09 | 196.5 | 81.4 | 333 MB |
| FP32 | 32 | 6.55 | 152.6 | 209.7 | 643 MB |
| FP32 | 64 | 6.64 | 150.5 | 425.1 | 1260 MB |
| FP16 | 1 | 9.84 | 101.7 | 9.8 | 34 MB |
| FP16 | 8 | 4.20 | 238.2 | 33.6 | 95 MB |
| FP16 | 16 | 3.98 | 251.4 | 63.7 | 166 MB |
| FP16 | 32 | 4.38 | 228.5 | 140.0 | 307 MB |
| FP16 | 64 | 4.36 | 229.1 | 279.3 | 590 MB |

### TensorRT FP16

| Batch | ms/image | images/sec | vs baseline |
| --- | --- | --- | --- |
| 1 | 7.83 | 127.7 | 1.41× |
| 8 | 4.35 | 229.6 | 2.54× |
| 16 | 4.50 | 222.1 | 2.45× |

During export, **297 of 321 nodes (92.5%)** were converted to FP16. Nodes in the detection head that handle large values (anchor counts and pixel coordinates up to 8400) were kept in FP32 to preserve precision. This is **mixed precision**.

### Accuracy check (FP32 vs FP16)

| Precision | Objects detected | Confidence scores |
| --- | --- | --- |
| FP32 | 6 | 0.873, 0.866, 0.853, 0.825, 0.261, 0.255 |
| FP16 | 6 | 0.873, 0.866, 0.853, 0.825, 0.261, 0.255 |

The detections were identical, so FP16 caused no accuracy loss.

### Pipeline stage breakdown (batch 16, averaged over 10 runs)

| Stage (ms per image) | PyTorch FP16 | TensorRT FP16 |
| --- | --- | --- |
| Preprocess (CPU) | 3.09 | 1.63 |
| Inference (GPU) | 2.22 | 0.95 |
| Postprocess | 2.01 | 1.01 |

---

## Key Findings

1. **Batching gave the largest gain.** Going from batch 1 to batch 8 roughly doubled throughput. A single small image leaves most GPU cores idle, and per-call overhead dominates.

2. **Throughput peaked at batch 16, then dropped.** The GPU still had plenty of free memory (at most 1.3 GB of 15 GB used), so the bottleneck shifted away from GPU compute, likely to CPU preprocessing and CPU→GPU data transfer.

3. **The model is only ~30% of the runtime.** At batch 16 with PyTorch FP16, about 70% of the time is spent in CPU preprocessing and postprocessing. This limits how much any model-level optimization can help overall (**Amdahl's law**).

4. **FP16 helps more at larger batches.** At batch 1 it was 1.07× faster than FP32; at batch 16, 1.28×. With bigger batches, GPU math is a larger share of total time.

5. **FP16 halves memory use.** At batch 64: 1260 MB (FP32) vs 590 MB (FP16).

6. **TensorRT cut GPU inference time by 2.3×** (2.22 → 0.95 ms per image) through layer fusion and GPU-specific kernel selection. End-to-end gains were smaller because inference was already the minority of the runtime.

7. **The best configuration depends on the use case:**
   - **Real-time / lowest latency:** TensorRT FP16, batch 1, at 7.83 ms per image
   - **Offline / highest throughput:** PyTorch FP16, batch 16, at 251.4 images/sec

   Batching trades latency for throughput: a batch of 16 takes 63.7 ms, so a live video app would wait for frames to accumulate.

## Limitations

- Per-stage timings and end-to-end timings don't fully agree. Timing each stage separately adds GPU synchronization points, which prevents CPU and GPU work from overlapping. A timeline profiler such as **NVIDIA Nsight Systems** would give a more exact picture.
- Benchmarks used a single repeated image. Real workloads with varied image sizes may behave differently.
- The TensorRT engine was built with dynamic batch size; fixed-size engines may perform slightly better.

## Future Work

- Move preprocessing to the GPU to remove the CPU bottleneck
- Overlap CPU and GPU work with asynchronous pipelines (CUDA streams)
- Profile with NVIDIA Nsight Systems
- Try INT8 quantization with calibration
- Repeat on larger YOLOv8 variants (s, m, l), where GPU compute is a bigger share of runtime

---

## How to Run

1. Open `yolov8_gpu_optimization.ipynb` in [Google Colab](https://colab.research.google.com).
2. Go to **Runtime → Change runtime type → T4 GPU**.
3. Run all cells in order.

The TensorRT export takes about 5–10 minutes. Results may vary slightly between runs and Colab sessions.

## Tech Stack

Python · PyTorch · Ultralytics YOLOv8 · ONNX · TensorRT · OpenCV · Google Colab (NVIDIA T4)
