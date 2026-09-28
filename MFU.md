# Model FLOPs utilization on GB10

MFU compares estimated model computation per second with a device's theoretical
compute throughput at the relevant precision:

```
MFU = estimated model FLOPs per training iteration / iteration seconds / peak FLOPs per second
```

`model.py` keeps nanoGPT's existing PaLM-style FLOPs estimate. The hardware peak
is now supplied by `train.py` instead of hardcoding the A100's 312 TFLOPS.

## GB10 reference value

For a CUDA device whose name contains `GB10`, using `bfloat16` or `float16`, the
automatic reference is **125 TFLOPS per device**. This is a **nominal estimate,
not a separately published NVIDIA BF16 specification**.

[NVIDIA's Spark hardware table](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)
advertises up to 1,000 TFLOPS at **FP4 with sparsity**. Dense BF16 training cannot
use that number directly. The 125 estimate assumes a factor of two for removing
sparsity and a further factor of four from FP4 to BF16/FP16:
`1000 / 2 / 4 = 125`. Precision and accumulator throughput ratios are hardware
dependent; this conversion is an assumption rather than a universal rule.
[Independent GB10 cuBLAS measurements](https://github.com/sxuff/gb10-dissection)
report roughly 98 TFLOPS for BF16, consistent with (but not proving) this reference.
Revise the override if a better established GB10 specification becomes available.

The denominator is fixed, not adjusted to instantaneous clocks, temperature or
measured GEMM throughput. Those affect achieved performance. Normalizing against
a measured GEMM benchmark would produce a different metric.

## Configuration

Existing BF16 commands on GB10 automatically use the new reference. You can also
set it explicitly, after any other training configuration:

```sh
python train.py config/train_gpt2.py --mfu_peak_tflops=125.0
```

`mfu_peak_tflops` is a float (include `.0` for integer CLI values). `0.0` means
automatic selection. For other devices or float32 training, automatic selection
leaves MFU unavailable (`nan`) until you provide a positive finite override
appropriate to the device, precision, and dense compute path. For an A100 BF16
reference matching upstream nanoGPT, pass `--mfu_peak_tflops=312.0`.

The resolved peak is printed at startup and saved in the checkpoint's `config`
and W&B run configuration. It is a **per-device** peak: nanoGPT already divides
gradient accumulation among DDP ranks, so do not multiply it by the GPU count.
On resume it is resolved from the current run's device/config, allowing a
checkpoint to move to another machine.

With the same model and iteration time, the old A100-normalized reading scales
by `312 / 125 = 2.496`. For example, 4.36% becomes about **10.88%**; training speed
and model quality have not changed.

## Measurement window

`time` in the training log is now the **average milliseconds per optimizer
update over the measured window**, whose update count is printed alongside it.
The window normally ends every `log_interval` iterations. A final partial window
is also logged when training finishes.

`TrainingTimer` uses a monotonic wall clock and synchronizes the selected CUDA
device (or MPS device) at segment boundaries. The end synchronization ensures
that queued GPU work has completed before reading elapsed time. There is no
extra synchronization between ordinary updates within a window.

Before evaluation the timer pauses, keeping elapsed training time and the update
count. It resumes after evaluation/checkpointing, even when `eval_interval` and
`log_interval` are not aligned. Console and W&B evaluation logging are excluded
as well. CPU batch preparation, transfers, forward/backward, optimizer work and
normal training stalls remain part of the measured training throughput.

The first five local updates are discarded from MFU timing, including after
resume. During warmup MFU is `nan`, rather than the old `-100%` placeholder.
Compilation during those updates is excluded; any later recompilation that
happens during training still counts. The initial batch fetch is outside the
window, while subsequent batch prefetches are included.

MFU uses the FLOPs for **all updates in the window divided by their combined
training time**, followed by the existing exponential smoothing (90% previous,
10% current). In DDP this is rank 0's per-device training throughput, including
its normal communication waits; it is not a cross-rank average. Evaluation logs
in W&B retain the most recently completed window's smoothed MFU.

MFU remains an approximation, not `nvidia-smi` GPU utilization. The PaLM-style
FLOPs estimate and assumed hardware peak still limit its absolute accuracy.
These measurements describe training throughput, not total job throughput
including evaluation and checkpointing.
