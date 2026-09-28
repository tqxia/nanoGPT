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

MFU remains an approximation, not `nvidia-smi` GPU utilization. This change only
fixes the configurable hardware denominator. The existing FLOPs estimate,
wall-clock timing, five-iteration warmup and exponential smoothing remain;
evaluation/checkpoint overhead and asynchronous CUDA execution can affect the
reported iteration time. Compare steady-state iterations under the same setup.
