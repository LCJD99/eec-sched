# Synthetic device-count scaling model

This generator creates 10, 20, or 50-device scenarios with `x` Configurations per tool. Its output is a **synthetic scenario**, separate from the v1.1 Profiling Database Snapshot, which requires exactly three devices and six direct links. The current evaluator cannot score the sparse scenarios until multi-hop routing and monetary cost are added to its contract.

## Device capacity and Cost

The three anchor GPU models have deliberately chosen **relative experimental units**:

| GPU | Compute capacity | GPU memory | Cost per hour |
| --- | ---: | ---: | ---: |
| RTX 5060 | 1.0 | 8 GiB | 1.0 |
| RTX 3090 | 1.8 | 24 GiB | 2.4 |
| RTX 4090 | 3.2 | 24 GiB | 5.0 |

The capacity and Cost numbers are scenario assumptions. They are neither measured inference throughput nor currency rental prices. The GPU memory sizes follow NVIDIA's published specifications for the [RTX 5060](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5060-family/), [RTX 3090](https://www.nvidia.com/en-eu/geforce/graphics-cards/30-series/rtx-3090/), and [RTX 4090](https://www.nvidia.com/en-eu/geforce/graphics-cards/40-series/). Those specifications justify treating the GPUs as distinct hardware classes, but do not determine one universal inference-speed ratio across tools.

By default, snapshot device IDs map as `device=rtx5060`, `edge=rtx3090`, and `cloud=rtx4090`; `--gpu` overrides the mapping. Each additional device gets a unique compute capacity on a log grid within the three anchor capacities. Its Cost is interpolated between adjacent anchors in log capacity–log Cost space. For all devices, both Cost and **Cost per unit of compute capacity** increase with capacity. Thus a faster device has a higher price, and for equal work its execution Cost is also higher.

## Configuration work and latency

For each tool, the source three-device snapshot supplies a reference Configuration and its observed latencies `T_{t,d}`. Given assigned device capacities `s_d`, the generator estimates one reference compute demand `W_{t,0}` by averaging `log(T_{t,d} s_d)`. It records the root mean squared log residual. This is the only parameter fitted from execution observations; if the three observations do not obey a shared-capacity model closely, the residual reveals the mismatch.

For `x` Configurations, define evenly spaced demand levels `z_j=j/(x−1)` for `j=0,…,x−1`; if `x=1`, use `z_0=0`. Then

\[
W_t(z)=W_{t,0}(1+\beta z^{\nu}),\qquad
T_{t,d}(z)=\frac{W_t(z)}{s_d},\qquad
C_{t,d}(z)=\frac{T_{t,d}(z)\,p_d}{3{,}600{,}000}.
\]

Here `T` is in milliseconds, `p_d` is relative Cost per hour, and default `β=0.50`, `ν=1.4`. Because `p_d/s_d` strictly increases with capacity, higher-capacity devices are faster but have higher execution Cost for the same Configuration. The generated `compute_demand` is explicit in each Configuration. Its GPU-memory requirement is `M_t(z)=M_{t,0}(1+γz)` with default `γ=0.25`; it is marked incompatible on devices without enough GPU memory. Quality lower bound is `min(1,q_{t,0}+δz)` with default `δ=0.10`.

The work, memory, and quality changes across demand levels are **declared mathematical assumptions**, not true model settings or measured Quality Profiles. With only a reference Configuration, their curve shapes cannot be identified from the source snapshot. All outputs remain marked `synthetic`, even if the source snapshot is measured.

## Shared sparse network

Each device has a `device_ordinal`. The three anchors use ordinal `0,1,2`; generated IDs `sim-003`, `sim-004`, etc. use their numeric suffix as ordinal. Neighboring ordinals always connect in both directions. Other pairs receive a link with probability `0.45 exp(-distance/4)`, using a fixed seed. For a present directed link at ordinal distance `h`,

\[
B(h)=B_0 e^{-\lambda h}U,\qquad
D(h)=D_0(1+\lambda h),\qquad U\sim\operatorname{Uniform}(0.85,1.15).
\]

`B_0` is the geometric mean bandwidth and `D_0` the mean propagation delay from the source snapshot; default `λ=0.12`. Therefore more distant IDs are less likely to connect and, when they do, have lower expected bandwidth. The graph is directed, reproducible, sparse, and connected by its neighbor links. It is shared by all tools. Missing direct links require multi-hop routing during future scheduling evaluation.

## Generate scenarios

```bash
uv run python scripts/generate_device_scaling_scenarios.py \
  --configurations-per-tool 5
```

The default input is the repository's synthetic three-device fixture. Supply `--snapshot` and `--schema` to use another structurally validated snapshot. For a different hardware mapping, add `--gpu device=rtx4090 --gpu edge=rtx5060 --gpu cloud=rtx3090`. Defaults generate 10, 20, and 50 devices under `runs/device-scaling-scenarios/`. The output records assumptions and fit residuals; it does not calculate or verify a scenario SHA-256 digest. Vary work and network parameters in sensitivity analysis before drawing scaling conclusions.
