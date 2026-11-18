# Track-Before-Detect with a Particle Filter: Joint Tracking and PSF Classification

MATLAB implementation of **track-before-detect (TBD)** for a weak target moving through a 1-D sensor array.
Instead of thresholding each scan into detections and then tracking, the particle filter works on the raw,
noisy measurements. It jointly estimates the target's **position and velocity** and **which point-spread
function (PSF) class** produced the returns.

## Problem setup

| Item | Value |
| --- | --- |
| Measurement | 200-cell complex array per scan, 50 scans, T = 0.5 s |
| Target return | Gaussian PSF exp(−α (x − x_t)²) × complex amplitude q (Swerling-type fluctuation) |
| PSF classes | 2, with 3 dB widths of 2 and 3 cells; prior P = [0.4, 0.6] |
| Motion model | Constant velocity, process noise σ_w = 2 |
| SNR | 20 dB, σ_n = 1 |
| Particles | 5 000 per class |

## Algorithm

For each scan *k* and each class *c*:

1. Evaluate each particle's likelihood with the complex amplitude **marginalised analytically**:
   log ℓ = −log γ + (σ_q² / σ_n²) |aᴴ z|² / γ, where γ = ‖a‖² σ_q² + σ_n².
2. Normalise the weights. The class evidence is the log-sum-exp of the particle likelihoods.
3. Update the class posterior P(c | z₁:ₖ) recursively.
4. Combine the per-class weighted means into the state estimate, weighted by the class posterior.
5. Resample (`resample.m`, multinomial) and propagate through the constant-velocity model.

`ParticleFilterPlayground.m` is a warm-up that compares a Kalman filter and a particle filter on a linear
Gaussian tracking problem.

## Running

```matlab
TBD_v4                     % main experiment; figures are saved to ./Results/
ParticleFilterPlayground   % Kalman vs. particle filter sanity check
```

Requires MATLAB R2020a+ (for `exportgraphics`).

## Notes

- In `TBD_v4.m` the propagation step computes `noise = chol(W)' * randn(2, N)`, where `W` already includes
  σ_w², and then adds `sigmaw * noise`. That scales the process noise by σ_w twice. Using `+ noise` matches
  the stated motion model.

Course project (SSDP), MSc Electrical Engineering, TU Delft.
