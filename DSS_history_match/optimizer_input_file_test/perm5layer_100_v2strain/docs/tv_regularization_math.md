# Total Variation Regularization Math

This note documents the objective and regularization terms used by
`inv/103_optimization_runner_TV.py`.

## Unknown Parameter

The inversion variable is the layer-wise log-permeability vector

```math
\alpha =
\begin{bmatrix}
\alpha_1 & \alpha_2 & \cdots & \alpha_N
\end{bmatrix}^T,
\qquad N = 200.
```

Each component is the base-10 logarithm of horizontal permeability:

```math
k_i = 10^{\alpha_i}.
```

In the Python code, this vector is named `x`.

## Data Misfit Objective

MOOSE computes the strain misfit objective using `OptimizationData` with
`variable = 'strain_yy'`.

Conceptually, the raw data objective is

```math
J_\mathrm{data}(\alpha)
=
\frac{1}{2}
\sum_{m=1}^{M}
\left(
s_m(\alpha) - d_m
\right)^2,
```

where:

- `d_m` is the measured synthetic `strain_yy`.
- `s_m(alpha)` is the simulated `strain_yy`.
- `M` is the number of receiver-time measurements.

In Python this value is parsed as:

```python
obj_val = OptimizationReporter/objective_value
```

The raw data gradient is read from:

```python
grad_perm_1, grad_perm_2, ..., grad_perm_200
```

and stored as `grad_array`.

## Total Variation Regularization

The intended nonsmooth total variation penalty is

```math
J_\mathrm{TV,nonsmooth}(\alpha)
=
\beta_\mathrm{TV}
\sum_{i=1}^{N-1}
\left|
\alpha_{i+1} - \alpha_i
\right|.
```

This favors piecewise-constant profiles because it penalizes jumps, but not
constant intervals.

The Python code uses a differentiable smoothed absolute value:

```math
|z|
\approx
\sqrt{z^2 + \delta_\mathrm{TV}^2}
-
\delta_\mathrm{TV}.
```

Therefore the implemented TV objective is

```math
J_\mathrm{TV}(\alpha)
=
\beta_\mathrm{TV}
\sum_{i=1}^{N-1}
\left[
\sqrt{
\left(\alpha_{i+1} - \alpha_i\right)^2
+
\delta_\mathrm{TV}^2
}
-
\delta_\mathrm{TV}
\right].
```

In Python:

```python
BETA_TV = 1e-7
DELTA_TV = 0.05

diffs = np.diff(x)
denom = np.sqrt(diffs * diffs + DELTA_TV * DELTA_TV)
obj = BETA_TV * np.sum(denom - DELTA_TV)
```

The subtraction of `DELTA_TV` makes the regularization zero when the model is
constant:

```math
\alpha_1 = \alpha_2 = \cdots = \alpha_N
\quad \Rightarrow \quad
J_\mathrm{TV}(\alpha) = 0.
```

This subtraction does not change the gradient, because it is a constant with
respect to `alpha`.

## TV Gradient

Define

```math
\Delta_i = \alpha_{i+1} - \alpha_i
```

and

```math
\phi(\Delta_i)
=
\sqrt{\Delta_i^2 + \delta_\mathrm{TV}^2}
-
\delta_\mathrm{TV}.
```

Then

```math
J_\mathrm{TV}
=
\beta_\mathrm{TV}
\sum_{i=1}^{N-1}
\phi(\Delta_i).
```

The derivative of the smoothed absolute value is

```math
\phi'(\Delta_i)
=
\frac{\Delta_i}
{\sqrt{\Delta_i^2 + \delta_\mathrm{TV}^2}}.
```

For an interior layer `j`, `alpha_j` appears in two differences:

```math
\Delta_{j-1} = \alpha_j - \alpha_{j-1},
\qquad
\Delta_j = \alpha_{j+1} - \alpha_j.
```

Therefore

```math
\frac{\partial J_\mathrm{TV}}{\partial \alpha_j}
=
\beta_\mathrm{TV}
\left[
\frac{\Delta_{j-1}}
{\sqrt{\Delta_{j-1}^2 + \delta_\mathrm{TV}^2}}
-
\frac{\Delta_j}
{\sqrt{\Delta_j^2 + \delta_\mathrm{TV}^2}}
\right].
```

The boundary terms are one-sided:

```math
\frac{\partial J_\mathrm{TV}}{\partial \alpha_1}
=
-\beta_\mathrm{TV}
\frac{\Delta_1}
{\sqrt{\Delta_1^2 + \delta_\mathrm{TV}^2}},
```

```math
\frac{\partial J_\mathrm{TV}}{\partial \alpha_N}
=
\beta_\mathrm{TV}
\frac{\Delta_{N-1}}
{\sqrt{\Delta_{N-1}^2 + \delta_\mathrm{TV}^2}}.
```

The Python implementation is:

```python
g = diffs / denom
grad = np.zeros_like(x)
grad[1:]  += BETA_TV * g
grad[:-1] -= BETA_TV * g
```

Each difference `diffs[i] = x[i+1] - x[i]` contributes:

- `+BETA_TV * g[i]` to layer `i+1`.
- `-BETA_TV * g[i]` to layer `i`.

This is exactly the vectorized form of the derivative above.

## Total Objective Sent To SciPy

The unscaled total objective is

```math
J(\alpha)
=
J_\mathrm{data}(\alpha)
+
J_\mathrm{TV}(\alpha).
```

The unscaled total gradient is

```math
\nabla J(\alpha)
=
\nabla J_\mathrm{data}(\alpha)
+
\nabla J_\mathrm{TV}(\alpha).
```

In Python:

```python
reg_obj, reg_grad = tv_obj_and_grad(x)
total_obj = float(obj_val) + reg_obj
total_grad = grad_array + reg_grad
```

The returned values are scaled:

```python
SCALE_FACTOR = 1e6

scaled_obj = total_obj * SCALE_FACTOR
scaled_grad = total_grad * SCALE_FACTOR
```

This scaling does not change the minimizer because the objective and gradient
are multiplied by the same positive constant.

## Why The TV Calculation Is Correct

The TV calculation is correct because:

1. The implemented objective is a standard differentiable approximation to
   total variation:

```math
|z| \approx \sqrt{z^2 + \delta^2} - \delta.
```

2. The derivative is analytically

```math
\frac{d}{dz}
\left(
\sqrt{z^2 + \delta^2}
-
\delta
\right)
=
\frac{z}{\sqrt{z^2 + \delta^2}}.
```

3. The Python line

```python
g = diffs / denom
```

is exactly this derivative evaluated at every layer difference.

4. Each layer difference depends on two adjacent alphas with opposite signs.
   Therefore the gradient contribution must be added to one layer and
   subtracted from the other:

```python
grad[1:]  += BETA_TV * g
grad[:-1] -= BETA_TV * g
```

5. The same `SCALE_FACTOR` is applied to objective and gradient, so SciPy gets
   a consistent objective-gradient pair:

```math
\nabla(cJ) = c\nabla J.
```

6. A finite-difference check for the TV regularization alone should compare

```math
\frac{J_\mathrm{TV}(\alpha + \epsilon p)
-
J_\mathrm{TV}(\alpha - \epsilon p)}
{2\epsilon}
```

with

```math
p^T \nabla J_\mathrm{TV}(\alpha).
```

This check can be done without MOOSE because `tv_obj_and_grad(x)` is a pure
NumPy function.

## Important Interpretation

TV regularization penalizes total jump magnitude. It does not directly penalize
the number of active layers or the area/thickness of the SRV anomaly.

For example, compare:

- A narrow high-contrast block: fewer active layers, but large jumps.
- A broad low-contrast anomaly: more active layers, but smaller total jump.

Plain TV can prefer the broad low-contrast profile if it has lower total jump
and still fits the strain data well. This is why TV alone can produce
lower-than-expected permeability over a larger SRV area.

To penalize broad SRV area, TV would need another term, for example an
area/sparsity penalty:

```math
J_\mathrm{area}(\alpha)
=
\beta_\mathrm{area}
\sum_{i=1}^{N}
\left|\alpha_i - \alpha_\mathrm{background}\right|.
```

Then the regularization would become

```math
J_\mathrm{reg}
=
J_\mathrm{TV}
+
J_\mathrm{area}.
```

TV would encourage blockiness, while the area term would discourage spreading
the anomaly across too many layers.
