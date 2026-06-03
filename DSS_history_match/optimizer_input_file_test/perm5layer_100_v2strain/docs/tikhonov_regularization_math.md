# Tikhonov Regularization Math

This note documents the objective and regularization terms used by
`inv/102_optimization_runner.py`.

## Unknown Parameter

The inversion variable is the layer-wise log-permeability vector

```math
\alpha =
\begin{bmatrix}
\alpha_1 & \alpha_2 & \cdots & \alpha_N
\end{bmatrix}^T,
\qquad N = 200.
```

Each component is the base-10 logarithm of the horizontal permeability for one
0.5 m layer:

```math
k_i = 10^{\alpha_i}.
```

In the Python code, this vector is named `x`.

## Data Misfit Objective

MOOSE computes the strain objective from the `OptimizationData` reporter in
`optimize.i` and `forward_and_adjoint.i`. The observation variable is
`strain_yy`.

Conceptually, the raw data misfit is

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

- `d_m` is the measured synthetic `strain_yy` value.
- `s_m(alpha)` is the simulated `strain_yy` value at the same receiver and time.
- `M` is the number of strain observations.

In the Python code this value is parsed as:

```python
obj_val = OptimizationReporter/objective_value
```

The MOOSE adjoint provides the raw gradient

```math
\nabla J_\mathrm{data}(\alpha),
```

which is read from columns:

```python
grad_perm_1, grad_perm_2, ..., grad_perm_200
```

and stored as `grad_array`.

## Regularization Used In `102`

The current `102_optimization_runner.py` uses **pure zero-order Tikhonov
regularization**, also called **L2 damping** or **ridge regularization**.

It does **not** use a layer-to-layer smoothness term anymore.

The regularization term is

```math
J_\mathrm{Tik}(\alpha)
=
\beta_\mathrm{Tik}
\sum_{i=1}^{N}
\left(\alpha_i - \alpha_\mathrm{ref}\right)^2.
```

In Python:

```python
ALPHA_REFERENCE = -18.0
BETA_TIKHONOV = 1e-9

deviation = x - ALPHA_REFERENCE
reg_obj = BETA_TIKHONOV * np.sum(deviation**2)
```

where:

- `alpha_ref = -18.0` is the reference/background log-permeability.
- `beta_Tik = 1e-9` controls how strongly the inversion is pulled toward that
  reference.

This term penalizes the **squared absolute departure from the reference model**.
It is called "zero-order" because it acts directly on `alpha_i`, not on a
derivative or neighbor difference.

## Regularization Gradient

For one component,

```math
J_{\mathrm{Tik},i}
=
\beta_\mathrm{Tik}
\left(\alpha_i - \alpha_\mathrm{ref}\right)^2.
```

Therefore

```math
\frac{\partial J_\mathrm{Tik}}{\partial \alpha_i}
=
2\beta_\mathrm{Tik}
\left(\alpha_i - \alpha_\mathrm{ref}\right).
```

The Python implementation is

```python
reg_grad = 2 * BETA_TIKHONOV * deviation
```

This is the exact derivative of the quadratic penalty.

## What This Is Not

The phrase "penalize the absolute model value" can mean two different things:

### L2 Damping / Zero-Order Tikhonov

This is what `102_optimization_runner.py` currently uses:

```math
J_\mathrm{L2}
=
\beta
\sum_i
(\alpha_i - \alpha_\mathrm{ref})^2.
```

It is smooth and differentiable, and its gradient is linear in the model
departure.

### L1 / Lasso

A literal sum of absolute values would be

```math
J_\mathrm{L1}
=
\lambda
\sum_i
\left|\alpha_i - \alpha_\mathrm{ref}\right|.
```

That is **not Tikhonov**. It is usually called **L1 regularization**, **lasso**,
or a sparsity penalty. It is nonsmooth at zero and tends to drive many
components exactly back to the reference.

## Removed Old Smoothness Term

An older version of `102_optimization_runner.py` also included first-order
Tikhonov smoothness:

```math
J_\mathrm{smooth}(\alpha)
=
\beta_\mathrm{smooth}
\sum_{i=1}^{N-1}
\left(\alpha_{i+1} - \alpha_i\right)^2.
```

That term penalizes layer-to-layer jumps. It was removed from `102`, so the
current regularization does **not** penalize roughness directly. Sharp jumps are
penalized only indirectly if they require layers to move far from
`alpha_ref = -18`.

## Total Objective Sent To SciPy

The unscaled total objective is

```math
J(\alpha)
=
J_\mathrm{data}(\alpha)
+
J_\mathrm{Tik}(\alpha).
```

The unscaled total gradient is

```math
\nabla J(\alpha)
=
\nabla J_\mathrm{data}(\alpha)
+
\nabla J_\mathrm{Tik}(\alpha).
```

In Python:

```python
deviation = x - ALPHA_REFERENCE
reg_obj = BETA_TIKHONOV * np.sum(deviation**2)
reg_grad = 2 * BETA_TIKHONOV * deviation

total_obj = float(obj_val) + reg_obj
total_grad = grad_array + reg_grad
```

Before returning to SciPy, both objective and gradient are multiplied by the
same constant:

```python
SCALE_FACTOR = 1e6

scaled_obj = total_obj * SCALE_FACTOR
scaled_grad = total_grad * SCALE_FACTOR
```

This scaling does not change the minimizer. It only changes the numerical scale
seen by L-BFGS-B. The scaling is mathematically consistent because

```math
\nabla \left(c J\right) = c \nabla J.
```

## Why The Regularization Calculation Is Correct

The Tikhonov calculation is correct because:

1. The Python objective directly implements the mathematical definition:

```python
BETA_TIKHONOV * np.sum((x - ALPHA_REFERENCE)**2)
```

2. The Python gradient is the exact derivative of that quadratic objective:

```math
\frac{d}{dx} \beta (x-a)^2 = 2\beta(x-a).
```

3. The same `SCALE_FACTOR` is applied to both objective and gradient, so SciPy
   receives a consistent scaled objective/gradient pair.

4. A finite-difference check of the regularization term alone can be done
   without MOOSE:

```python
eps = 1e-6
p = np.random.randn(len(x))
p /= np.linalg.norm(p)

def tik(x):
    return BETA_TIKHONOV * np.sum((x - ALPHA_REFERENCE)**2)

fd = (tik(x + eps * p) - tik(x - eps * p)) / (2 * eps)
adj = p @ (2 * BETA_TIKHONOV * (x - ALPHA_REFERENCE))
```

For the full objective, a finite-difference check should compare the directional
derivative

```math
\frac{J(\alpha + \epsilon p) - J(\alpha - \epsilon p)}{2\epsilon}
```

against

```math
p^T \nabla J(\alpha).
```

## Interpretation

Zero-order Tikhonov favors profiles close to `alpha = -18`. It can suppress
large model departures, but it does not directly enforce smoothness or sharp
block boundaries.

Because the penalty is quadratic, a large anomaly over a few layers can be more
expensive than a smaller anomaly spread over more layers, depending on the data
misfit. This is why L2 damping can shrink anomaly amplitudes.
