# D2-A-E003 fixed coherent representation design

## Model comparison

- **S0:** the frozen real-valued IQ `TaskStudent` from D1-E001, with six input planes `[Re(x0), Re(x1), Re(x2), Im(x0), Im(x1), Im(x2)]`.
- **S2:** deterministic 28-channel physical feature map, then a learned 1×1 convolution `28→6`, then the same `TaskStudent` body and heads. There is no raw-IQ bypass, teacher head, or additional temporal CNN.

## Feature equations

For complex input `x_c[n,r]`, define per-scene observed power

\[
q(X)=\max\left(\frac{1}{3NR}\sum_{c,n,r}|x_c[n,r]|^2,10^{-12}\right).
\]

For each channel `c` and fixed delay `τ∈{1,4,16}`:

\[
z_{c,τ}[n,r]=\frac{x_c[n+τ,r]x_c^*[n,r]}{q(X)},\quad n+τ<N.
\]

For `n+τ≥N`, both parts are zero. The nine complex sequences contribute 18 real channels.

For channel pairs `(0,1),(0,2),(1,2)`:

\[
u_{ij}[n,r]=\frac{x_i[n,r]x_j^*[n,r]}{q(X)}.
\]

These three complex sequences contribute six real channels. Add normalized instantaneous powers

\[
p_c[n,r]=|x_c[n,r]|^2/q(X),\qquad c=0,1,2,
\]

and one broadcast `log q(X)` channel. The total is `18+6+3+1=28` real channels.

## What the features preserve

Under a common phase rotation `X′=Xe^{jφ}`, each complex product is unchanged, as are the powers and `q`. Thus the complete S2 input is invariant to a public common phase rotation up to finite precision. A same-channel single tone gives

\[
x[n+τ]x^*[n]=|A|^2e^{j2πfτ/PRF},
\]

so its phase tracks slow-time frequency. Lag 1 is one-to-one over the principal PRF interval; lags 4 and 16 wrap more often and are individually aliased. The front end keeps three pairwise receive-channel phase differences through `x_i x_j*`, which can encode array geometry and angle-dependent coherence. All products use the same range cell, so there is no cross-range mixing; the original range grid remains explicit.

## What can be lost or confounded

Each correlation is quadratic. In a sum `X=H0+S`, the feature contains `F(H0)`, `F(S)`, and cross terms; it does not isolate the target additively. Scene normalization attenuates absolute scale in the first 27 channels, while `log q` provides a coarse scale cue. The 28→6 projection can discard information; its effect is part of the tested S2 representation. Multiple delays do not remove PRF ambiguity, and different mixtures can share the same finite set of second-order products. The representation cannot infer source identity in the exact-overlap D condition without a prior.

## Numerical rule

The float32/complex64 implementation is compared with a complex128 reference using relative max error `max|a−b|/max(1,max|b|)≤1e-4`. Public-phase invariance is checked on the full feature map and on task outputs. This is a forward-computation tolerance; task effects are separately evaluated in Hz, range cells, detection probability and joint success.

