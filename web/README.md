# Boltzmannator, JavaScript edition (v 1.1)

An interactive visualiser for one-dimensional **normalizing flows** and
**Boltzmann generators**, running entirely in the web browser. A simple
latent density *p_z(z)* is reshaped by a parametric, invertible
transformation *x = f_θ(z)* into the push-forward density *p_x(x)*, which
can be trained to match a target Boltzmann distribution
*p\*(x) ∝ exp(−U(x)/k_BT)* or example data.

This is a port of the Python/NiceGUI Boltzmannator (in the parent folder)
to plain HTML, CSS and JavaScript. **Nothing runs on a server**: all
numerics (densities, transformations, analytic gradients, training)
execute in the browser of the local machine.

## Run

Open **`index.html`** in any modern browser; double-click it or drag it
onto a browser window. No Python, no installation, no server.

Hosted on a web server (for example GitHub Pages), the app can be used
directly at its URL. It is installable as a web app (Add to Home Screen)
and works offline after the first visit.

## The interface

- **Densities tab**: latent distribution (Gaussian, Uniform, Laplace,
  Bimodal) with mean and width; target defined by the quartic potential
  U(x) = u₁x + u₂x² + u₃x³ + u₄x⁴ and temperature kT; overlays for the
  target, the exact transformation, the potential U(x) and the
  construction of the exact map from the two cumulative distribution
  functions.
- **Map tab**: transformation families (polynomial, single-layer
  perceptron with K = 1…8 sigmoid units, monotone rational-quadratic
  spline with K = 2…4 bins); mapping lines and transport bands of equal
  probability mass; an animation that morphs the whole figure between the
  identity map and the current map along the displacement interpolation
  (the one-dimensional optimal-transport path), or replays the parameter
  history of the last training run.
- **Training tab**: sampling and target-data generation; energy-based
  (reverse Kullback-Leibler) and example-based (maximum-likelihood)
  training with Adam, SGD, SGD+momentum or RMSprop and analytic gradients
  for all three transformation families; live figure updates, loss
  decomposition, training trails and a loss slice through parameter
  space.
- **Figure**: seven panels (latent density, map, inverse Jacobian,
  transformed density, sample histograms, training loss) plus a live
  readout of the variational free energy βF_flow, the exact
  βF_exact = −ln Z and their difference KL(p_x ∥ p\*); importance
  weights with the effective sample size N_eff; a pointer crosshair with
  local values of z, x, J and the densities (tap on touch screens).
- **Documentation in the app**: a Theory page (Boltzmann generators,
  change of variables, both training objectives, the Gibbs-Bogoliubov
  inequality, reweighting, references) and a Help page with one-click
  experiment presets, both with LaTeX-typeset equations; explanatory
  boxes appear when hovering over any control, panel or readout.
- **Utilities**: PNG download of the current figure; share links that
  encode the complete state of the app in the URL; light and dark mode;
  responsive layout for phones and tablets.

## Files

| File | Purpose |
|---|---|
| `index.html` | page skeleton, Theory and Help pages |
| `style.css`  | light/dark styling |
| `js/math.js` | numerics: distributions, transformations, losses, analytic gradients |
| `js/plots.js`| canvas plotting toolkit (axes, ticks, histograms, legends) |
| `js/app.js`  | application: state, controls, figure, training, animation |
| `vendor/katex/` | KaTeX (local copy) for the typeset equations |
| `img/`       | header pictures and app icons |
| `manifest.webmanifest`, `sw.js` | installable web app, offline support |

Keep these files together in one folder; there are no external
dependencies and no network access is required.

## Fidelity and validation

The JavaScript numerics were validated against the Python original on
identical inputs: latent PDFs and log-PDFs, Gaussian KDE, energy-based
and example-based losses, and the gradients of all three transformation
families agree to 13+ significant digits. The analytic
rational-quadratic-spline gradients were additionally verified against
finite differences (relative error below 10⁻⁷ away from bin
boundaries).

Boltzmannator v 1.1, © 2026 Christoph Dellago
