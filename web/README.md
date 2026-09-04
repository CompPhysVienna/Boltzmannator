# Boltzmannator — JavaScript edition

An interactive visualiser for one-dimensional **normalising flows**, running
entirely in your web browser. Choose a latent distribution *p_z(z)*, shape it
with a parametric transformation *x = f_θ(z)*, and watch the push-forward
density *p_x(x)*. Then train the transformation to match a target Boltzmann
distribution *p\*(x)* or example data.

This is a faithful port of the Python/NiceGUI Boltzmannator (in the parent
folder) to plain HTML, CSS and JavaScript. **Nothing runs on a server** — all
numerics (densities, transformations, analytic gradients, Adam/SGD/RMSprop
training) execute in the browser of the local machine.

## Run

Open **`index.html`** in any modern browser — double-click it, or drag it
onto a browser window. That's all: no Python, no installation, no server.

If you host the folder on a web server (e.g. GitHub Pages), visitors can use
the app directly at the URL, or download the folder and run it locally.

## Files

| File | Purpose |
|---|---|
| `index.html` | page skeleton |
| `style.css`  | modern light/dark styling |
| `js/math.js` | numerics: distributions, transformations, losses, gradients |
| `js/plots.js`| canvas plotting toolkit (axes, ticks, histograms, legends) |
| `js/app.js`  | application: state, controls, figure, training loop |
| `img/`       | header pictures |

Keep these files together in one folder; there are no external dependencies.

## Fidelity to the Python version

The JavaScript numerics were validated against the Python original on
identical inputs: latent PDFs/log-PDFs, Gaussian KDE, energy-based and
example-based losses, and the analytic/finite-difference gradients of all
three transformation families (polynomial, single-layer perceptron,
rational-quadratic spline) agree to 13+ significant digits.

## Features

- Latent distributions: Gaussian, Uniform, Laplace, Bimodal
- Transformations: polynomial, single-layer perceptron (K = 1…8 sigmoids),
  monotone rational-quadratic spline (K = 2…4 bins)
- Boltzmann target *p\*(x) ∝ exp(−U(x)/kT)* with quartic potential,
  exact transformation overlay, importance weights with N_eff
- Energy-based and example-based training with Adam, SGD, SGD+momentum and
  RMSprop; live animated figure, loss decomposition, best-parameter finish
- Sampling and target-data generation, mapping lines, auto-rescale lock,
  dark/light mode, header-picture chooser
- Responsive layout: on a phone in portrait the plot sits as a live band
  above the scrollable controls; in landscape it returns to the desktop
  side-by-side view. Touch targets are enlarged on touch devices.

© 2026 Christoph Dellago
