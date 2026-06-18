# Boltzmannator

An interactive visualiser for one‑dimensional **normalising flows**. Choose a
latent distribution *p_z(z)*, shape it with a parametric transformation
*x = f_θ(z)*, and watch the push‑forward density *p_x(x)*. Then train the
transformation to match a target Boltzmann distribution *p\*(x)* or example
data. Built with [NiceGUI](https://nicegui.io); it runs in your web browser.

---

## What you need

- **Python 3.10 or newer** (this is required — see the warning below)
- The files in this folder:
  - `Boltzmannator.py` — the application
  - `requirements.txt` — the dependencies
  - `Boltzmannator_title.png` — the header image
  - `Boltzmannator_doc.pdf` — description of the program and the theory
  (keep `Boltzmannator.py`, `requirements.txt` and the `.png` together in
  the same folder)

> ⚠️ **Check your Python version first.** Run:
> ```bash
> python --version
> ```
> It must report **3.10 or higher**. If it shows 3.9 or lower (or the command
> is not found), see *"`Could not find a version that satisfies the
> requirement nicegui`"* under Troubleshooting — that error means your Python
> is too old, **not** that anything is wrong with the program.

## Install (once)

Open a terminal **in this folder** and run:

```bash
pip install -r requirements.txt
```

> Tip: to keep things isolated you can first create a virtual environment
> (use a Python 3.10+ interpreter):
> ```bash
> python -m venv venv
> source venv/bin/activate        # macOS / Linux
> venv\Scripts\activate           # Windows
> pip install -r requirements.txt
> ```

## Run

```bash
python Boltzmannator.py
```

Then open **http://localhost:8080** in your browser. (A browser window usually
opens automatically.)

To stop the app, press **Ctrl‑C** in the terminal.

---

## Using it in a classroom (same network)

The app also listens on the local network, so several people can use it at once,
each in their **own independent session** — your sliders, plots and training
never affect anyone else's.

1. Start the app on one machine (the "host").
2. Find that machine's local IP address:
   - macOS: `ipconfig getifaddr en0`
   - Windows: `ipconfig` (look for the IPv4 address)
   - Linux: `hostname -I`
3. Everyone on the **same Wi‑Fi / LAN** opens `http://<that-IP>:8080`
   (for example `http://10.0.0.70:8080`).

Note: each session runs on the host machine, so with very many simultaneous
training runs the host's CPU can get busy.

---

## The three tabs

- **Densities** – pick the latent distribution and its parameters, and define
  the target Boltzmann distribution *p\*(x) ∝ exp(−U(x)/kT)*.
- **Map** – choose the transformation family (Polynomial, Single‑layer
  perceptron, or Rational‑quadratic spline) and adjust its parameters.
- **Training** – sample points, generate target data, and optimise the
  transformation (energy‑based or example‑based), choosing the optimiser,
  learning rate, batch size, etc.

---

## Troubleshooting

- **`ERROR: Could not find a version that satisfies the requirement nicegui ...`**
  (pip lists only versions up to 2.x) — your Python is **older than 3.10**.
  NiceGUI 3.x, which this program needs, is only offered to Python 3.10+, so
  pip on an older Python can't see it. Fix it by using a newer Python:
  - Check what you have: `python --version` (also try `python3 --version`,
    `python3.11 --version`, `python3.12 --version`).
  - If a newer one exists, use it explicitly, e.g.:
    ```bash
    python3.12 -m venv venv
    source venv/bin/activate          # Windows: venv\Scripts\activate
    pip install -r requirements.txt
    python Boltzmannator.py
    ```
  - Otherwise install Python 3.10+ from <https://www.python.org/downloads/>
    (or via Anaconda/Homebrew) and repeat.
- **"Address already in use"** — an old copy is still running. Stop it:
  - macOS / Linux: `pkill -f Boltzmannator.py`
  - or just close the other terminal / restart your machine.
- **Blank page or missing image** — make sure `Boltzmannator_title.png` is in
  the same folder as `Boltzmannator.py`.
- **Want a native desktop window** instead of a browser tab? Install the
  optional dependency and run again:
  ```bash
  pip install pywebview
  python Boltzmannator.py
  ```

---

## License & credits

Released under the **MIT License** (see `LICENSE`), Copyright (c) 2026
Christoph Dellago.

The header images are derived from a public-domain 1902 photograph of Ludwig
Boltzmann ([Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Boltzmann2.jpg));
see `CREDITS.md`.
