"""Boltzmannator — NiceGUI normalising-flow visualiser."""

import os
import threading
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import asyncio

# Trapezoidal integration: `np.trapz` was renamed to `np.trapezoid` in NumPy
# 2.0 and removed under the old name, so pick whichever this NumPy provides.
try:
    from numpy import trapezoid as _trapz      # NumPy >= 2.0
except ImportError:
    from numpy import trapz as _trapz          # NumPy < 2.0
from nicegui import ui, app as nicegui_app, run
from nicegui.element import Element


# ── Pure-numpy helpers ────────────────────────────────────────────────────────

def gaussian_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def latent_pdf(z, mu, sigma, name):
    """Evaluate the chosen latent density at points z."""
    if name == "Gaussian":
        return gaussian_pdf(z, mu, sigma)
    elif name == "Uniform":
        a, b = mu - sigma * np.sqrt(3), mu + sigma * np.sqrt(3)
        return np.where((z >= a) & (z <= b), 1.0 / (b - a), 0.0)
    elif name == "Laplace":
        s = sigma / np.sqrt(2)
        return np.exp(-np.abs(z - mu) / s) / (2 * s)
    elif name == "Cauchy":
        return 1.0 / (np.pi * sigma * (1 + ((z - mu) / sigma) ** 2))
    elif name == "Bimodal":
        return 0.5 * gaussian_pdf(z, -mu, sigma) + \
               0.5 * gaussian_pdf(z,  mu, sigma)
    return np.zeros_like(z)


def sample_latent(n, mu, sigma, name, rng):
    """Draw n samples from the chosen latent distribution."""
    if name == "Gaussian":
        return rng.normal(mu, sigma, n)
    elif name == "Uniform":
        a, b = mu - sigma * np.sqrt(3), mu + sigma * np.sqrt(3)
        return rng.uniform(a, b, n)
    elif name == "Laplace":
        return rng.laplace(mu, sigma / np.sqrt(2), n)
    elif name == "Cauchy":
        return mu + sigma * rng.standard_cauchy(n)
    elif name == "Bimodal":
        mask = rng.random(n) < 0.5
        a = rng.normal(-mu, sigma, n)
        b = rng.normal( mu, sigma, n)
        return np.where(mask, a, b)
    return np.zeros(n)


def log_latent_pdf(z, mu, sigma, name):
    """Numerically-stable log-density of the latent distribution."""
    if name == "Gaussian":
        return (-0.5 * ((z - mu) / sigma) ** 2
                - np.log(sigma) - 0.5 * np.log(2 * np.pi))
    elif name == "Uniform":
        a, b = mu - sigma * np.sqrt(3), mu + sigma * np.sqrt(3)
        return np.where((z >= a) & (z <= b),
                        -np.log(b - a), -np.inf * np.ones_like(z))
    elif name == "Laplace":
        s = sigma / np.sqrt(2)
        return -np.abs(z - mu) / s - np.log(2 * s)
    elif name == "Cauchy":
        return -(np.log(np.pi) + np.log(sigma)
                 + np.log1p(((z - mu) / sigma) ** 2))
    elif name == "Bimodal":
        c = -np.log(sigma) - 0.5 * np.log(2 * np.pi) + np.log(0.5)
        lp1 = c - 0.5 * ((z + mu) / sigma) ** 2
        lp2 = c - 0.5 * ((z - mu) / sigma) ** 2
        return np.logaddexp(lp1, lp2)
    return -np.inf * np.ones_like(z)


def gaussian_kde(data):
    """Gaussian KDE with Scott's bandwidth rule; returns a callable."""
    bw = len(data) ** (-1 / 5) * data.std(ddof=1)
    def _eval(xq):
        diff = (xq[:, None] - data[None, :]) / bw
        return np.mean(np.exp(-0.5 * diff ** 2), axis=1) / (bw * np.sqrt(2 * np.pi))
    return _eval


# ── App ───────────────────────────────────────────────────────────────────────

class NormFlowApp:

    # Slider specifications: (label, key, from_, to, default, resolution)
    TRANSFORM_SLIDERS = [
        ("θ₀", "θ0",  -3.0,  3.0,  0.0,  0.05),
        ("θ₁", "θ1",  -3.0,  3.0,  1.0,  0.05),
        ("θ₂", "θ2",  -2.0,  2.0,  0.0,  0.05),
        ("θ₃", "θ3",  -0.5,  0.5,  0.0,  0.01),
    ]
    GAUSSIAN_SLIDERS = [
        ("μ",  "mu",  -2.0,  2.0,  0.0,  0.05),
        ("σ",  "sg",   0.1,  2.0,  1.0,  0.05),
    ]

    CZ       = "#1565C0"
    CT_POS   = "#C62828"
    CT_NEG   = "#E65100"
    CX       = "#2E7D32"
    CT_TARGET= "#FF8F00"
    CL       = "#263238"   # dark slate  — total / NLL  (the sum)
    CM       = "#757575"
    CL_ENER  = "#1E88E5"   # blue        — energy / latent term
    CL_ENTR  = "#FB8C00"   # orange      — entropy / Jacobian term
    FA       = 0.15

    def __init__(self):
        # ── Plain-Python state (replacing tk.*Var) ────────────────────────
        self.vals: dict[str, float] = {}   # all slider values
        self._sliders: dict        = {}    # key → ui.slider widget
        self._val_labels: dict     = {}    # key → ui.label widget
        self._scale_widgets        = self._sliders  # alias kept for compat

        self.dist_val        = "Gaussian"
        self.transform_val   = "Single layer perceptron"
        self.K_val           = 3
        self.K_rqs_val       = 3

        self._rescale_axes_val   = True
        self._show_iw_val        = False
        self._show_map_lines_val = False
        self._n_map_pts_val      = "10"
        self._show_data_val      = True
        self._show_target_val    = False
        self._show_exact_val     = False

        self._train_mode_val  = "Energy-based"
        self._optimizer_val   = "Adam"
        self._n_entry_val     = "1000"
        self._n_epochs_val    = "500"
        self._lr_val          = "0.01"
        self._n_batch_val     = "1000"
        self._stride_val      = "10"
        self._delay_val       = "20"
        self._resample_val    = True

        # ── Training / display state (unchanged from tkinter version) ─────
        self._samples_z: np.ndarray | None = None
        self._data_x:    np.ndarray | None = None
        self._loss_history:         list[float] = []
        self._loss_energy_history:  list[float] = []
        self._loss_entropy_history: list[float] = []
        self._training: bool = False
        self._tgt_cache_key = None
        self._tgt_cache_val = None
        self._train_params_pending  = None
        self._training_epoch: int   = 0
        self._n_epochs_total: int   = 1
        self._frozen_static: dict | None = None
        self._train_params_live:    np.ndarray | None = None
        self._train_params_target:  np.ndarray | None = None
        self._last_trained_params:    np.ndarray | None = None
        self._slider_snapshot_at_end: np.ndarray | None = None
        self._use_trained_params:     bool = False
        self._train_starting_params:  np.ndarray | None = None
        self._train_z_batch: np.ndarray | None = None
        self._axis_lock: dict | None = None
        self._suppress_redraw: bool = False

        # NiceGUI-specific
        self._render_dirty:       bool  = False
        self._rendering:          bool  = False   # single-flight render guard
        self._last_render_t:      float = 0.0     # for throttling during training
        self._training_was_active: bool = False
        self._train_status: tuple | None = None   # (text, color) from thread
        self._epoch_time_us: float = 0.0
        self._dark: bool = False                   # dark-mode flag
        self._prog_kind: str = "neutral"           # neutral | done | error

        self._build_ui()
        self._on_transform_change()
        self._redraw()
        ui.timer(0.03, self._tick)

    # ── UI construction ───────────────────────────────────────────────────────

    # ── Header-picture chooser ────────────────────────────────────────────────

    def _pic_dot_style(self, active):
        """Style for one selector circle (shown beside the header image on the
        grey panel); filled blue when current, hollow grey otherwise."""
        return ("width:13px; height:13px; border-radius:50%; cursor:pointer; "
                + ("background:#1565C0; border:2px solid #1565C0"
                   if active else "background:transparent; border:2px solid #999"))

    def _choose_pic(self, i):
        self._header_img.set_source(f"/static/{self._pic_files[i]}")
        for j, dot in enumerate(self._pic_dots):
            dot.style(replace=self._pic_dot_style(j == i))

    def _default_pic_index(self):
        """Default header picture per mode: the plain Boltzmann portrait
        (Boltzmann.png, index 2) in dark mode, the terminator-glasses portrait
        (Boltzmannator_title.png, index 0) in light mode."""
        return 2 if self._dark else 0

    # ── Dark / light mode ─────────────────────────────────────────────────────

    def _prog_color(self, kind=None):
        """Theme-aware colour for the progress text by semantic kind:
        neutral (live), done (success), error.  Brighter variants in dark."""
        kind = kind or self._prog_kind
        if kind == "done":
            return "#66BB6A" if self._dark else "#1B5E20"
        if kind == "error":
            return "#EF5350" if self._dark else "#B71C1C"
        return "#e6e6e6" if self._dark else "#222222"      # neutral

    def _set_prog_kind(self, kind):
        """Remember what the progress text represents and (re)apply its colour
        for the current mode."""
        self._prog_kind = kind
        if hasattr(self, "_prog_label"):
            self._prog_label.style(f"color:{self._prog_color()} !important")

    def _toggle_dark(self, value):
        self._dark = bool(value)
        if self._dark:
            self._dark_mode.enable()
        else:
            self._dark_mode.disable()
        # re-apply the progress text colour for the new mode, preserving its
        # meaning (a finished "Done" message stays green, etc.)
        self._set_prog_kind(self._prog_kind)
        # switch the header picture to this mode's default
        if hasattr(self, "_header_img"):
            self._choose_pic(self._default_pic_index())
        self._request_render()      # repaint the figure with the new theme

    def _apply_fig_theme(self):
        """Recolour the matplotlib figure for the current mode.  Called at the
        end of every redraw.  Colours are set explicitly for BOTH modes (not
        just dark) so that switching dark->light restores the original
        light-mode appearance rather than leaving the dark colours behind."""
        dark = self._dark
        bg      = "#23232a" if dark else "white"
        fg      = "#e6e6e6" if dark else "black"      # text / ticks / spines
        grid_c  = "#5a5a66" if dark else "#b0b0b0"    # matplotlib default grey
        main_axes = [self.ax_top, self.ax_main, self.ax_logj, self.ax_right,
                     self.ax_hist_z, self.ax_hist_x, self.ax_loss]
        self.fig.patch.set_facecolor(bg)
        self.ax_hist_x_twin.set_facecolor("none")   # keep twin transparent
        for ax in main_axes:
            ax.set_facecolor(bg)
            ax.title.set_color(fg)
            ax.xaxis.label.set_color(fg)
            ax.yaxis.label.set_color(fg)
            ax.tick_params(axis="both", colors=fg)
            for s in ax.spines.values():
                s.set_color(fg)
            for gl in ax.get_xgridlines() + ax.get_ygridlines():
                gl.set_color(grid_c)
            leg = ax.get_legend()
            if leg is not None:
                for t in leg.get_texts():
                    t.set_color(fg)
        # The x-histogram's twin axis overlays it; recolour its shared spines
        # and x-ticks too (otherwise its default-black axis looks darker than
        # the other panels).  Leave the right spine — that's the brown w(x)
        # weights axis, styled separately when shown.
        tw = self.ax_hist_x_twin
        for name in ("bottom", "top", "left"):
            tw.spines[name].set_color(fg)
        tw.tick_params(axis="x", colors=fg)

    def _build_ui(self):
        # Per-page CSS — each browser gets its own page, so this is added per
        # client.  Static files are registered once, globally, in the entry
        # point (registering a route per page would error).
        ui.add_css("""
        .ctrl .q-tab-panels, .ctrl .q-tab-panel { padding: 2px 4px !important; }
        .ctrl .nicegui-column { gap: 1px !important; }
        .ctrl .nicegui-row    { gap: 4px !important; }
        .ctrl .q-separator--horizontal { margin: 1px 0 !important; }
        .ctrl .q-field--dense .q-field__control,
        .ctrl .q-field--dense .q-field__marginal { min-height: 26px !important;
                                                    height: 26px !important; }
        /* dropdowns: a touch taller + lift the text off the underline */
        .ctrl .q-select.q-field--dense .q-field__control,
        .ctrl .q-select.q-field--dense .q-field__marginal {
            min-height: 30px !important; height: 30px !important; }
        .ctrl .q-select.q-field--dense .q-field__native { padding-bottom: 3px !important; }
        /* uniform text size for input numbers and dropdown text */
        .ctrl .q-field__native, .ctrl .q-field__native input,
        .ctrl .q-field__input { font-size: 13px !important; }
        .ctrl .q-checkbox { min-height: 0 !important; }
        .ctrl .q-checkbox__inner { font-size: 18px !important; }
        .ctrl .q-checkbox__label { font-size: 13px !important; line-height: 1.1 !important; }
        .ctrl .q-radio { min-height: 0 !important; }
        .ctrl .q-radio__inner    { font-size: 18px !important; }
        .ctrl .q-radio__label    { font-size: 13px !important; line-height: 1.1 !important; }
        .ctrl .q-checkbox__bg, .ctrl .q-radio__bg { } /* keep box visuals */
        .ctrl .q-checkbox > .q-checkbox__inner,
        .ctrl .q-radio > .q-radio__inner { padding: 0 !important; }
        /* Sliders: give the track SYMMETRIC vertical padding so it stays
           centred in its row — then the left label and right value (which are
           vertically centred) line up with the track.  Small padding = tight
           rows for Densities/Training; the Map tab overrides with more. */
        .ctrl .q-slider { min-height: 0 !important; }
        .ctrl .q-slider--h { min-height: 0 !important; }
        .ctrl .q-slider--h .q-slider__track-container { padding: 2px 0 !important; }
        .ctrl .sldrow { min-height: 0 !important; margin: 0 !important; }
        .ctrl .maptab .sldrow { margin: 3px 0 !important; }
        .ctrl .maptab .q-slider--h .q-slider__track-container {
            padding: 7px 0 !important; }
        /* Training tab: stack rows with modest spacing */
        .ctrl .traintab { display: flex !important; flex-direction: column !important;
                          gap: 5px !important; }
        .ctrl .traintab .q-btn { min-height: 0 !important;
                          padding-top: 4px !important; padding-bottom: 4px !important; }
        /* segmented / pill tabs */
        .ctrl .q-tabs { background: #e2e2ea !important; border-radius: 9px !important;
                        padding: 3px !important; }
        .ctrl .q-tabs__content { width: 100% !important; }
        .ctrl .q-tab { flex: 1 1 0 !important; min-height: 34px !important;
                       padding: 0 6px !important; border-radius: 7px !important;
                       text-transform: none !important; font-size: 12.5px !important;
                       color: #555 !important; opacity: 1 !important;
                       transition: background .15s, color .15s; }
        .ctrl .q-tab--active { background: #1565C0 !important; color: #fff !important;
                               font-weight: 600 !important;
                               box-shadow: 0 1px 3px rgba(0,0,0,.25) !important; }
        .ctrl .q-tab__indicator { display: none !important; }
        .ctrl .q-tab .q-icon { font-size: 18px !important; }
        .ctrl .q-linear-progress { margin: 2px 0 !important; }

        /* ── Dark mode (Quasar adds body.body--dark) ── */
        .body--dark .ctrl { background: #26262b !important; }
        .body--dark .ctrl .q-tabs { background: #3a3a42 !important; }
        /* make every text element in the panel legible on the dark bg;
           progress-status colours are set inline with !important so they win */
        .body--dark .ctrl * { color: #e2e2e2 !important; }
        /* keep the progress-bar fill green (the rule above would grey it) */
        .body--dark .ctrl .q-linear-progress__model {
            background: #21BA45 !important; color: #21BA45 !important; }
        /* in dark mode, fill the plot panel's letterbox margin with the same
           grey as the figure background so there is no black border */
        .body--dark .plotpanel { background: #23232a !important; }
        /* keep the footer muted in dark mode (override the blanket text rule) */
        .body--dark .ctrl .appfooter { color: #888 !important; }
        """)
        # Dark-mode controller (starts in light mode).
        self._dark_mode = ui.dark_mode()

        # Remove NiceGUI's default page padding/gap so the 100vh layout starts
        # flush at the top (otherwise the top margin pushes the bottom controls
        # off-screen).
        ui.query(".nicegui-content").classes("p-0 gap-0")

        with ui.row().classes("w-full no-wrap").style(
                "height:100vh; gap:0; overflow:hidden"):

            # ── Left control panel ────────────────────────────────────────
            # CSS grid: image / tabs / (flexible tab-scroll) / bottom controls.
            # The 3rd row is minmax(0,1fr) so ONLY the tab area shrinks, which
            # keeps the header and the Reset/Exit buttons always fully visible.
            with ui.column().classes("ctrl no-wrap").style(
                    "display:grid; grid-template-rows:auto auto minmax(0,1fr) auto; "
                    "box-sizing:border-box; width:340px; min-width:340px; "
                    "height:100vh; gap:4px; background:#f0f0f0; padding:4px; "
                    "overflow:hidden"):

                # Header image with three small circles (overlaid in the
                # corner, so they cost no extra vertical space) to choose the
                # picture; the filled circle marks the current choice.
                self._pic_files = ["Boltzmannator_title.png",
                                   "Boltzmannator_movie.png",
                                   "Boltzmann.png"]
                self._pic_dots = []
                try:
                    with ui.row().classes("no-wrap").style(
                            "width:100%; flex:0 0 auto; justify-content:center; "
                            "align-items:center; gap:8px; margin:0 auto 4px"):
                        _def = self._default_pic_index()
                        self._header_img = ui.image(
                            f"/static/{self._pic_files[_def]}").style(
                            "width:230px; display:block")
                        with ui.column().style("gap:7px"):
                            for i in range(len(self._pic_files)):
                                dot = ui.element("div").classes("cursor-pointer")
                                dot.style(self._pic_dot_style(i == _def))
                                dot.on("click", lambda e, i=i: self._choose_pic(i))
                                self._pic_dots.append(dot)
                except Exception:
                    pass

                # Tabs — segmented/pill look with Material icons.
                with ui.tabs().classes("w-full").props(
                        "dense inline-label").style("flex:0 0 auto") as tabs:
                    t_dist  = ui.tab("dist",  label="Densities",
                                     icon="area_chart")
                    # Map: inline SVG of a smooth monotonically increasing
                    # curve with axes (inherits the tab text colour).
                    _graph_svg = (
                        '<svg width="18" height="18" viewBox="0 0 24 24" '
                        'fill="none" style="margin-right:6px;'
                        'vertical-align:middle">'
                        '<path d="M4 3 V20 H21" stroke="currentColor" '
                        'stroke-width="1.6" stroke-linecap="round" '
                        'stroke-linejoin="round" opacity="0.55"/>'
                        '<path d="M4 19 C 10 19, 13 12, 20 4" '
                        'stroke="currentColor" stroke-width="2.4" '
                        'stroke-linecap="round"/></svg>')
                    t_map   = ui.tab("map",   label="")
                    with t_map:
                        ui.html(_graph_svg + "<span>Map</span>").style(
                            "display:inline-flex;align-items:center")
                    # Training: ∇ glyph (no Material nabla icon), sized to
                    # match the Material icons on the other tabs.
                    t_train = ui.tab("train", label="")
                    with t_train:
                        ui.html('<span style="font-size:19px;font-weight:600;'
                                'margin-right:6px;line-height:1">∇</span>'
                                '<span>Training</span>').style(
                            "display:inline-flex;align-items:center")

                with ui.scroll_area().classes("w-full").style(
                        "min-height:0; height:100%"):
                    with ui.tab_panels(tabs, value=t_dist).classes("w-full"):

                        # ── Densities tab ─────────────────────────────────────
                        with ui.tab_panel(t_dist):
                            ui.html("Latent p<sub>z</sub>(z)").style(
                                "font-weight:bold; font-size:13px; margin-top:4px")
                            self._dist_select = ui.select(
                                ["Gaussian", "Uniform", "Laplace", "Bimodal"],
                                value="Gaussian",
                                on_change=lambda e: self._on_dist_change(e.value),
                            ).classes("w-full").props("dense")

                            for spec in self.GAUSSIAN_SLIDERS:
                                self._add_slider(*spec)

                            ui.separator()
                            ui.html("Target p<sup>*</sup>(x)").style(
                                "font-weight:bold; font-size:13px")
                            ui.html(
                                "p<sup>*</sup>(x) ∝ exp(−U(x)/kT),  "
                                "U = u₁x+u₂x²+u₃x³+u₄x⁴"
                            ).style("font-size:13px; color:#555; font-style:italic")

                            self._show_target_cb = ui.checkbox(
                                "Show target", value=False,
                                on_change=lambda e: self._cb_change(
                                    "_show_target_val", e.value, rescale=True))
                            self._show_exact_cb = ui.checkbox(
                                "Show exact transformation", value=False,
                                on_change=lambda e: self._cb_change(
                                    "_show_exact_val", e.value, rescale=True))

                            self._add_slider("kT", "kT",  0.1, 3.0, 1.0, 0.05)
                            self._add_slider("u₁", "u1", -2.0, 2.0, 0.0, 0.10)
                            self._add_slider("u₂", "u2", -2.0, 2.0, 1.0, 0.10)
                            self._add_slider("u₃", "u3", -1.0, 1.0, 0.0, 0.05)
                            self._add_slider("u₄", "u4",  0.05, 1.5, 0.1, 0.05)

                            ui.separator()
                            with ui.row().classes("items-center gap-2 w-full"):
                                self._show_map_cb = ui.checkbox(
                                    "Show mapping lines", value=False,
                                    on_change=lambda e: self._cb_change(
                                        "_show_map_lines_val", e.value))
                                ui.label("N =").style("font-size:13px")
                                self._n_map_input = (
                                    ui.input(value="10")
                                    .style("width:50px")
                                    .props("dense")
                                )
                                self._n_map_input.on(
                                    "change", lambda _: self._request_render())

                        # ── Map tab ───────────────────────────────────────────
                        with ui.tab_panel(t_map).classes("maptab"):
                            ui.html("Transformation f<sub>θ</sub>(z)").style(
                                "font-weight:bold; font-size:13px; margin-top:4px")
                            self._transform_select = ui.select(
                                ["Polynomial",
                                 "Single layer perceptron",
                                 "Rational-quadratic spline"],
                                value="Single layer perceptron",
                                on_change=lambda e: self._on_transform_change(e.value),
                            ).classes("w-full").props("dense")

                            # Polynomial
                            with ui.column().classes("w-full gap-0") as self._poly_section:
                                ui.label("x = θ₀+θ₁z+θ₂z²+θ₃z³").style(
                                    "font-size:13px; color:#555; font-style:italic")
                                for spec in self.TRANSFORM_SLIDERS:
                                    self._add_slider(*spec)

                            # Single layer perceptron
                            with ui.column().classes("w-full gap-0") as self._sig_section:
                                ui.label("x = a + bz + Σwₖσ((z−cₖ)/sₖ)").style(
                                    "font-size:13px; color:#555; font-style:italic")
                                with ui.row().classes("items-center gap-1"):
                                    ui.label("K =").style("font-size:13px")
                                    self._k_radio = ui.radio(
                                        {1: "1", 2: "2", 3: "3", 4: "4",
                                         5: "5", 6: "6", 7: "7", 8: "8"},
                                        value=3,
                                        on_change=lambda e: self._on_K_change(int(e.value)),
                                    ).props("inline dense")
                                K_MAX = 8
                                defaults_c = [-2.5, -1.79, -1.07, -0.36,
                                              0.36, 1.07, 1.79, 2.5]
                                _grp = ("font-size:13px; font-style:italic; "
                                        "color:#666; margin-top:9px")

                                # Group: a, b (linear part)
                                with ui.column().classes("w-full gap-0"):
                                    self._add_slider("a", "sig_off",   -3.0, 3.0, 0.0, 0.05)
                                    self._add_slider("b", "sig_slope", -3.0, 3.0, 1.0, 0.05)

                                # Group: weights wₖ
                                ui.label("weights  wₖ").style(_grp)
                                self._sig_w_rows = []
                                for k in range(K_MAX):
                                    with ui.column().classes("w-full gap-0") as r:
                                        self._add_slider(
                                            f"w{k+1}", f"w{k}", 0.01, 3.0, 0.01, 0.05)
                                    self._sig_w_rows.append(r)

                                # Group: centres cₖ
                                ui.label("centres  cₖ").style(_grp)
                                self._sig_c_rows = []
                                for k in range(K_MAX):
                                    with ui.column().classes("w-full gap-0") as r:
                                        self._add_slider(
                                            f"c{k+1}", f"c{k}", -4.0, 4.0,
                                            defaults_c[k], 0.1)
                                    self._sig_c_rows.append(r)

                                # Group: scales sₖ
                                ui.label("scales  sₖ").style(_grp)
                                self._sig_s_rows = []
                                for k in range(K_MAX):
                                    with ui.column().classes("w-full gap-0") as r:
                                        self._add_slider(
                                            f"s{k+1}", f"s{k}", 0.05, 3.0, 1.0, 0.05)
                                    self._sig_s_rows.append(r)

                            # Rational-quadratic spline
                            with ui.column().classes("w-full gap-0") as self._rqs_section:
                                ui.label(
                                    "Monotone spline on [−B, B], linear outside"
                                ).style("font-size:13px; color:#555; font-style:italic")
                                with ui.row().classes("items-center gap-1"):
                                    ui.label("K =").style("font-size:13px")
                                    self._k_rqs_radio = ui.radio(
                                        {2: "2", 3: "3", 4: "4"},
                                        value=3,
                                        on_change=lambda e: self._on_K_rqs_change(
                                            int(e.value)),
                                    ).props("inline dense")
                                self._add_slider("B", "rqs_B", 1.0, 6.0, 3.0, 0.1)
                                K_MAX_RQS = 4
                                ui.label("Bin widths").style(
                                    "font-size:13px; font-style:italic; color:#666")
                                self._rqs_w_rows = []
                                for k in range(K_MAX_RQS):
                                    with ui.column().classes("w-full gap-0") as r:
                                        self._add_slider(
                                            f"w{k+1}", f"rqs_w{k}", -3.0, 3.0, 0.0, 0.05)
                                    self._rqs_w_rows.append(r)
                                ui.label("Bin heights").style(
                                    "font-size:13px; font-style:italic; color:#666")
                                self._rqs_h_rows = []
                                for k in range(K_MAX_RQS):
                                    with ui.column().classes("w-full gap-0") as r:
                                        self._add_slider(
                                            f"h{k+1}", f"rqs_h{k}", -3.0, 3.0, 0.0, 0.05)
                                    self._rqs_h_rows.append(r)
                                ui.label("Knot derivatives").style(
                                    "font-size:13px; font-style:italic; color:#666")
                                self._rqs_d_rows = []
                                for k in range(K_MAX_RQS + 1):
                                    with ui.column().classes("w-full gap-0") as r:
                                        self._add_slider(
                                            f"d{k}", f"rqs_d{k}", -3.0, 3.0, 0.0, 0.05)
                                    self._rqs_d_rows.append(r)

                            ui.separator()
                            with ui.row().classes("gap-2 w-full no-wrap"):
                                self._btn("Defaults", "settings_backup_restore",
                                          "blue-grey-6", self._reset_map_params,
                                          tooltip="Reset the transformation "
                                                  "parameters to their defaults")
                                self._btn("Randomize", "shuffle",
                                          "deep-purple-5", self._randomize_map_params,
                                          tooltip="Set random transformation "
                                                  "parameters")

                        # ── Training tab ──────────────────────────────────────
                        with ui.tab_panel(t_train).classes("traintab"):
                            ui.label("Sampling").style(
                                "font-weight:bold; font-size:13px; margin-top:4px")
                            with ui.row().classes("items-center gap-2"):
                                ui.label("N =").style("font-size:13px")
                                self._n_entry_input = (
                                    ui.input(value="1000")
                                    .style("width:80px").props("dense"))
                            with ui.row().classes("gap-2 w-full no-wrap"):
                                self._btn("Sample!", "casino",
                                          "primary", self._do_sampling,
                                          tooltip="Draw N samples from the latent "
                                                  "distribution and push them "
                                                  "through the transformation")
                                self._btn("Data", "scatter_plot",
                                          "teal-7", self._do_generate_data,
                                          tooltip="Generate N example data points "
                                                  "from the target distribution")
                            self._show_data_cb = ui.checkbox(
                                "Show target data", value=True,
                                on_change=lambda e: self._cb_change(
                                    "_show_data_val", e.value))
                            self._show_iw_cb = ui.checkbox(
                                "Show importance weights", value=False,
                                on_change=lambda e: self._cb_change(
                                    "_show_iw_val", e.value))

                            ui.separator()
                            ui.label("Training").style(
                                "font-weight:bold; font-size:13px")
                            self._mode_radio = ui.radio(
                                {"Energy-based": "Energy-based",
                                 "Example-based": "Example-based"},
                                value="Energy-based",
                                on_change=lambda e: setattr(
                                    self, "_train_mode_val", e.value),
                            ).props("inline dense")
                            self._opt_select = ui.select(
                                ["Adam", "SGD", "SGD+momentum", "RMSprop"],
                                value="Adam",
                                on_change=lambda e: setattr(
                                    self, "_optimizer_val", e.value),
                            ).classes("w-full").props("dense")

                            with ui.row().classes("items-center gap-2"):
                                ui.label("Epochs =").style("font-size:13px")
                                self._n_epochs_input = (
                                    ui.input(value="500")
                                    .style("width:70px").props("dense"))
                                ui.label("lr =").style("font-size:13px")
                                self._lr_input = (
                                    ui.input(value="0.01")
                                    .style("width:70px").props("dense"))
                            with ui.row().classes("items-center gap-2"):
                                ui.label("N batch =").style("font-size:13px")
                                self._n_batch_input = (
                                    ui.input(value="1000")
                                    .style("width:70px").props("dense"))
                                self._resample_cb = ui.checkbox(
                                    "resample", value=True,
                                    on_change=lambda e: setattr(
                                        self, "_resample_val", e.value))
                            with ui.row().classes("items-center gap-2 w-full no-wrap"):
                                ui.label("Every =").style("font-size:13px")
                                self._stride_input = (
                                    ui.input(value="10")
                                    .style("width:50px").props("dense"))
                                ui.label("epochs").style("font-size:13px")
                                ui.space()
                                ui.label("Delay =").style("font-size:13px")
                                self._delay_input = (
                                    ui.input(value="20")
                                    .style("width:55px").props("dense"))
                                ui.label("ms").style("font-size:13px")

                            with ui.row().classes("gap-2 w-full no-wrap"):
                                self._btn("Train!", "rocket_launch",
                                          "positive", self._do_training,
                                          tooltip="Optimise the transformation "
                                                  "parameters")
                                self._btn("Stop", "stop",
                                          "negative", self._stop_training,
                                          tooltip="Stop the running training")

                            self._prog_bar = (
                                ui.linear_progress(value=0, show_value=False,
                                                   size="20px")
                                .classes("w-full")
                                .props("instant-feedback color=positive rounded"))
                            with self._prog_bar:
                                self._prog_epoch = ui.label("0 / 0").classes(
                                    "absolute-center").style(
                                    "font-size:13px; font-weight:600; color:#fff; "
                                    "text-shadow:0 0 2px rgba(0,0,0,.55)")
                            self._prog_label = ui.label("").style(
                                "font-size:13px; color:#222; "
                                "text-align:center; white-space:pre")

                            self._btn("Reset training", "refresh",
                                      "orange-8", self._reset_training, full=True,
                                      tooltip="Clear loss history and reset the "
                                              "transformation to the identity")

                # ── Always-visible controls (never shrink: flex 0 0 auto) ──
                with ui.column().classes("w-full no-wrap").style(
                        "flex:0 0 auto; gap:4px"):
                    ui.separator()
                    with ui.row().classes("items-center gap-2 w-full no-wrap"):
                        self._rescale_cb = ui.checkbox(
                            "Auto-rescale x / z axes", value=True,
                            on_change=lambda e: self._cb_change(
                                "_rescale_axes_val", e.value))
                        ui.space()
                        self._dark_switch = ui.switch(
                            "Dark", value=False,
                            on_change=lambda e: self._toggle_dark(e.value)
                        ).props("dense").tooltip("Toggle dark / light mode")
                    ui.separator()
                    with ui.row().classes("gap-2 w-full no-wrap"):
                        self._btn("Reset", "restart_alt",
                                  "blue-grey-7", self._reset,
                                  tooltip="Reset everything to defaults")
                        self._btn("Exit", "logout",
                                  "grey-9", self._exit_app,
                                  tooltip="Close this session")
                    ui.label("© 2026 Christoph Dellago").classes(
                        "appfooter").style(
                        "width:100%; text-align:center; font-size:10px; "
                        "color:#888; margin-top:2px")

            # ── Right matplotlib panel ────────────────────────────────────
            with ui.column().classes("flex-grow plotpanel").style(
                    "height:100vh; padding:2px 2px 2px 20px; overflow:hidden"):
                self._plot = (
                    ui.matplotlib(figsize=(12.5, 9.2))
                    .classes("w-full h-full")
                    .style("flex:1; min-height:0"))

        # ── Figure layout (same GridSpec as tkinter version) ──────────────
        self.fig = self._plot.figure
        self.fig.patch.set_facecolor("white")

        # Tight outer margins so the axes fill the canvas (matplotlib's
        # defaults leave ~12% white on the left and ~10% elsewhere).
        outer    = self.fig.add_gridspec(
            1, 2, width_ratios=[3.5, 2], wspace=0.22,
            left=0.055, right=0.975, top=0.955, bottom=0.065)
        gs_left  = outer[0].subgridspec(
            3, 2, height_ratios=[1, 3, 1],
            width_ratios=[3, 1], hspace=0.04, wspace=0.04)
        gs_right = outer[1].subgridspec(3, 1, hspace=0.55)

        self.ax_top    = self.fig.add_subplot(gs_left[0, 0])
        self.ax_main   = self.fig.add_subplot(gs_left[1, 0], sharex=self.ax_top)
        self.ax_logj   = self.fig.add_subplot(gs_left[2, 0], sharex=self.ax_top)
        self.ax_right  = self.fig.add_subplot(gs_left[1, 1], sharey=self.ax_main)
        self.ax_hist_z = self.fig.add_subplot(gs_right[0])
        self.ax_hist_x = self.fig.add_subplot(gs_right[1])
        self.ax_hist_x_twin = self.ax_hist_x.twinx()
        self.ax_loss   = self.fig.add_subplot(gs_right[2])

    def _exit_app(self):
        """Native mode: quit the process.  Server mode: just close this
        client's own view (never kills the server for other users)."""
        if NATIVE:
            os.kill(os.getpid(), 9)
        else:
            ui.navigate.to("about:blank")

    # ── Button helper ─────────────────────────────────────────────────────────

    def _btn(self, label, icon, color, on_click, full=False, tooltip=None):
        """Consistent solid button: Material icon, mixed-case label, white text
        on a Quasar-named colour (auto-contrasted), subtle rounding."""
        b = ui.button(label, icon=icon, on_click=on_click, color=color)
        b.props("no-caps unelevated dense")
        b.classes("w-full" if full else "grow")
        b.style("flex:1; border-radius:7px; font-weight:500; "
                "letter-spacing:.2px; padding:5px 10px")
        if tooltip:
            b.tooltip(tooltip)
        return b

    # ── Slider helpers ────────────────────────────────────────────────────────

    def _add_slider(self, label, key, from_, to, default, resolution):
        self.vals[key] = default
        with ui.row().classes("w-full items-center no-wrap sldrow").style(
                "gap:6px; padding:0; margin:0; min-height:0; line-height:1"):
            ui.label(label).style(
                "min-width:20px; font-size:13px; flex:0 0 auto")
            sl = ui.slider(
                min=from_, max=to, value=default, step=resolution,
                on_change=lambda e, k=key: self._on_slider_change(k, float(e.value)),
            ).classes("flex-grow").props("dense label=false thumb-size=14px")
            lbl = ui.label(f"{default:+.2f}").style(
                "min-width:44px; flex:0 0 auto; font-size:13px; "
                "text-align:right; font-weight:bold; color:#333")
        self._sliders[key] = sl
        self._val_labels[key] = lbl
        return sl

    def _set_val(self, key, value):
        """Set a parameter value and sync the slider widget + value label."""
        value = float(np.clip(value, -1e6, 1e6))
        self.vals[key] = value
        if key in self._sliders:
            s  = self._sliders[key]
            lo = float(s._props.get("min", -1e6))
            hi = float(s._props.get("max",  1e6))
            s.value = float(np.clip(value, lo, hi))
        if key in self._val_labels:
            self._val_labels[key].text = f"{value:+.2f}"

    def _on_slider_change(self, key, value):
        self.vals[key] = value
        if key in self._val_labels:
            self._val_labels[key].text = f"{value:+.2f}"
        if self._suppress_redraw:
            return
        self._use_trained_params = False
        self._axis_lock = None
        self._render_dirty = True

    def _refresh_value_labels(self):
        for key, lbl in self._val_labels.items():
            if key in self.vals:
                lbl.text = f"{self.vals[key]:+.2f}"

    def _get_params(self):
        return np.array([self.vals[k] for k in self._get_param_keys()])

    def _set_params(self, params):
        for k, v in zip(self._get_param_keys(), params):
            self._set_val(k, float(np.clip(v, -1e6, 1e6)))

    # ── Small UI callbacks ────────────────────────────────────────────────────

    def _cb_change(self, attr, value, rescale=False):
        """Generic checkbox handler."""
        setattr(self, attr, value)
        if rescale:
            self._axis_lock = None
        self._request_render()

    # ── Distribution / transform configuration ────────────────────────────────

    def _configure_latent_sliders_for_dist(self, dist):
        mu_sl = self._sliders.get("mu")
        self._suppress_redraw = True
        try:
            if dist == "Bimodal":
                if mu_sl is not None:
                    mu_sl._props["min"] = 0.0
                    mu_sl._props["max"] = 2.0
                self._set_val("mu", 1.0)
                self._set_val("sg", 0.5)
            else:
                if mu_sl is not None:
                    mu_sl._props["min"] = -2.0
                    mu_sl._props["max"] =  2.0
        finally:
            self._suppress_redraw = False
        self._refresh_value_labels()

    def _on_dist_change(self, val=None):
        if val is not None:
            self.dist_val = val
        self._configure_latent_sliders_for_dist(self.dist_val)
        self._axis_lock = None
        self._request_render()

    def _on_transform_change(self, val=None):
        if val is not None:
            self.transform_val = val
        if not hasattr(self, "_poly_section"):
            return
        self._poly_section.visible = (self.transform_val == "Polynomial")
        self._sig_section.visible  = (self.transform_val == "Single layer perceptron")
        self._rqs_section.visible  = (self.transform_val == "Rational-quadratic spline")
        if self.transform_val == "Rational-quadratic spline":
            self._update_rqs_rows()
        else:
            self._update_sig_rows()
        self._use_trained_params = False
        self._axis_lock = None
        self._request_render()

    def _on_K_change(self, k=None):
        if k is not None:
            self.K_val = int(k)
        self._update_sig_rows()
        self._use_trained_params = False
        self._axis_lock = None
        self._request_render()

    def _on_K_rqs_change(self, k=None):
        if k is not None:
            self.K_rqs_val = int(k)
        self._update_rqs_rows()
        self._use_trained_params = False
        self._axis_lock = None
        self._request_render()

    def _update_sig_rows(self):
        K = self.K_val
        for rows in (self._sig_w_rows, self._sig_c_rows, self._sig_s_rows):
            for i, row in enumerate(rows):
                row.visible = (i < K)

    def _update_rqs_rows(self):
        K = self.K_rqs_val
        for i, row in enumerate(self._rqs_w_rows):
            row.visible = (i < K)
        for i, row in enumerate(self._rqs_h_rows):
            row.visible = (i < K)
        for i, row in enumerate(self._rqs_d_rows):
            row.visible = (i <= K)

    # ── Reset methods ─────────────────────────────────────────────────────────

    def _reset(self):
        self.dist_val = "Gaussian"
        if hasattr(self, "_dist_select"):
            self._dist_select.value = "Gaussian"
        self._configure_latent_sliders_for_dist("Gaussian")
        for _, key, _, _, default, _ in self.GAUSSIAN_SLIDERS:
            self._set_val(key, default)
        for _, key, _, _, default, _ in self.TRANSFORM_SLIDERS:
            self._set_val(key, default)
        self.transform_val = "Single layer perceptron"
        if hasattr(self, "_transform_select"):
            self._transform_select.value = "Single layer perceptron"
        self._on_transform_change()
        defaults_c = [-2.5, -1.79, -1.07, -0.36, 0.36, 1.07, 1.79, 2.5]
        for k in range(8):
            self._set_val(f"w{k}", 0.01)
            self._set_val(f"c{k}", defaults_c[k])
            self._set_val(f"s{k}", 1.0)
        self._set_val("sig_off",   0.0)
        self._set_val("sig_slope", 1.0)
        self.K_val = 3
        if hasattr(self, "_k_radio"):
            self._k_radio.value = 3
        self._update_sig_rows()
        for key, val in [("kT", 1.0), ("u1", 0.0), ("u2", 1.0),
                          ("u3", 0.0), ("u4", 0.1)]:
            self._set_val(key, val)
        self._show_target_val = False
        self._show_exact_val  = False
        self._tgt_cache_key   = None
        self._axis_lock       = None
        if hasattr(self, "_show_target_cb"): self._show_target_cb.value = False
        if hasattr(self, "_show_exact_cb"):  self._show_exact_cb.value  = False
        if hasattr(self, "_show_data_cb"):   self._show_data_cb.value   = True
        if hasattr(self, "_show_iw_cb"):     self._show_iw_cb.value     = False
        if hasattr(self, "_show_map_cb"):    self._show_map_cb.value    = False
        if hasattr(self, "_rescale_cb"):     self._rescale_cb.value     = True
        if hasattr(self, "_n_entry_input"):  self._n_entry_input.value  = "1000"
        if hasattr(self, "_n_map_input"):    self._n_map_input.value    = "10"
        self._show_data_val      = True
        self._show_iw_val        = False
        self._show_map_lines_val = False
        self._n_map_pts_val      = "10"
        self._rescale_axes_val   = True
        self._loss_history          = []
        self._loss_energy_history   = []
        self._loss_entropy_history  = []
        self._last_trained_params    = None
        self._slider_snapshot_at_end = None
        self._use_trained_params     = False
        self._samples_z  = None
        self._data_x     = None
        if hasattr(self, "_prog_bar"):   self._prog_bar.value = 0.0
        if hasattr(self, "_prog_epoch"): self._prog_epoch.text = "0 / 0"
        if hasattr(self, "_prog_label"):
            self._prog_label.text = ""; self._set_prog_kind("neutral")
        self._refresh_value_labels()
        self._request_render()

    def _reset_training(self):
        self._training = False
        self._loss_history          = []
        self._loss_energy_history   = []
        self._loss_entropy_history  = []
        self._train_params_pending  = None
        self._train_params_live     = None
        self._train_params_target   = None
        self._train_z_batch         = None
        self._last_trained_params    = None
        self._slider_snapshot_at_end = None
        self._use_trained_params     = False
        self._training_epoch        = 0
        self._frozen_static         = None
        self._axis_lock             = None
        self._training_was_active   = False
        if hasattr(self, "_prog_bar"):   self._prog_bar.value = 0.0
        if hasattr(self, "_prog_epoch"): self._prog_epoch.text = "0 / 0"
        if hasattr(self, "_prog_label"):
            self._prog_label.text = ""; self._set_prog_kind("neutral")
        ttype = self.transform_val
        self._suppress_redraw = True
        try:
            if ttype == "Polynomial":
                self._set_val("θ0", 0.0); self._set_val("θ1", 1.0)
                self._set_val("θ2", 0.0); self._set_val("θ3", 0.0)
            elif ttype == "Rational-quadratic spline":
                K = self.K_rqs_val
                self._set_val("rqs_B", 3.0)
                for k in range(K):
                    self._set_val(f"rqs_w{k}", 0.0)
                    self._set_val(f"rqs_h{k}", 0.0)
                for k in range(K + 1):
                    self._set_val(f"rqs_d{k}", 0.0)
            else:
                self._set_val("sig_off", 0.0); self._set_val("sig_slope", 1.0)
                defaults_c = [-2.5, -1.79, -1.07, -0.36, 0.36, 1.07, 1.79, 2.5]
                for k in range(8):
                    self._set_val(f"w{k}", 0.01)
                    self._set_val(f"c{k}", defaults_c[k])
                    self._set_val(f"s{k}", 1.0)
        finally:
            self._suppress_redraw = False
        self._refresh_value_labels()
        self._request_render()

    def _reset_map_params(self):
        self._use_trained_params = False
        self._axis_lock = None
        self._suppress_redraw = True
        try:
            ttype = self.transform_val
            if ttype == "Polynomial":
                for _, key, _, _, default, _ in self.TRANSFORM_SLIDERS:
                    self._set_val(key, default)
            elif ttype == "Rational-quadratic spline":
                K = self.K_rqs_val
                self._set_val("rqs_B", 3.0)
                for k in range(K):
                    self._set_val(f"rqs_w{k}", 0.0)
                    self._set_val(f"rqs_h{k}", 0.0)
                for k in range(K + 1):
                    self._set_val(f"rqs_d{k}", 0.0)
            else:
                defaults_c = [-2.5, -1.79, -1.07, -0.36, 0.36, 1.07, 1.79, 2.5]
                self._set_val("sig_off", 0.0); self._set_val("sig_slope", 1.0)
                for k in range(8):
                    self._set_val(f"w{k}", 0.01)
                    self._set_val(f"c{k}", defaults_c[k])
                    self._set_val(f"s{k}", 1.0)
                self.K_val = 3
                if hasattr(self, "_k_radio"): self._k_radio.value = 3
                self._update_sig_rows()
        finally:
            self._suppress_redraw = False
        self._refresh_value_labels()
        self._request_render()

    def _randomize_map_params(self):
        self._use_trained_params = False
        self._axis_lock = None
        rng = np.random.default_rng()
        self._suppress_redraw = True
        try:
            ttype = self.transform_val
            if ttype == "Polynomial":
                for _, key, lo, hi, _, _ in self.TRANSFORM_SLIDERS:
                    self._set_val(key, float(rng.uniform(lo, hi)))
            elif ttype == "Rational-quadratic spline":
                K = self.K_rqs_val
                self._set_val("rqs_B", float(rng.uniform(1.5, 5.0)))
                for k in range(K):
                    self._set_val(f"rqs_w{k}", float(rng.uniform(-2.0, 2.0)))
                    self._set_val(f"rqs_h{k}", float(rng.uniform(-2.0, 2.0)))
                for k in range(K + 1):
                    self._set_val(f"rqs_d{k}", float(rng.uniform(-1.5, 1.5)))
            else:
                self._set_val("sig_off",   float(rng.uniform(-2.0, 2.0)))
                self._set_val("sig_slope", float(rng.uniform(0.2,  2.0)))
                K = self.K_val
                for k in range(K):
                    self._set_val(f"w{k}", float(rng.uniform(0.1, 2.0)))
                    self._set_val(f"c{k}", float(rng.uniform(-3.0, 3.0)))
                    self._set_val(f"s{k}", float(rng.uniform(0.1, 2.0)))
        finally:
            self._suppress_redraw = False
        self._refresh_value_labels()
        self._request_render()

    # ── Transform evaluation ──────────────────────────────────────────────────

    def _eval_transform(self, z):
        """Return (x, J, is_monotone) for the active transformation."""
        live  = self._train_params_live
        if live is None and self._use_trained_params and \
                self._last_trained_params is not None:
            live = self._last_trained_params
        ttype = self.transform_val
        if ttype == "Polynomial":
            if live is not None:
                θ0, θ1, θ2, θ3 = live
            else:
                θ0 = self.vals["θ0"]; θ1 = self.vals["θ1"]
                θ2 = self.vals["θ2"]; θ3 = self.vals["θ3"]
            x = θ0 + θ1*z + θ2*z**2 + θ3*z**3
            J = θ1 + 2*θ2*z + 3*θ3*z**2
            return x, J, np.all(J > 1e-7) or np.all(J < -1e-7)
        elif ttype == "Rational-quadratic spline":
            K = self.K_rqs_val
            p = live if live is not None else self._get_params()
            B, widths, heights, derivs, xk, yk = self._rqs_to_knots(p, K)
            x, J = self._rqs_forward(z, B, xk, yk, widths, heights, derivs)
            return x, J, True
        else:  # Single layer perceptron
            K = self.K_val
            if live is not None:
                off   = live[0]; slope = live[1]
                ws = [live[2 + 3*k] for k in range(K)]
                cs = [live[3 + 3*k] for k in range(K)]
                ss = [max(live[4 + 3*k], 1e-6) for k in range(K)]
            else:
                off   = self.vals["sig_off"]
                slope = self.vals["sig_slope"]
                ws = [self.vals[f"w{k}"] for k in range(K)]
                cs = [self.vals[f"c{k}"] for k in range(K)]
                ss = [max(self.vals[f"s{k}"], 1e-6) for k in range(K)]
            x = np.full_like(z, off, dtype=float) + slope * z
            J = np.full_like(z, slope, dtype=float)
            for w, c, s in zip(ws, cs, ss):
                t  = np.clip((z - c) / s, -50, 50)
                sg = 1.0 / (1.0 + np.exp(-t))
                x  = x + w * sg
                J  = J + (w / s) * sg * (1.0 - sg)
            return x, J, True

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _request_render(self):
        """Mark the figure dirty; the _tick timer renders it off the event
        loop.  Non-blocking — safe to call from any UI event handler."""
        self._render_dirty = True

    async def _render_async(self):
        """Render off the event loop.  A full matplotlib redraw + SVG encode
        takes ~200 ms; doing it on the loop would starve the websocket and
        drop the client connection.  run.io_bound moves the heavy work to a
        worker thread; only the tiny enqueue_update runs on the loop.
        A single-flight guard prevents overlapping renders.
        _rendering is set True by the caller (_tick) at schedule time."""
        try:
            while True:
                self._render_dirty = False
                await run.io_bound(self._draw_and_convert)
                Element.update(self._plot)        # emit only (loop-safe)
                # During training, render one frame per dispatch so the
                # min-gap throttle in _tick leaves the training thread clear
                # GIL windows.  Idle: coalesce any frame requested mid-render.
                if self._training or self._training_was_active \
                        or not self._render_dirty:
                    break
        finally:
            self._last_render_t = time.monotonic()
            self._rendering = False

    def _draw_and_convert(self):
        """Heavy work, run in a worker thread: draw the figure and encode it
        to SVG (stored in the element's innerHTML prop)."""
        self._draw_figure()
        self._plot._convert_to_html()

    def _redraw(self, force_draw=False):
        """Synchronous draw + update.  Used once at start-up (before the event
        loop runs) and for one-off blocking renders on button clicks."""
        self._draw_figure()
        self._plot.update()

    def _draw_figure(self):
        frozen = self._frozen_static
        if frozen is not None:
            μ = frozen["μ"]; σ = frozen["σ"]; dist = frozen["dist"]
            z = frozen["z"]; p_z = frozen["p_z"]; N = z.size
        else:
            μ = self.vals["mu"]; σ = self.vals["sg"]; dist = self.dist_val
            N = 500
            # Fixed threshold — independent of "Show exact transformation" so
            # that toggling that checkbox does not change the z (or x) range.
            thr = 0.005
            if dist == "Bimodal":
                _half = abs(μ) + 15.0 * σ
                z_probe = np.linspace(-_half, _half, 800)
            else:
                z_probe = np.linspace(μ - 15.0*σ, μ + 15.0*σ, 800)
            p_probe = latent_pdf(z_probe, μ, σ, dist)
            peak = p_probe.max() or 1.0
            sig_idx = np.where(p_probe > thr * peak)[0]
            if sig_idx.size > 0:
                z_sig_lo = float(z_probe[sig_idx[0]])
                z_sig_hi = float(z_probe[sig_idx[-1]])
            else:
                if dist == "Bimodal":
                    z_sig_lo = -(abs(μ) + 4.5*σ); z_sig_hi = abs(μ) + 4.5*σ
                else:
                    z_sig_lo, z_sig_hi = μ - 4.5*σ, μ + 4.5*σ
            pad_z = (z_sig_hi - z_sig_lo) * 0.12
            z = np.linspace(z_sig_lo - pad_z, z_sig_hi + pad_z, N)
            p_z = latent_pdf(z, μ, σ, dist)

        x, J, is_monotone = self._eval_transform(z)
        Jabs = np.abs(J)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_Jabs  = np.where(Jabs > 1e-9, 1.0 / Jabs, np.nan)
            p_x_curve = np.where(Jabs > 1e-9, p_z / Jabs, 0.0)
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]; p_x_sorted = p_x_curve[sort_idx]

        ax_top, ax_main, ax_logj, ax_right = (
            self.ax_top, self.ax_main, self.ax_logj, self.ax_right)

        if not self._rescale_axes_val:
            _xlim = ax_main.get_xlim(); _ylim = ax_main.get_ylim()
            _hz_xlim = self.ax_hist_z.get_xlim()
            _hx_xlim = self.ax_hist_x.get_xlim()
            if _xlim == (0.0, 1.0) and _ylim == (0.0, 1.0): _xlim = _ylim = None
            if _hz_xlim == (0.0, 1.0): _hz_xlim = None
            if _hx_xlim == (0.0, 1.0): _hx_xlim = None
        else:
            _xlim = _ylim = _hz_xlim = _hx_xlim = None

        _show_grid = not self._show_map_lines_val

        def _grid(ax):
            ax.grid(True, alpha=0.22, linestyle="--", zorder=0) if _show_grid \
                else ax.grid(False)

        for ax in (ax_main, ax_logj, ax_right):
            ax.cla(); _grid(ax); ax.tick_params(labelsize=10)

        if frozen is None:
            ax_top.cla(); _grid(ax_top); ax_top.tick_params(labelsize=10)
            ax_top.plot(z, p_z, color=self.CZ, lw=2, zorder=3, clip_on=False)
            ax_top.fill_between(z, p_z, alpha=self.FA, color=self.CZ, zorder=2)
            ax_top.set_ylabel("$p_z(z)$", fontsize=12, labelpad=4)
            ax_top.set_ylim(bottom=0)
            ax_top.spines[["top", "right", "bottom"]].set_visible(False)
            ax_top.tick_params(labelbottom=False, bottom=False, labelsize=10)

        if not self._show_map_lines_val:
            ax_main.axhline(0, color="#999", lw=0.7, zorder=1)
            ax_main.axvline(0, color="#999", lw=0.7, zorder=1)
        pos = J >= 0
        for mask, color in [(pos, self.CT_POS), (~pos, self.CT_NEG)]:
            idxs = np.where(mask)[0]
            if idxs.size == 0: continue
            for run in np.split(idxs, np.where(np.diff(idxs) != 1)[0] + 1):
                if run.size >= 2:
                    ax_main.plot(z[run], x[run], color=color, lw=2, zorder=3)
        if not is_monotone:
            for idx in np.where(np.diff(np.sign(J)))[0]:
                zt = (z[idx] + z[idx+1]) / 2
                xt_arr, _, _ = self._eval_transform(np.array([zt]))
                ax_main.plot(zt, xt_arr[0], "o", color="#FF6D00", ms=8, zorder=5,
                             markeredgecolor="white", markeredgewidth=1.2)
            ax_main.text(0.02, 0.98, "Not invertible ✗",
                         transform=ax_main.transAxes, color="#B71C1C",
                         fontsize=10, va="top", fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFEBEE",
                                   edgecolor="#B71C1C", lw=1.2))
        ax_main.set_ylabel("$x = f_{\\theta}(z)$", fontsize=12, labelpad=4)
        ax_main.spines[["top", "right", "bottom", "left"]].set_visible(True)
        ax_main.tick_params(labelbottom=False, bottom=False)

        for mask, color in [(pos, self.CT_POS), (~pos, self.CT_NEG)]:
            idxs = np.where(mask)[0]
            if idxs.size == 0: continue
            for run in np.split(idxs, np.where(np.diff(idxs) != 1)[0] + 1):
                if run.size >= 2:
                    ax_logj.plot(z[run], inv_Jabs[run], color=color, lw=2, zorder=3)
        ax_logj.axhline(1, color="#999", lw=0.8, ls="--", zorder=1)
        _iJ_finite = inv_Jabs[np.isfinite(inv_Jabs)]
        _iJ_top = (np.percentile(_iJ_finite, 99)*1.15 if _iJ_finite.size > 0 else 2.0)
        ax_logj.set_ylim(0, max(_iJ_top, 0.1))
        ax_logj.set_xlabel("$z$", fontsize=12, labelpad=4)
        ax_logj.set_ylabel(r"$|J|^{-1}$", fontsize=12, labelpad=4)
        ax_logj.spines[["top", "right"]].set_visible(False)

        if is_monotone:
            x_right, p_x_right = x_sorted, p_x_sorted
            ax_right.plot(p_x_right, x_right, color=self.CX, lw=2, zorder=3,
                          clip_on=False)
            ax_right.fill_betweenx(x_right, p_x_right,
                                   alpha=self.FA, color=self.CX, zorder=2)
        else:
            x_right = p_x_right = np.array([])
            try:
                rng0 = np.random.default_rng(0)
                z_s = sample_latent(5_000, μ, σ, dist, rng0)
                x_s, _, _ = self._eval_transform(z_s)
                kde = gaussian_kde(x_s)
                x_right = np.linspace(x_s.min(), x_s.max(), 400)
                p_x_right = kde(x_right)
                ax_right.plot(p_x_right, x_right, color=self.CX, lw=2,
                              ls="--", zorder=3, clip_on=False)
                ax_right.fill_betweenx(x_right, p_x_right,
                                       alpha=self.FA, color=self.CX, zorder=2)
                ax_right.text(0.95, 0.02, "KDE", transform=ax_right.transAxes,
                              color="#B71C1C", fontsize=9, va="bottom", ha="right",
                              bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE",
                                        edgecolor="#B71C1C", lw=1.2))
            except Exception:
                pass
        ax_right.set_xlabel("$p_x(x)$", fontsize=12, labelpad=4)
        ax_right.spines[["top", "right", "left"]].set_visible(False)
        ax_right.tick_params(labelleft=False, left=False)

        kT = self.vals["kT"]; u1b = self.vals["u1"]; u2b = self.vals["u2"]
        u3b = self.vals["u3"]; u4b = self.vals["u4"]
        p_tgt_max = 0.0; x_tgt_lo = x_tgt_hi = None
        x_disp_tgt = p_tgt_curve = None
        show_tgt = self._show_target_val; show_exact = self._show_exact_val
        if show_tgt or show_exact:
            try:
                tgt_key = (round(kT,3), round(u1b,3), round(u2b,3),
                           round(u3b,3), round(u4b,3))
                if tgt_key == self._tgt_cache_key:
                    x_wide, p_wide, Z, shift = self._tgt_cache_val
                else:
                    x_wide = np.linspace(-15, 15, 2000)
                    U_wide = u1b*x_wide + u2b*x_wide**2 + u3b*x_wide**3 + u4b*x_wide**4
                    log_p = -U_wide / kT; shift = log_p.max()
                    p_wide = np.exp(log_p - shift)
                    Z = _trapz(p_wide, x_wide)
                    self._tgt_cache_key = tgt_key
                    self._tgt_cache_val = (x_wide, p_wide, Z, shift)
                if Z > 0:
                    dx = x_wide[1] - x_wide[0]; cdf = np.cumsum(p_wide)*dx/Z
                    thr_ = 0.001
                    lo_idx = max(0, np.searchsorted(cdf, thr_))
                    hi_idx = min(len(x_wide)-1, np.searchsorted(cdf, 1.0-thr_))
                    x_tgt_lo = float(x_wide[lo_idx]); x_tgt_hi = float(x_wide[hi_idx])
                    margin = (x_tgt_hi - x_tgt_lo)*0.10
                    x_tgt_lo -= margin; x_tgt_hi += margin
                    x_disp = np.linspace(x_tgt_lo, x_tgt_hi, 500)
                    U_disp = u1b*x_disp + u2b*x_disp**2 + u3b*x_disp**3 + u4b*x_disp**4
                    p_tgt = np.exp(-U_disp/kT - shift)/Z; p_tgt_max = p_tgt.max()
                    if show_tgt:
                        x_disp_tgt = x_disp; p_tgt_curve = p_tgt
                        ax_right.plot(p_tgt, x_disp, color=self.CT_TARGET, lw=2,
                                      zorder=4, clip_on=False)
                        ax_right.fill_betweenx(x_disp, p_tgt, alpha=self.FA,
                                               color=self.CT_TARGET, zorder=3)
                    if show_exact:
                        dz_grid = z[1] - z[0]
                        cdf_z = np.cumsum(p_z)*dz_grid
                        if cdf_z[-1] > 0: cdf_z = cdf_z/cdf_z[-1]
                        x_star = np.interp(cdf_z, cdf, x_wide)
                        # Drawn clipped to the current axes — it must not
                        # change the z/x range (see range note below).
                        ax_main.plot(z, x_star, color=self.CT_TARGET, lw=1.6,
                                     ls="-", zorder=6, label="exact")
            except Exception:
                pass

        p_x_max = p_x_right.max() if p_x_right.size > 0 else 0.0
        ax_right.set_xlim(0, max(p_x_max, p_tgt_max)*1.25 or 1.0)

        x_curve_lo = float(x.min()); x_curve_hi = float(x.max())
        # Base range = the transformation curve with 12 % padding (same whether
        # or not the target is shown).  Only EXTEND it to include the target
        # when "Show target" is on AND the target is wider — so showing a target
        # that is narrower than the transformed distribution leaves the range
        # unchanged (no spurious rescale).  ("Show exact" never affects this.)
        span = x_curve_hi - x_curve_lo or 1.0
        y_lo_main = x_curve_lo - 0.12*span
        y_hi_main = x_curve_hi + 0.12*span
        if show_tgt and x_tgt_lo is not None:
            y_lo_main = min(y_lo_main, x_tgt_lo)
            y_hi_main = max(y_hi_main, x_tgt_hi)
        ax_main.set_ylim(y_lo_main, y_hi_main)
        ax_main.set_xlim(float(z.min()), float(z.max()))

        if self._show_map_lines_val:
            try:
                n_ml_str = self._n_map_pts_val
                if hasattr(self, "_n_map_input"):
                    n_ml_str = self._n_map_input.value
                n_ml = max(1, min(100, int(n_ml_str)))
                if int(n_ml_str) > 100 and hasattr(self, "_n_map_input"):
                    self._n_map_input.value = "100"
            except (ValueError, AttributeError):
                n_ml = 0
            if n_ml > 0:
                z_ml = np.linspace(float(z.min()), float(z.max()), n_ml)
                x_ml, _, _ = self._eval_transform(z_ml)
                CML, AML, LML = self.CM, 0.6, 0.7
                x_hi_ml = float(z.max())
                for zi, xi in zip(z_ml, x_ml):
                    # L-shaped guide confined to the transformation plot:
                    # vertical from the top edge down to the curve, then
                    # horizontal from the curve out to the right edge.
                    ax_main.plot([zi, zi], [y_hi_main, xi],
                                 color=CML, lw=LML, alpha=AML, ls="-", zorder=2)
                    ax_main.plot([zi, x_hi_ml], [xi, xi],
                                 color=CML, lw=LML, alpha=AML, ls="-", zorder=2)

        self.ax_hist_x.cla()
        self.ax_hist_x.grid(True, alpha=0.22, linestyle="--", zorder=0)
        self.ax_hist_x.tick_params(labelsize=10)
        self.ax_hist_x_twin.cla()
        self.ax_hist_x_twin.set_yticks([])
        self.ax_hist_x_twin.spines["right"].set_visible(False)
        self.ax_hist_x_twin.spines["top"].set_visible(False)
        train_batch = self._train_z_batch if self._training else None
        if frozen is None or train_batch is not None:
            self.ax_hist_z.cla()
            self.ax_hist_z.grid(True, alpha=0.22, linestyle="--", zorder=0)
            self.ax_hist_z.tick_params(labelsize=10)

        display_samples = train_batch if train_batch is not None else self._samples_z

        if display_samples is not None:
            if frozen is None or train_batch is not None:
                self.ax_hist_z.hist(display_samples, bins=40, density=True,
                                    color=self.CZ, alpha=0.4, zorder=2)
                self.ax_hist_z.plot(z, p_z, color=self.CZ, lw=2, zorder=3,
                                    clip_on=False)
                self.ax_hist_z.set_ylim(0, p_z.max()*1.3)
                # Match the main plot's z-axis range (locked range if
                # auto-rescale is off, else the current z-grid extent).
                _z_lo, _z_hi = (_xlim if _xlim is not None
                                else (float(z.min()), float(z.max())))
                self.ax_hist_z.set_xlim(_z_lo, _z_hi)
                self.ax_hist_z.set_xlabel("$z$", fontsize=12, labelpad=4)
                self.ax_hist_z.set_ylabel("density", fontsize=12, labelpad=4)
                self.ax_hist_z.set_title("Latent $z$", fontsize=12, pad=3)
                self.ax_hist_z.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))
                self.ax_hist_z.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
                self.ax_hist_z.tick_params(axis="both", labelsize=10)
                self.ax_hist_z.spines[["top", "right"]].set_visible(False)

            samples_x, _, _ = self._eval_transform(display_samples)
            self.ax_hist_x.hist(samples_x, bins=40, density=True,
                                color=self.CX, alpha=0.4, zorder=2)
            y_top = 0.0
            if is_monotone:
                self.ax_hist_x.plot(x_sorted, p_x_sorted,
                                    color=self.CX, lw=2, zorder=3, clip_on=False)
                y_top = p_x_sorted.max()
            else:
                try:
                    kde2 = gaussian_kde(samples_x)
                    xq = np.linspace(samples_x.min(), samples_x.max(), 400)
                    p_xq = kde2(xq)
                    self.ax_hist_x.plot(xq, p_xq, color=self.CX, lw=2,
                                        ls="--", zorder=3, clip_on=False)
                    y_top = p_xq.max()
                except Exception:
                    pass
            if x_disp_tgt is not None and p_tgt_curve is not None:
                self.ax_hist_x.plot(x_disp_tgt, p_tgt_curve,
                                    color=self.CT_TARGET, lw=2, zorder=4,
                                    clip_on=False)
                self.ax_hist_x.fill_between(x_disp_tgt, p_tgt_curve,
                                            alpha=self.FA,
                                            color=self.CT_TARGET, zorder=3)
                y_top = max(y_top, float(p_tgt_curve.max()))
            if self._data_x is not None and self._show_data_val:
                self.ax_hist_x.hist(self._data_x, bins=40, density=True,
                                    color=self.CT_TARGET, alpha=0.35,
                                    zorder=2, label="data")
                y_top = max(y_top,
                            float(np.histogram(self._data_x, bins=40,
                                               density=True)[0].max()))
            if y_top > 0:
                self.ax_hist_x.set_ylim(0, y_top*1.3)
            # Match the main plot's x-axis range (the vertical axis of the
            # transformation panel): locked range if auto-rescale is off, else
            # the current main x-extent.
            _x_lo, _x_hi = (_ylim if _ylim is not None
                            else (y_lo_main, y_hi_main))
            self.ax_hist_x.set_xlim(_x_lo, _x_hi)
            self.ax_hist_x.set_xlabel("$x$", fontsize=12, labelpad=4)
            self.ax_hist_x.set_ylabel("density", fontsize=12, labelpad=4)
            self.ax_hist_x.set_title("Transformed $x$", fontsize=12, pad=3)
            self.ax_hist_x.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))
            self.ax_hist_x.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
            self.ax_hist_x.tick_params(axis="both", labelsize=10)
            self.ax_hist_x.spines[["top", "right"]].set_visible(False)

            if (show_tgt and is_monotone and self._show_iw_val
                    and self._tgt_cache_val is not None):
                try:
                    _, _, Z_w, shift_w = self._tgt_cache_val
                    if Z_w > 0:
                        # warm brown; brighter tan in dark mode for contrast
                        CW = "#C9A27A" if self._dark else "#6D4C41"
                        ax_tw = self.ax_hist_x_twin
                        ok_px = p_x_sorted > 1e-10
                        U_xs = (u1b*x_sorted + u2b*x_sorted**2
                                + u3b*x_sorted**3 + u4b*x_sorted**4)
                        pstar_xs = np.exp(-U_xs/kT - shift_w)/Z_w
                        w_smooth = np.full_like(x_sorted, np.nan)
                        w_smooth[ok_px] = pstar_xs[ok_px]/p_x_sorted[ok_px]
                        _, J_samp, _ = self._eval_transform(display_samples)
                        pz_samp = latent_pdf(display_samples, μ, σ, dist)
                        px_samp = np.where(np.abs(J_samp) > 1e-9,
                                           pz_samp/np.abs(J_samp), 0.0)
                        sx, _, _ = self._eval_transform(display_samples)
                        U_samp = (u1b*sx + u2b*sx**2 + u3b*sx**3 + u4b*sx**4)
                        pstar_s = np.exp(-U_samp/kT - shift_w)/Z_w
                        ok_w = ((px_samp > 1e-10) & np.isfinite(pstar_s)
                                & np.isfinite(px_samp))
                        if ok_w.any() and np.any(np.isfinite(w_smooth)):
                            w_samp = pstar_s[ok_w]/px_samp[ok_w]
                            N_eff = float(w_samp.sum()**2/(w_samp**2).sum())
                            N_tot = int(ok_w.sum())
                            ax_tw.plot(x_sorted, w_smooth, color=CW, lw=1.4,
                                       ls="--", zorder=5)
                            ax_tw.axhline(1.0, color=CW, lw=0.8, ls=":",
                                          zorder=4, alpha=0.6)
                            w_fin = w_smooth[np.isfinite(w_smooth)]
                            w_hi = float(np.percentile(w_fin, 99))*1.25
                            ax_tw.set_ylim(0, max(w_hi, 1.5))
                            ax_tw.set_ylabel("$w(x)$", fontsize=10,
                                             color=CW, labelpad=4)
                            ax_tw.yaxis.set_label_position("right")
                            ax_tw.tick_params(axis="y", labelsize=8, labelcolor=CW)
                            ax_tw.yaxis.set_major_locator(plt.MaxNLocator(nbins=3))
                            ax_tw.spines["right"].set_visible(True)
                            ax_tw.spines["right"].set_color(CW)
                            ax_tw.text(0.97, 0.97,
                                       f"$N_{{\\mathrm{{eff}}}}="
                                       f"{100*N_eff/N_tot:.1f}\\%$",
                                       transform=ax_tw.transAxes,
                                       ha="right", va="top", fontsize=8, color=CW)
                except Exception:
                    pass
        else:
            _ph_fc = "#3a3a42" if self._dark else "#f5f5f5"
            _ph_ec = "#555"    if self._dark else "#ccc"
            empty_axes = [(self.ax_hist_x, "Transformed $x$", "$x$")]
            if frozen is None:
                empty_axes.insert(0, (self.ax_hist_z, "Latent $z$", "$z$"))
            for ax, name, xlab in empty_axes:
                ax.set_title(name, fontsize=12, pad=3)
                ax.set_xlabel(xlab, fontsize=12, labelpad=4)
                ax.set_ylabel("density", fontsize=12, labelpad=4)
                ax.spines[["top", "right"]].set_visible(False)
                ax.text(0.5, 0.5, "Press  'Sample!'\nto generate points",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=9, color="#888",
                        bbox=dict(boxstyle="round,pad=0.4", facecolor=_ph_fc,
                                  edgecolor=_ph_ec, alpha=0.9))
                ax.tick_params(labelbottom=False, bottom=False,
                               labelleft=False, left=False)

        self.ax_loss.cla()
        self.ax_loss.grid(True, alpha=0.22, linestyle="--", zorder=0)
        self.ax_loss.set_title("Training loss", fontsize=12, pad=22)
        self.ax_loss.spines[["top", "right"]].set_visible(False)
        if self._loss_history:
            n_pts = len(self._loss_history); epochs = np.arange(1, n_pts+1)
            MAX_PLOT_PTS = 100; n_total = max(n_pts, self._n_epochs_total)
            if n_total > MAX_PLOT_PTS:
                step = max(1, n_total // MAX_PLOT_PTS)
                max_idx = int(np.argmax(self._loss_history))
                idx = np.unique(np.r_[np.arange(0, n_pts, step), max_idx])
                ep_plot = epochs[idx]
                tot_plot  = np.asarray(self._loss_history)[idx]
                ener_plot = np.asarray(self._loss_energy_history)[idx]
                entr_plot = np.asarray(self._loss_entropy_history)[idx]
            else:
                ep_plot = epochs; tot_plot = self._loss_history
                ener_plot = self._loss_energy_history
                entr_plot = self._loss_entropy_history
            LW = 1.2; mode = self._train_mode_val
            lbl1 = (r"$\langle U\rangle/k_BT$" if mode == "Energy-based"
                    else r"$-\langle\log p_z\rangle$")
            lbl2 = (r"$-\langle\log|J|\rangle$" if mode == "Energy-based"
                    else r"$\langle\log|J|\rangle$")
            if self._loss_energy_history:
                self.ax_loss.plot(ep_plot, ener_plot, color=self.CL_ENER,
                                  lw=LW, zorder=2, label=lbl1)
            if self._loss_entropy_history:
                self.ax_loss.plot(ep_plot, entr_plot, color=self.CL_ENTR,
                                  lw=LW, zorder=2, label=lbl2)
            # the total/NLL curve is dark slate in light mode; brighten it for
            # dark mode so it stays visible against the dark background
            _cl = "#ECEFF1" if self._dark else self.CL
            self.ax_loss.plot(ep_plot, tot_plot, color=_cl, lw=LW, zorder=3,
                              label="NLL" if mode=="Example-based" else "total")
            self.ax_loss.legend(fontsize=11, loc="lower center",
                                bbox_to_anchor=(0.5, 0.93), ncol=3, frameon=False,
                                handlelength=1.2, columnspacing=0.9, handletextpad=0.3)
            self.ax_loss.set_xlabel("epoch", fontsize=12, labelpad=4)
            self.ax_loss.set_ylabel("loss",  fontsize=12, labelpad=4)
            self.ax_loss.set_xlim(1, max(epochs))
            _all = [v for v in (self._loss_history + self._loss_energy_history
                                + self._loss_entropy_history) if np.isfinite(v)]
            if _all:
                _ly, _hy = min(_all), max(_all)
                _pad = (_hy-_ly)*0.08 or abs(_hy)*0.08 or 0.1
                self.ax_loss.set_ylim(_ly-_pad, _hy+_pad)
            self.ax_loss.xaxis.set_major_locator(plt.MaxNLocator(nbins=4, integer=True))
            self.ax_loss.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
            self.ax_loss.tick_params(axis="both", which="both", direction="in",
                                     labelsize=10, bottom=True, left=True,
                                     labelbottom=True, labelleft=True)
        else:
            self.ax_loss.set_xlabel("epoch", fontsize=12, labelpad=4)
            self.ax_loss.set_ylabel("loss",  fontsize=12, labelpad=4)
            self.ax_loss.text(0.5, 0.5, "Press  'Train!'\nto start training",
                              transform=self.ax_loss.transAxes,
                              ha="center", va="center", fontsize=9, color="#888",
                              bbox=dict(boxstyle="round,pad=0.4",
                                        facecolor=("#3a3a42" if self._dark else "#f5f5f5"),
                                        edgecolor=("#555" if self._dark else "#ccc"),
                                        alpha=0.9))
            self.ax_loss.tick_params(labelbottom=False, bottom=False,
                                     labelleft=False, left=False)

        if _xlim is not None:    ax_main.set_xlim(_xlim)
        if _ylim is not None:    ax_main.set_ylim(_ylim)
        # (histogram x-ranges are tied to the main z/x axes above, so they need
        # no separate restore here.)

        self._apply_fig_theme()   # recolour for dark / light mode

    # ── Sampling ──────────────────────────────────────────────────────────────

    def _do_sampling(self):
        try:
            N = max(1, int(self._n_entry_input.value))
        except (ValueError, AttributeError):
            N = 1000
        μ    = self.vals["mu"]
        σ    = self.vals["sg"]
        dist = self.dist_val
        self._samples_z = sample_latent(N, μ, σ, dist, np.random.default_rng())
        self._request_render()

    def _do_generate_data(self):
        """Generate N example data points from the target via CDF inversion."""
        try:
            N = max(1, int(self._n_entry_input.value))
        except (ValueError, AttributeError):
            N = 1000
        kT  = self.vals["kT"];  u1b = self.vals["u1"]; u2b = self.vals["u2"]
        u3b = self.vals["u3"];  u4b = self.vals["u4"]
        tgt_key = (round(kT,3), round(u1b,3), round(u2b,3),
                   round(u3b,3), round(u4b,3))
        if tgt_key == self._tgt_cache_key:
            x_wide, p_wide, Z, shift = self._tgt_cache_val
        else:
            x_wide = np.linspace(-15, 15, 2000)
            U_wide = u1b*x_wide + u2b*x_wide**2 + u3b*x_wide**3 + u4b*x_wide**4
            log_p  = -U_wide / kT; shift = log_p.max()
            p_wide = np.exp(log_p - shift)
            Z      = _trapz(p_wide, x_wide)
            self._tgt_cache_key = tgt_key
            self._tgt_cache_val = (x_wide, p_wide, Z, shift)
        if Z <= 0:
            return
        dx    = x_wide[1] - x_wide[0]
        cdf_x = np.cumsum(p_wide) * dx / Z
        cdf_x = np.clip(cdf_x, 0.0, 1.0)
        μ = self.vals["mu"]; σ = self.vals["sg"]; dist = self.dist_val
        rng = np.random.default_rng()
        z_samp = sample_latent(N, μ, σ, dist, rng)
        z_grid = np.linspace(z_samp.min() - 3*σ, z_samp.max() + 3*σ, 4000)
        p_grid = latent_pdf(z_grid, μ, σ, dist)
        dz     = z_grid[1] - z_grid[0]
        cdf_z_g = np.cumsum(p_grid) * dz
        cdf_z_g = np.clip(cdf_z_g / max(cdf_z_g[-1], 1e-12), 0.0, 1.0)
        u_vals = np.interp(z_samp, z_grid, cdf_z_g)
        self._data_x = np.interp(u_vals, cdf_x, x_wide)
        self._request_render()

    # ── Training ──────────────────────────────────────────────────────────────

    def _get_param_keys(self):
        ttype = self.transform_val
        if ttype == "Polynomial":
            return ["θ0", "θ1", "θ2", "θ3"]
        if ttype == "Rational-quadratic spline":
            K = self.K_rqs_val
            keys = ["rqs_B"]
            keys += [f"rqs_w{k}" for k in range(K)]
            keys += [f"rqs_h{k}" for k in range(K)]
            keys += [f"rqs_d{k}" for k in range(K + 1)]
            return keys
        K = self.K_val
        keys = ["sig_off", "sig_slope"]
        for k in range(K):
            keys += [f"w{k}", f"c{k}", f"s{k}"]
        return keys

    def _eval_transform_params(self, z, params, transform_type=None, K=None):
        if transform_type is None:
            transform_type = self.transform_val
        if transform_type == "Polynomial":
            θ0, θ1, θ2, θ3 = params
            x = θ0 + θ1*z + θ2*z**2 + θ3*z**3
            J = θ1 + 2*θ2*z + 3*θ3*z**2
        elif transform_type == "Rational-quadratic spline":
            K_rqs = K if K is not None else self.K_rqs_val
            B, widths, heights, derivs, xk, yk = self._rqs_to_knots(params, K_rqs)
            x, J = self._rqs_forward(z, B, xk, yk, widths, heights, derivs)
        else:
            if K is None:
                K = self.K_val
            off = params[0]; slope = params[1]
            x = np.full_like(z, off, dtype=float) + slope * z
            J = np.full_like(z, slope, dtype=float)
            for k in range(K):
                w = params[2+3*k]; c = params[3+3*k]; s = max(params[4+3*k], 1e-6)
                t  = np.clip((z - c) / s, -50, 50)
                sg = 1.0 / (1.0 + np.exp(-t))
                x += w * sg; J += (w / s) * sg * (1.0 - sg)
        return x, J

    def _compute_loss(self, params, z_batch, target=None,
                       transform_type=None, K=None, return_components=False):
        if target is None:
            kT  = self.vals["kT"];  u1b = self.vals["u1"]; u2b = self.vals["u2"]
            u3b = self.vals["u3"];  u4b = self.vals["u4"]
        else:
            kT, u1b, u2b, u3b, u4b = target
        x, J = self._eval_transform_params(z_batch, params,
                                            transform_type=transform_type, K=K)
        Jabs = np.abs(J)
        U    = u1b*x + u2b*x**2 + u3b*x**3 + u4b*x**4
        with np.errstate(divide="ignore", invalid="ignore"):
            log_J = np.where(Jabs > 1e-300, np.log(Jabs + 1e-300), -700.0)
        energy  = U / kT
        entropy = -log_J
        vals    = energy + entropy
        ok      = np.isfinite(vals)
        total   = float(np.mean(vals[ok]))
        if return_components:
            return total, float(np.mean(energy[ok])), float(np.mean(entropy[ok]))
        return total

    def _invert_transform_params(self, x_data, params, transform_type=None,
                                  K=None, tol=1e-7, max_iter=30):
        if transform_type is None:
            transform_type = self.transform_val
        if transform_type == "Rational-quadratic spline":
            K_rqs = K if K is not None else self.K_rqs_val
            B, widths, heights, derivs, xk, yk = self._rqs_to_knots(params, K_rqs)
            return self._rqs_inverse_analytic(x_data, B, xk, yk,
                                               widths, heights, derivs)
        if K is None and transform_type != "Polynomial":
            K = self.K_val
        z_lo = np.full(len(x_data), -20.0); z_hi = np.full(len(x_data),  20.0)
        for _ in range(max_iter):
            z_mid = 0.5*(z_lo + z_hi)
            x_mid, _ = self._eval_transform_params(z_mid, params,
                                                     transform_type=transform_type,
                                                     K=K)
            z_lo = np.where(x_mid < x_data, z_mid, z_lo)
            z_hi = np.where(x_mid < x_data, z_hi,  z_mid)
            if np.max(z_hi - z_lo) < tol:
                break
        return 0.5*(z_lo + z_hi)

    def _compute_loss_example(self, params, x_data, latent,
                               transform_type=None, K=None,
                               return_components=False):
        μ, σ, dist = latent
        if transform_type is None: transform_type = self.transform_val
        if K is None and transform_type != "Polynomial": K = self.K_val
        z = self._invert_transform_params(x_data, params,
                                           transform_type=transform_type, K=K)
        _, J = self._eval_transform_params(z, params,
                                             transform_type=transform_type, K=K)
        Jabs = np.abs(J)
        log_pz = log_latent_pdf(z, μ, σ, dist)
        log_J  = np.where(Jabs > 1e-300, np.log(Jabs + 1e-300), -700.0)
        vals   = -log_pz + log_J; ok = np.isfinite(vals)
        total  = float(np.mean(vals[ok]))
        if return_components:
            return (total, float(np.mean(-log_pz[ok])), float(np.mean(log_J[ok])))
        return total

    def _gradient_analytic(self, params, z_batch, target=None,
                            transform_type=None, K=None):
        """Analytical gradient of L = E[U(f(z))/kT - log|J(z)|].
        z is the fixed sampled batch (no implicit θ-dependence)."""
        if target is None:
            kT  = self.vals["kT"];  u1b = self.vals["u1"]; u2b = self.vals["u2"]
            u3b = self.vals["u3"];  u4b = self.vals["u4"]
        else:
            kT, u1b, u2b, u3b, u4b = target
        if transform_type is None: transform_type = self.transform_val
        if K is None and transform_type != "Polynomial": K = self.K_val
        z = z_batch; grad = np.zeros(len(params))

        if transform_type == "Polynomial":
            θ0, θ1, θ2, θ3 = params
            x = θ0 + θ1*z + θ2*z**2 + θ3*z**3
            J = θ1 + 2*θ2*z + 3*θ3*z**2
            dUdx = u1b + 2*u2b*x + 3*u3b*x**2 + 4*u4b*x**3
            ok = np.isfinite(J) & np.isfinite(dUdx) & (np.abs(J) > 1e-10)
            for i, (dx_dθ, dJ_dθ) in enumerate([
                    (np.ones_like(z),  np.zeros_like(z)),
                    (z,                np.ones_like(z)),
                    (z**2,             2*z),
                    (z**3,             3*z**2)]):
                v = dUdx * dx_dθ / kT - dJ_dθ / J
                grad[i] = np.mean(v[ok & np.isfinite(v)])
        else:
            off   = params[0]
            slope = params[1]
            x = np.full_like(z, off, dtype=float) + slope * z
            J = np.full_like(z, slope, dtype=float)
            sigs = []
            for k in range(K):
                w = params[2 + 3*k]; c = params[3 + 3*k]
                s = max(params[4 + 3*k], 1e-6)
                t = np.clip((z - c) / s, -50, 50)
                sg = 1.0 / (1.0 + np.exp(-t))
                x += w * sg
                J += w * sg * (1 - sg) / s
                sigs.append((w, c, s, sg))
            dUdx = u1b + 2*u2b*x + 3*u3b*x**2 + 4*u4b*x**3
            ok = np.isfinite(J) & np.isfinite(dUdx) & (np.abs(J) > 1e-10)
            v = dUdx / kT
            grad[0] = np.mean(v[ok & np.isfinite(v)])
            v = dUdx * z / kT - 1.0 / J
            grad[1] = np.mean(v[ok & np.isfinite(v)])
            for k, (w, c, s, sg) in enumerate(sigs):
                dsg = sg * (1 - sg); zc = z - c
                v = dUdx * sg / kT - dsg / (s * J)
                grad[2 + 3*k] = np.mean(v[ok & np.isfinite(v)])
                dx_dc = -w * dsg / s
                dJ_dc = -w * dsg * (1 - 2*sg) / s**2
                v = dUdx * dx_dc / kT - dJ_dc / J
                grad[3 + 3*k] = np.mean(v[ok & np.isfinite(v)])
                dx_ds = -w * dsg * zc / s**2
                dJ_ds = -w * dsg * ((1 - 2*sg) * zc / s + 1.0) / s**2
                v = dUdx * dx_ds / kT - dJ_ds / J
                grad[4 + 3*k] = np.mean(v[ok & np.isfinite(v)])
        return grad

    def _gradient_analytic_example(self, params, x_data, latent,
                                    transform_type=None, K=None):
        μ, σ, dist = latent
        if transform_type is None: transform_type = self.transform_val
        if K is None and transform_type != "Polynomial": K = self.K_val
        z = self._invert_transform_params(x_data, params,
                                           transform_type=transform_type, K=K)
        _, J = self._eval_transform_params(z, params,
                                             transform_type=transform_type, K=K)
        Jabs = np.abs(J); ok = (Jabs > 1e-10)
        if dist == "Gaussian":   score = -(z-μ)/(σ**2)
        elif dist == "Laplace":  s_=σ/np.sqrt(2); score = -np.sign(z-μ)/s_
        elif dist == "Cauchy":   score = -2*(z-μ)/(σ**2+(z-μ)**2)
        elif dist == "Bimodal":
            s_=σ*0.6; lp1=-0.5*((z+μ)/s_)**2; lp2=-0.5*((z-μ)/s_)**2
            lse=np.logaddexp(lp1,lp2)
            score=(-(z+μ)/s_**2*np.exp(lp1-lse)-(z-μ)/s_**2*np.exp(lp2-lse))
        else:
            h=1e-5; score=(log_latent_pdf(z+h,μ,σ,dist)-log_latent_pdf(z-h,μ,σ,dist))/(2*h)
        grad = np.zeros(len(params))
        if transform_type == "Polynomial":
            θ0,θ1,θ2,θ3 = params; dJdz = 2*θ2+6*θ3*z
            alpha = np.where(ok, (score-dJdz/J)/J, 0.0)
            for i,(dfdp,dJdp) in enumerate([
                    (np.ones_like(z),np.zeros_like(z)),(z,np.ones_like(z)),
                    (z**2,2*z),(z**3,3*z**2)]):
                v = alpha*dfdp + np.where(ok,dJdp/J,0.0)
                grad[i] = np.mean(v[ok & np.isfinite(v)])
        else:
            off=params[0]; slope=params[1]; sigs=[]
            for k in range(K):
                w=params[2+3*k]; c=params[3+3*k]; s=max(params[4+3*k],1e-6)
                t=np.clip((z-c)/s,-50,50); sg=1.0/(1.0+np.exp(-t)); sigs.append((w,c,s,sg))
            dJdz=np.zeros_like(z)
            for w,c,s,sg in sigs: dJdz+=(w/s**2)*sg*(1-sg)*(1-2*sg)
            alpha=np.where(ok,(score-dJdz/J)/J,0.0)
            v=alpha*1.0; grad[0]=np.mean(v[ok & np.isfinite(v)])
            v=alpha*z+np.where(ok,1.0/J,0.0); grad[1]=np.mean(v[ok & np.isfinite(v)])
            for k,(w,c,s,sg) in enumerate(sigs):
                dsg=sg*(1-sg); dsg2=dsg*(1-2*sg)
                v=alpha*sg+np.where(ok,dsg/(s*J),0.0); grad[2+3*k]=np.mean(v[ok & np.isfinite(v)])
                dx_dc=-w*dsg/s; dJ_dc=-w*dsg2/s**2
                v=alpha*dx_dc+np.where(ok,dJ_dc/J,0.0); grad[3+3*k]=np.mean(v[ok & np.isfinite(v)])
                zc=z-c; dx_ds=-w*dsg*zc/s**2; dJ_ds=-(w/s**2)*(dsg+zc/s*dsg2)
                v=alpha*dx_ds+np.where(ok,dJ_ds/J,0.0); grad[4+3*k]=np.mean(v[ok & np.isfinite(v)])
        return grad

    def _loss_and_grad_example(self, params, x_data, latent,
                               transform_type=None, K=None):
        μ, σ, dist = latent
        if transform_type is None: transform_type = self.transform_val
        if K is None and transform_type != "Polynomial": K = self.K_val
        z = self._invert_transform_params(x_data, params,
                                           transform_type=transform_type, K=K)
        if transform_type == "Polynomial":
            θ0,θ1,θ2,θ3 = params; J=θ1+2*θ2*z+3*θ3*z**2; dJdz=2*θ2+6*θ3*z; sigs=None
        else:
            slope=params[1]; J=np.full_like(z,slope,dtype=float); sigs=[]
            for k in range(K):
                w=params[2+3*k]; c=params[3+3*k]; s=max(params[4+3*k],1e-6)
                t=np.clip((z-c)/s,-50,50); sg=1.0/(1.0+np.exp(-t))
                J+=(w/s)*sg*(1.0-sg); sigs.append((w,c,s,sg))
            dJdz=np.zeros_like(z)
            for w,c,s,sg in sigs: dJdz+=(w/s**2)*sg*(1-sg)*(1-2*sg)
        Jabs=np.abs(J); ok=Jabs>1e-10
        log_pz=log_latent_pdf(z,μ,σ,dist)
        log_J=np.where(Jabs>1e-300,np.log(Jabs+1e-300),-700.0)
        vals=-log_pz+log_J; ok_fin=np.isfinite(vals)
        if not ok_fin.any():
            return float('nan'),float('nan'),float('nan'),np.zeros(len(params))
        total=float(np.mean(vals[ok_fin]))
        ener_term=float(np.mean(-log_pz[ok_fin])); entr_term=float(np.mean(log_J[ok_fin]))
        if dist=="Gaussian":    score=-(z-μ)/σ**2
        elif dist=="Laplace":   sl=σ/np.sqrt(2); score=-np.sign(z-μ)/sl
        elif dist=="Cauchy":    score=-2*(z-μ)/(σ**2+(z-μ)**2)
        elif dist=="Bimodal":
            sb=σ*0.6; lp1=-0.5*((z+μ)/sb)**2; lp2=-0.5*((z-μ)/sb)**2
            lse=np.logaddexp(lp1,lp2)
            score=(-(z+μ)/sb**2*np.exp(lp1-lse)-(z-μ)/sb**2*np.exp(lp2-lse))
        else:
            h=1e-5; score=(log_latent_pdf(z+h,μ,σ,dist)-log_latent_pdf(z-h,μ,σ,dist))/(2*h)
        alpha=np.where(ok,(score-dJdz/J)/J,0.0); grad=np.zeros(len(params))
        if transform_type=="Polynomial":
            for i,(dfdp,dJdp) in enumerate([
                    (np.ones_like(z),np.zeros_like(z)),(z,np.ones_like(z)),
                    (z**2,2*z),(z**3,3*z**2)]):
                v=alpha*dfdp+np.where(ok,dJdp/J,0.0); grad[i]=np.mean(v[ok & np.isfinite(v)])
        else:
            v=alpha*1.0; grad[0]=np.mean(v[ok & np.isfinite(v)])
            v=alpha*z+np.where(ok,1.0/J,0.0); grad[1]=np.mean(v[ok & np.isfinite(v)])
            for k,(w,c,s,sg) in enumerate(sigs):
                dsg=sg*(1-sg); dsg2=dsg*(1-2*sg)
                v=alpha*sg+np.where(ok,dsg/(s*J),0.0); grad[2+3*k]=np.mean(v[ok & np.isfinite(v)])
                dx_dc=-w*dsg/s; dJ_dc=-w*dsg2/s**2
                v=alpha*dx_dc+np.where(ok,dJ_dc/J,0.0); grad[3+3*k]=np.mean(v[ok & np.isfinite(v)])
                zc=z-c; dx_ds=-w*dsg*zc/s**2; dJ_ds=-(w/s**2)*(dsg+zc/s*dsg2)
                v=alpha*dx_ds+np.where(ok,dJ_ds/J,0.0); grad[4+3*k]=np.mean(v[ok & np.isfinite(v)])
        return total, ener_term, entr_term, grad

    def _gradient_fd(self, params, z_batch, target=None,
                     transform_type=None, K=None, eps=1e-4):
        grad = np.zeros_like(params)
        for i in range(len(params)):
            p_p, p_m = params.copy(), params.copy()
            p_p[i] += eps; p_m[i] -= eps
            grad[i] = (self._compute_loss(p_p, z_batch, target=target,
                                           transform_type=transform_type, K=K) -
                       self._compute_loss(p_m, z_batch, target=target,
                                           transform_type=transform_type, K=K)
                       ) / (2.0 * eps)
        return grad

    def _loss_and_grad_fd_example(self, params, x_data, latent,
                                   transform_type=None, K=None, eps=1e-4):
        loss, ener, entr = self._compute_loss_example(
            params, x_data, latent, transform_type=transform_type, K=K,
            return_components=True)
        grad = np.zeros_like(params)
        for i in range(len(params)):
            p_p, p_m = params.copy(), params.copy()
            p_p[i] += eps; p_m[i] -= eps
            l_p = self._compute_loss_example(p_p, x_data, latent,
                                              transform_type=transform_type, K=K)
            l_m = self._compute_loss_example(p_m, x_data, latent,
                                              transform_type=transform_type, K=K)
            grad[i] = (l_p - l_m) / (2.0 * eps)
        return loss, ener, entr, grad

    def _rqs_to_knots(self, params, K):
        B = max(float(params[0]), 0.5); W = 2.0 * B
        w_raw = np.asarray(params[1:K+1], dtype=float)
        w_raw = w_raw - w_raw.max(); w_exp = np.exp(w_raw)
        widths = W * w_exp / w_exp.sum()
        h_raw = np.asarray(params[K+1:2*K+1], dtype=float)
        h_raw = h_raw - h_raw.max(); h_exp = np.exp(h_raw)
        heights = W * h_exp / h_exp.sum()
        d_raw = np.asarray(params[2*K+1:3*K+2], dtype=float)
        derivs = np.exp(np.clip(d_raw, -6.0, 6.0))
        x_knots = np.concatenate([[-B], -B + np.cumsum(widths)])
        y_knots = np.concatenate([[-B], -B + np.cumsum(heights)])
        return B, widths, heights, derivs, x_knots, y_knots

    def _rqs_forward(self, z, B, x_knots, y_knots, widths, heights, derivs):
        K = len(widths); z = np.asarray(z, dtype=float)
        x = z.copy(); J = np.ones_like(z)
        inside = (z > -B) & (z < B)
        if not inside.any(): return x, J
        zi = z[inside]; k = np.clip(np.searchsorted(x_knots[1:-1], zi), 0, K-1)
        Δx=widths[k]; Δy=heights[k]; xk=x_knots[k]; yk=y_knots[k]
        dk=derivs[k]; dk1=derivs[k+1]; sk=Δy/Δx
        ξ=np.clip((zi-xk)/Δx, 0.0, 1.0); ξ1=1.0-ξ; γ=dk1+dk-2.0*sk
        den=sk+γ*ξ*ξ1
        x[inside]=yk+Δy*(sk*ξ**2+dk*ξ*ξ1)/den
        J[inside]=(sk**2*(dk1*ξ**2+2.0*sk*ξ*ξ1+dk*ξ1**2)/den**2)
        return x, J

    def _rqs_inverse_analytic(self, x_out, B, x_knots, y_knots,
                               widths, heights, derivs):
        K=len(widths); x_out=np.asarray(x_out, dtype=float); z=x_out.copy()
        inside=(x_out>-B)&(x_out<B)
        if not inside.any(): return z
        xi=x_out[inside]; k=np.clip(np.searchsorted(y_knots[1:-1],xi),0,K-1)
        Δx=widths[k]; Δy=heights[k]; xk=x_knots[k]; yk=y_knots[k]
        dk=derivs[k]; dk1=derivs[k+1]; sk=Δy/Δx; τ=xi-yk; γ=dk1+dk-2.0*sk
        a=Δy*(sk-dk)+τ*γ; b=Δy*dk-τ*γ; c=-sk*τ
        disc=np.maximum(b**2-4.0*a*c, 0.0); sq=np.sqrt(disc)
        a_safe=np.where(np.abs(a)<1e-9, np.sign(a+1e-18)*1e-9, a)
        r1=(-b+sq)/(2.0*a_safe); r2=(-b-sq)/(2.0*a_safe)
        lin=np.where(np.abs(b)>1e-9,-c/b,np.zeros_like(b))
        r1=np.where(np.abs(a)<1e-9,lin,r1); r2=np.where(np.abs(a)<1e-9,lin,r2)
        r1_ok=(r1>=-1e-4)&(r1<=1.0+1e-4)
        ξ=np.clip(np.where(r1_ok,r1,r2),0.0,1.0)
        z[inside]=xk+ξ*Δx
        return z

    def _clip_params(self, params, transform_type=None, K=None):
        params = params.copy()
        if transform_type is None: transform_type = self.transform_val
        if transform_type == "Single layer perceptron":
            if K is None: K = self.K_val
            for k in range(K):
                params[2+3*k] = max(0.01, params[2+3*k])
                params[4+3*k] = max(0.05, params[4+3*k])
        elif transform_type == "Rational-quadratic spline":
            params[0] = max(0.5, params[0])
        return params

    def _freeze_static_for_training(self):
        μ = self.vals["mu"]; σ = self.vals["sg"]; dist = self.dist_val
        N = 800
        if dist == "Bimodal":
            _half = abs(μ) + 15.0 * σ
            z_probe = np.linspace(-_half, _half, 2000)
        else:
            z_probe = np.linspace(μ - 15.0*σ, μ + 15.0*σ, 2000)
        p_probe = latent_pdf(z_probe, μ, σ, dist); peak = p_probe.max() or 1.0
        sig_idx = np.where(p_probe > 0.005 * peak)[0]
        if sig_idx.size > 0:
            z_sig_lo = float(z_probe[sig_idx[0]])
            z_sig_hi = float(z_probe[sig_idx[-1]])
        else:
            if dist == "Bimodal":
                z_sig_lo = -(abs(μ)+4.5*σ); z_sig_hi = abs(μ)+4.5*σ
            else:
                z_sig_lo, z_sig_hi = μ-4.5*σ, μ+4.5*σ
        pad_z = (z_sig_hi - z_sig_lo)*0.12
        z = np.linspace(z_sig_lo-pad_z, z_sig_hi+pad_z, N)
        p_z = latent_pdf(z, μ, σ, dist)
        self._frozen_static = {"μ": μ, "σ": σ, "dist": dist, "z": z, "p_z": p_z}

    def _do_training(self):
        if self._training:
            return
        # Snapshot the text-input widgets into the plain-attribute mirrors that
        # the (background) training thread reads — the thread must not touch
        # widgets directly.
        if hasattr(self, "_n_epochs_input"): self._n_epochs_val = self._n_epochs_input.value
        if hasattr(self, "_lr_input"):       self._lr_val       = self._lr_input.value
        if hasattr(self, "_n_batch_input"):  self._n_batch_val  = self._n_batch_input.value
        if hasattr(self, "_stride_input"):   self._stride_val   = self._stride_input.value
        if hasattr(self, "_delay_input"):    self._delay_val    = self._delay_input.value
        self._train_params_pending = None
        self._training_epoch = 0
        try:
            _n_tot = max(1, int(self._n_epochs_val))
        except (ValueError, AttributeError):
            _n_tot = 0
        if hasattr(self, "_prog_bar"):   self._prog_bar.value = 0.0
        if hasattr(self, "_prog_epoch"): self._prog_epoch.text = f"0 / {_n_tot}"
        if hasattr(self, "_prog_label"):
            self._prog_label.text = ""
            self._set_prog_kind("neutral")   # reset any leftover status colour
        # Make the target distribution visible so convergence is observable,
        # and tick its checkbox to reflect that.
        self._show_target_val = True
        if hasattr(self, "_show_target_cb"):
            self._show_target_cb.value = True
        self._axis_lock = None   # rescale so the target fits in view
        current = self._get_params()
        if (self._last_trained_params is not None
                and self._slider_snapshot_at_end is not None
                and current.shape == self._slider_snapshot_at_end.shape
                and np.allclose(current, self._slider_snapshot_at_end, atol=1e-6)):
            self._train_starting_params = self._last_trained_params.copy()
        else:
            self._train_starting_params = current
        self._training = True
        self._use_trained_params = False
        self._training_was_active = True
        self._frozen_static = None
        self._redraw()
        self._use_trained_params = False
        self._freeze_static_for_training()
        threading.Thread(target=self._train_loop, daemon=True).start()

    def _stop_training(self):
        self._training = False

    def _tick(self):
        """30 ms timer (runs on the event loop): cheap bookkeeping only.
        The expensive matplotlib render is dispatched to _render_async, which
        runs it in a worker thread so this callback never blocks the loop."""
        if self._training or self._training_was_active:
            params = self._train_params_pending
            if params is not None:
                self._train_params_pending = None
                self._train_params_target  = params
                if self._train_params_live is None:
                    self._train_params_live = params.copy()

            if self._train_params_target is not None and self._train_params_live is not None:
                if not self._training:
                    self._train_params_live = self._train_params_target.copy()
                    if self._train_z_batch is not None:
                        self._samples_z = self._train_z_batch.copy()
                    self._render_dirty = True
                else:
                    diff = self._train_params_target - self._train_params_live
                    if np.max(np.abs(diff)) > 1e-5:
                        self._train_params_live = self._train_params_live + 0.25*diff
                        self._render_dirty = True

            t  = self._training_epoch; n = self._n_epochs_total
            loss = self._loss_history[-1] if self._loss_history else float('nan')
            ep_us = getattr(self, '_epoch_time_us', 0.0)
            if hasattr(self, '_prog_bar'):
                self._prog_bar.value = t / max(n, 1)
            if hasattr(self, '_prog_epoch'):
                self._prog_epoch.text = f"{t} / {n}"
            if hasattr(self, '_prog_label'):
                self._prog_label.text = f"loss {loss:.3f}  ·  {ep_us:.0f} μs"

            if not self._training:
                if self._training_was_active:
                    self._training_was_active = False
                    self._do_end_of_training()
            else:
                self._training_was_active = True

        # Unified render dispatch — schedule an off-loop render if one is
        # pending and none is already running.  _rendering is set here (at
        # schedule time) so a second task can't slip in before the first runs.
        # During training, enforce a minimum gap between renders so the figure
        # work (which is GIL-heavy) doesn't starve the training thread.
        gap = 0.12 if (self._training or self._training_was_active) else 0.0
        if (self._render_dirty and not self._rendering
                and (time.monotonic() - self._last_render_t) >= gap):
            self._rendering = True
            asyncio.create_task(self._render_async())

    def _do_end_of_training(self):
        """Cheap end-of-training bookkeeping; rendering is deferred to the
        next _tick (via _render_dirty) so this stays off the event loop."""
        final = self._train_params_target
        if final is not None:
            self._suppress_redraw = True
            try:
                self._set_params(final)
            finally:
                self._suppress_redraw = False
            self._refresh_value_labels()
            self._last_trained_params    = np.asarray(final, dtype=float).copy()
            self._slider_snapshot_at_end = self._get_params()
        self._train_params_target = None
        if self._train_z_batch is not None:
            self._samples_z = self._train_z_batch.copy()
        self._train_z_batch  = None
        self._frozen_static  = None
        # Set the trained-params override BEFORE requesting the render: with
        # _train_params_live cleared, _eval_transform falls back to
        # _last_trained_params (== final), so the final frame shows the exact
        # converged state with no slider-clipping jump.
        self._train_params_live  = None
        self._use_trained_params = True
        self._render_dirty = True
        n = self._n_epochs_total
        if hasattr(self, '_prog_bar'):   self._prog_bar.value = 1.0
        if hasattr(self, '_prog_epoch'): self._prog_epoch.text = f"{n} / {n}"
        if hasattr(self, '_prog_label'):
            avg_us = getattr(self, "_avg_epoch_us", 0.0)
            if avg_us >= 1000.0:
                avg_str = f"{avg_us/1000.0:.1f} ms/epoch"
            else:
                avg_str = f"{avg_us:.0f} μs/epoch"
            # success (green) or error (red); _set_prog_kind picks the
            # theme-aware colour and remembers it so a dark<->light toggle
            # keeps the right colour.
            if self._train_status is not None:
                msg, _ = self._train_status     # message text; colour by kind
                self._train_status = None
                self._set_prog_kind("error")
            else:
                msg = f"Done — {n} epochs  ·  avg {avg_str}"
                self._set_prog_kind("done")
            self._prog_label.text = msg

    def _train_loop(self):
        try:
            n_epochs = max(1, int(self._n_epochs_val))
        except (ValueError, AttributeError):
            n_epochs = 200
        self._n_epochs_total = n_epochs
        try:
            lr = float(self._lr_val)
        except (ValueError, AttributeError):
            lr = 0.01
        μ = self.vals["mu"]; σ = self.vals["sg"]; dist = self.dist_val
        try:
            n_batch = max(10, int(self._n_batch_val))
        except (ValueError, AttributeError):
            n_batch = 1000
        train_mode = self._train_mode_val
        resample_each_epoch = bool(self._resample_val)
        rng = np.random.default_rng(42)
        z_batch = sample_latent(n_batch, μ, σ, dist, rng)
        self._train_z_batch = z_batch
        x_data = self._data_x if self._data_x is not None else None
        if train_mode == "Example-based" and (x_data is None or len(x_data) == 0):
            self._train_status = ("No data — click 'Generate data' first", "#B71C1C")
            self._training = False; return
        latent_tuple = (μ, σ, dist)
        try:
            stride = max(1, int(self._stride_val))
        except (ValueError, AttributeError):
            stride = 10
        try:
            delay_s = max(0.0, float(self._delay_val)) / 1000.0
        except (ValueError, AttributeError):
            delay_s = 0.05
        params = (self._train_starting_params.copy()
                  if self._train_starting_params is not None
                  else self._get_params())
        optimizer = self._optimizer_val
        target = (self.vals["kT"], self.vals["u1"], self.vals["u2"],
                  self.vals["u3"], self.vals["u4"])
        transform_type = self.transform_val
        if transform_type == "Polynomial":             K_val = None
        elif transform_type == "Rational-quadratic spline": K_val = self.K_rqs_val
        else:                                          K_val = self.K_val
        m = np.zeros_like(params); v = np.zeros_like(params)
        b1, b2, eps_opt = 0.9, 0.999, 1e-8
        self._epoch_time_us = 0.0; ema_alpha = 0.1
        epoch_us_sum = 0.0     # running total of per-epoch compute time
        self._avg_epoch_us = 0.0
        # Track the best (lowest-loss) parameters seen, so the curve shown
        # after training is never worse than what was displayed during it.
        best_loss = np.inf
        best_params = params.copy()

        for t in range(1, n_epochs + 1):
            if not self._training:
                self._train_status = ("Stopped", "#B71C1C"); return
            t_epoch_start = time.perf_counter()
            if train_mode == "Energy-based":
                if resample_each_epoch:
                    z_batch = sample_latent(n_batch, μ, σ, dist, rng)
                    self._train_z_batch = z_batch
                loss, ener_term, entr_term = self._compute_loss(
                    params, z_batch, target=target,
                    transform_type=transform_type, K=K_val,
                    return_components=True)
                if not np.isfinite(loss):
                    self._train_status = ("Loss diverged — reduce lr", "#B71C1C")
                    self._training = False; return
                if transform_type == "Rational-quadratic spline":
                    grad = self._gradient_fd(params, z_batch, target=target,
                                             transform_type=transform_type, K=K_val)
                else:
                    grad = self._gradient_analytic(params, z_batch, target=target,
                                                   transform_type=transform_type,
                                                   K=K_val)
            else:
                if transform_type == "Rational-quadratic spline":
                    loss, ener_term, entr_term, grad = \
                        self._loss_and_grad_fd_example(
                            params, x_data, latent=latent_tuple,
                            transform_type=transform_type, K=K_val)
                else:
                    loss, ener_term, entr_term, grad = \
                        self._loss_and_grad_example(
                            params, x_data, latent=latent_tuple,
                            transform_type=transform_type, K=K_val)
                if not np.isfinite(loss):
                    self._train_status = ("Loss diverged — reduce lr", "#B71C1C")
                    self._training = False; return

            self._loss_history.append(loss)
            self._loss_energy_history.append(ener_term)
            self._loss_entropy_history.append(entr_term)

            # `loss` was evaluated at the current `params` (before this epoch's
            # update), so remember those params if this is the best loss so far.
            if loss < best_loss:
                best_loss = loss
                best_params = params.copy()

            if optimizer == "Adam":
                m  = b1*m + (1-b1)*grad; v  = b2*v + (1-b2)*grad**2
                mh = m/(1-b1**t);        vh = v/(1-b2**t)
                params = params - lr*mh/(np.sqrt(vh)+eps_opt)
            elif optimizer == "SGD":
                params = params - lr*grad
            elif optimizer == "SGD+momentum":
                m = b1*m + grad; params = params - lr*m
            elif optimizer == "RMSprop":
                v = b2*v + (1-b2)*grad**2
                params = params - lr*grad/(np.sqrt(v)+eps_opt)

            params = self._clip_params(params, transform_type=transform_type, K=K_val)
            dt_us = (time.perf_counter() - t_epoch_start)*1e6
            epoch_us_sum += dt_us
            if t == 1: self._epoch_time_us = dt_us
            else: self._epoch_time_us = (1-ema_alpha)*self._epoch_time_us + ema_alpha*dt_us

            if t % stride == 0 or t == n_epochs:
                self._train_params_pending = params.copy()
                self._training_epoch = t
                if delay_s > 0.0:
                    time.sleep(delay_s)

        # Average (pure-compute) time per epoch over the whole run.
        self._avg_epoch_us = epoch_us_sum / max(n_epochs, 1)
        # Finish on the best-loss parameters (not the last, possibly-overshot
        # iterate) so the final figure is at least as good as any frame shown.
        self._train_params_pending = best_params.copy()
        self._training_epoch = n_epochs
        self._training = False

# ── Entry point ───────────────────────────────────────────────────────────────

NATIVE = False   # set in the main block below; read by NormFlowApp._exit_app


@ui.page("/")
def _index():
    # Build a fresh, fully-isolated NormFlowApp for every browser/tab.  Each
    # visitor gets their own figure, sliders, state and training thread, so no
    # user can affect what another sees.
    NormFlowApp()


if __name__ in {"__main__", "__mp_main__"}:
    # Register the image/static route once (not per page).
    nicegui_app.add_static_files(
        "/static", os.path.dirname(os.path.abspath(__file__)))
    # Native desktop window (needs pywebview) is single-user/local.  If it
    # isn't installed we serve over HTTP, which is also the multi-user mode.
    try:
        import webview  # noqa: F401
        NATIVE = True
    except Exception:
        NATIVE = False
        print("pywebview not found — serving in the browser.\n"
              "Open http://localhost:8080  (or the machine's address for "
              "remote users).")
    ui.run(
        title="Boltzmannator",
        native=NATIVE,
        window_size=(1600, 960) if NATIVE else None,
        host="0.0.0.0",          # listen on all interfaces (LAN/public)
        port=8080,
        reload=False,
        storage_secret="boltzmannator-secret",  # enables per-client storage
    )
