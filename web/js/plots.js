"use strict";
/* plots.js — a small canvas plotting toolkit for the Boltzmannator.
   Provides matplotlib-like axes (ticks, grid, spines, labels with sub/
   superscripts, lines, fills, histograms, legends) on an HTML canvas.
   All drawing happens in a fixed logical coordinate system (the canvas
   context is pre-scaled), so sizes below are "matplotlib pixels". */

const FS = 100 / 72;          // matplotlib pt -> px at 100 dpi (figsize scale)

/* ── Nice tick locations (matplotlib MaxNLocator-ish) ───────────────────── */

function niceStep(rough) {
    const mag = Math.pow(10, Math.floor(Math.log10(rough)));
    const r = rough / mag;
    if (r <= 1.0) return mag;
    if (r <= 2.0) return 2 * mag;
    if (r <= 2.5) return 2.5 * mag;
    if (r <= 5.0) return 5 * mag;
    return 10 * mag;
}

function niceTicks(lo, hi, maxTicks = 5, integer = false) {
    if (!(hi > lo)) return [lo];
    let step = niceStep((hi - lo) / Math.max(maxTicks, 1));
    if (integer) step = Math.max(1, Math.round(step));
    const first = Math.ceil(lo / step - 1e-9) * step;
    const out = [];
    for (let v = first; v <= hi + 1e-9 * step; v += step)
        out.push(Math.abs(v) < 1e-12 ? 0 : v);
    return out;
}

function tickLabel(v, step) {
    if (Number.isInteger(step) && Number.isInteger(v)) return String(v);
    const dec = Math.max(0, Math.min(6, -Math.floor(Math.log10(step)) + 1));
    let s = v.toFixed(dec);
    if (s.includes(".")) s = s.replace(/0+$/, "").replace(/\.$/, "");
    if (s === "-0") s = "0";
    return s;
}

/* ── Rich text: array of segments; each 'str' or [str, 'sub'|'sup'] ─────── */

function segList(spec) {
    if (typeof spec === "string") return [spec];
    return spec;
}

function fontStr(size, { italic = false, bold = false } = {}) {
    return `${italic ? "italic " : ""}${bold ? "600 " : ""}${size}px ` +
           `-apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif`;
}

function measureRich(ctx, spec, size, opts = {}) {
    let w = 0;
    for (const seg of segList(spec)) {
        const [t, mode] = typeof seg === "string" ? [seg, ""] : seg;
        const s = mode ? size * 0.72 : size;
        ctx.font = fontStr(s, opts);
        w += ctx.measureText(t).width;
    }
    return w;
}

/* draw rich text; (x,y) is the anchor; align: left|center|right;
   baseline: alphabetic|middle|top — applied to the main-size text */
function drawRich(ctx, spec, x, y, {
    size = 12 * FS, color = "#000", align = "left", baseline = "alphabetic",
    italic = false, bold = false, rotate = 0 } = {}) {
    ctx.save();
    ctx.translate(x, y);
    if (rotate) ctx.rotate(rotate);
    const total = measureRich(ctx, spec, size, { italic, bold });
    let cx = align === "center" ? -total / 2 : (align === "right" ? -total : 0);
    let cy = 0;
    if (baseline === "middle") cy = size * 0.36;
    else if (baseline === "top") cy = size * 0.8;
    ctx.fillStyle = color;
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    for (const seg of segList(spec)) {
        const [t, mode] = typeof seg === "string" ? [seg, ""] : seg;
        const s = mode ? size * 0.72 : size;
        const dy = mode === "sub" ? size * 0.18 : (mode === "sup" ? -size * 0.42 : 0);
        ctx.font = fontStr(s, { italic, bold });
        ctx.fillText(t, cx, cy + dy);
        cx += ctx.measureText(t).width;
    }
    ctx.restore();
}

/* rounded-rect path helper */
function roundRectPath(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
}

/* ── Axes ───────────────────────────────────────────────────────────────── */

class Axes {
    /* rect = {x, y, w, h} in logical px; theme = {fg, bg, grid} */
    constructor(ctx, rect, theme) {
        this.ctx = ctx;
        this.rect = rect;
        this.theme = theme;
        this.xlim = [0, 1];
        this.ylim = [0, 1];
        this.xticks = [];
        this.yticks = [];
        this.xstep = 1;
        this.ystep = 1;
    }

    setXlim(lo, hi) { if (hi <= lo) hi = lo + 1; this.xlim = [lo, hi]; }
    setYlim(lo, hi) { if (hi <= lo) hi = lo + 1; this.ylim = [lo, hi]; }

    px(v) {
        return this.rect.x
            + (v - this.xlim[0]) / (this.xlim[1] - this.xlim[0]) * this.rect.w;
    }
    py(v) {
        return this.rect.y + this.rect.h
            - (v - this.ylim[0]) / (this.ylim[1] - this.ylim[0]) * this.rect.h;
    }

    computeTicks({ nx = 5, ny = 4, integerX = false } = {}) {
        this.xticks = niceTicks(this.xlim[0], this.xlim[1], nx, integerX);
        this.yticks = niceTicks(this.ylim[0], this.ylim[1], ny);
        this.xstep = this.xticks.length > 1
            ? this.xticks[1] - this.xticks[0] : 1;
        this.ystep = this.yticks.length > 1
            ? this.yticks[1] - this.yticks[0] : 1;
    }

    clip(fn) {
        const { ctx, rect } = this;
        ctx.save();
        ctx.beginPath();
        ctx.rect(rect.x, rect.y, rect.w, rect.h);
        ctx.clip();
        fn();
        ctx.restore();
    }

    grid({ alpha = 0.22 } = {}) {
        const { ctx, rect } = this;
        ctx.save();
        ctx.globalAlpha = alpha;
        ctx.strokeStyle = this.theme.grid;
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);
        for (const t of this.xticks) {
            const x = this.px(t);
            ctx.beginPath();
            ctx.moveTo(x, rect.y); ctx.lineTo(x, rect.y + rect.h);
            ctx.stroke();
        }
        for (const t of this.yticks) {
            const y = this.py(t);
            ctx.beginPath();
            ctx.moveTo(rect.x, y); ctx.lineTo(rect.x + rect.w, y);
            ctx.stroke();
        }
        ctx.restore();
    }

    /* spines: {top, right, bottom, left} booleans;
       ticksX/ticksY: draw tick marks + labels */
    frame({ spines = {}, ticksX = true, ticksY = true,
            labelX = true, labelY = true, fontSize = 10 * FS,
            tickLen = 5 } = {}) {
        const { ctx, rect } = this;
        const fg = this.theme.fg;
        ctx.save();
        ctx.strokeStyle = fg;
        ctx.lineWidth = 1.1;
        ctx.setLineDash([]);
        const L = rect.x, R = rect.x + rect.w, T = rect.y, B = rect.y + rect.h;
        const seg = (x0, y0, x1, y1) => {
            ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
        };
        if (spines.bottom !== false) seg(L, B, R, B);
        if (spines.left   !== false) seg(L, T, L, B);
        if (spines.top    === true)  seg(L, T, R, T);
        if (spines.right  === true)  seg(R, T, R, B);

        if (ticksX) {
            for (const t of this.xticks) {
                const x = this.px(t);
                if (x < rect.x - 0.5 || x > rect.x + rect.w + 0.5) continue;
                seg(x, B, x, B + tickLen);
                if (labelX)
                    drawRich(ctx, tickLabel(t, this.xstep), x, B + tickLen + fontSize,
                             { size: fontSize, color: fg, align: "center" });
            }
        }
        if (ticksY) {
            for (const t of this.yticks) {
                const y = this.py(t);
                if (y < rect.y - 0.5 || y > rect.y + rect.h + 0.5) continue;
                seg(L, y, L - tickLen, y);
                if (labelY)
                    drawRich(ctx, tickLabel(t, this.ystep), L - tickLen - 3, y,
                             { size: fontSize, color: fg, align: "right",
                               baseline: "middle" });
            }
        }
        ctx.restore();
    }

    line(xs, ys, { color = "#000", lw = 2, dash = null, alpha = 1,
                   clip = true } = {}) {
        const draw = () => {
            const { ctx } = this;
            ctx.save();
            ctx.globalAlpha = alpha;
            ctx.strokeStyle = color;
            ctx.lineWidth = lw;
            ctx.lineJoin = "round";
            ctx.setLineDash(dash || []);
            ctx.beginPath();
            let started = false;
            for (let i = 0; i < xs.length; i++) {
                const xv = xs[i], yv = ys[i];
                if (!Number.isFinite(xv) || !Number.isFinite(yv)) {
                    started = false; continue;
                }
                const x = this.px(xv), y = this.py(yv);
                if (!started) { ctx.moveTo(x, y); started = true; }
                else ctx.lineTo(x, y);
            }
            ctx.stroke();
            ctx.restore();
        };
        if (clip) this.clip(draw); else draw();
    }

    /* filled polygon in data coordinates */
    polygon(pts, { color = "#000", alpha = 1, clip = true } = {}) {
        const draw = () => {
            const { ctx } = this;
            ctx.save();
            ctx.globalAlpha = alpha;
            ctx.fillStyle = color;
            ctx.beginPath();
            for (let i = 0; i < pts.length; i++) {
                const x = this.px(pts[i][0]), y = this.py(pts[i][1]);
                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }
            ctx.closePath();
            ctx.fill();
            ctx.restore();
        };
        if (clip) this.clip(draw); else draw();
    }

    /* fill between the curve and y = 0 (fill_between) */
    fillUnder(xs, ys, opts) {
        const pts = [];
        pts.push([xs[0], 0]);
        for (let i = 0; i < xs.length; i++) pts.push([xs[i], ys[i]]);
        pts.push([xs[xs.length - 1], 0]);
        this.polygon(pts, opts);
    }

    /* fill between the curve and x = 0 (fill_betweenx): xs = values,
       ys = positions */
    fillLeft(xs, ys, opts) {
        const pts = [];
        pts.push([0, ys[0]]);
        for (let i = 0; i < xs.length; i++) pts.push([xs[i], ys[i]]);
        pts.push([0, ys[ys.length - 1]]);
        this.polygon(pts, opts);
    }

    /* density histogram; returns max density */
    hist(data, { bins = 40, color = "#000", alpha = 0.4 } = {}) {
        if (!data || data.length === 0) return 0;
        let lo = Infinity, hi = -Infinity;
        for (const v of data) { if (v < lo) lo = v; if (v > hi) hi = v; }
        if (!(hi > lo)) { lo -= 0.5; hi += 0.5; }
        const counts = new Float64Array(bins);
        const w = (hi - lo) / bins;
        for (const v of data) {
            let b = Math.floor((v - lo) / w);
            if (b >= bins) b = bins - 1;
            if (b < 0) b = 0;
            counts[b]++;
        }
        const n = data.length;
        let maxD = 0;
        this.clip(() => {
            const { ctx } = this;
            ctx.save();
            ctx.globalAlpha = alpha;
            ctx.fillStyle = color;
            const y0 = this.py(0);
            for (let b = 0; b < bins; b++) {
                const d = counts[b] / (n * w);
                if (d > maxD) maxD = d;
                if (d <= 0) continue;
                const x0 = this.px(lo + b * w), x1 = this.px(lo + (b + 1) * w);
                const y1 = this.py(d);
                ctx.fillRect(x0, Math.min(y0, y1),
                             Math.max(1, x1 - x0 - 0.5), Math.abs(y0 - y1));
            }
            ctx.restore();
        });
        return maxD;
    }

    histMaxDensity(data, bins = 40) {
        if (!data || data.length === 0) return 0;
        let lo = Infinity, hi = -Infinity;
        for (const v of data) { if (v < lo) lo = v; if (v > hi) hi = v; }
        if (!(hi > lo)) return 0;
        const counts = new Float64Array(bins);
        const w = (hi - lo) / bins;
        for (const v of data) {
            let b = Math.floor((v - lo) / w);
            if (b >= bins) b = bins - 1;
            if (b < 0) b = 0;
            counts[b]++;
        }
        let m = 0;
        for (let b = 0; b < bins; b++)
            m = Math.max(m, counts[b] / (data.length * w));
        return m;
    }

    hline(y, { color = "#999", lw = 1, dash = null, alpha = 1 } = {}) {
        this.line([this.xlim[0], this.xlim[1]], [y, y],
                  { color, lw, dash, alpha });
    }
    vline(x, { color = "#999", lw = 1, dash = null, alpha = 1 } = {}) {
        this.line([x, x], [this.ylim[0], this.ylim[1]],
                  { color, lw, dash, alpha });
    }

    marker(x, y, { r = 6, color = "#FF6D00", edge = "white", edgeW = 1.7 } = {}) {
        const { ctx } = this;
        ctx.save();
        ctx.beginPath();
        ctx.arc(this.px(x), this.py(y), r, 0, 2 * Math.PI);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.lineWidth = edgeW;
        ctx.strokeStyle = edge;
        ctx.stroke();
        ctx.restore();
    }

    xlabel(spec, { size = 12 * FS, offset = null } = {}) {
        const off = offset != null ? offset : 34;
        drawRich(this.ctx, spec,
                 this.rect.x + this.rect.w / 2, this.rect.y + this.rect.h + off,
                 { size, color: this.theme.fg, align: "center", italic: true });
    }

    ylabel(spec, { size = 12 * FS, offset = null } = {}) {
        const off = offset != null ? offset : 40;
        drawRich(this.ctx, spec,
                 this.rect.x - off, this.rect.y + this.rect.h / 2,
                 { size, color: this.theme.fg, align: "center",
                   italic: true, rotate: -Math.PI / 2 });
    }

    /* right-hand-side ylabel (for the twin axis) */
    ylabelRight(spec, { size = 10 * FS, offset = 34, color = null } = {}) {
        drawRich(this.ctx, spec,
                 this.rect.x + this.rect.w + offset,
                 this.rect.y + this.rect.h / 2,
                 { size, color: color || this.theme.fg, align: "center",
                   italic: true, rotate: Math.PI / 2 });
    }

    title(spec, { size = 12 * FS, pad = 6 } = {}) {
        drawRich(this.ctx, spec,
                 this.rect.x + this.rect.w / 2, this.rect.y - pad,
                 { size, color: this.theme.fg, align: "center" });
    }

    /* text at axes-fraction coordinates, optional rounded bbox */
    textAxes(fx, fy, spec, { size = 9 * FS, color = "#888", ha = "center",
                             va = "middle", bold = false, italic = false,
                             bbox = null, lineHeight = 1.35 } = {}) {
        const { ctx, rect } = this;
        const lines = (typeof spec === "string" ? spec.split("\n") : [spec]);
        const x = rect.x + fx * rect.w;
        const y = rect.y + (1 - fy) * rect.h;
        const lh = size * lineHeight;
        const totalH = lh * lines.length;
        let widest = 0;
        for (const ln of lines)
            widest = Math.max(widest, measureRich(ctx, ln, size, { bold, italic }));
        let topY;
        if (va === "top") topY = y;
        else if (va === "bottom") topY = y - totalH;
        else topY = y - totalH / 2;
        let leftX;
        if (ha === "left") leftX = x;
        else if (ha === "right") leftX = x - widest;
        else leftX = x - widest / 2;
        if (bbox) {
            ctx.save();
            const padB = bbox.pad != null ? bbox.pad : 7;
            roundRectPath(ctx, leftX - padB, topY - padB,
                          widest + 2 * padB, totalH + 2 * padB, 6);
            ctx.globalAlpha = bbox.alpha != null ? bbox.alpha : 1;
            ctx.fillStyle = bbox.fc || "#fff";
            ctx.fill();
            if (bbox.ec) {
                ctx.lineWidth = bbox.lw || 1.2;
                ctx.strokeStyle = bbox.ec;
                ctx.stroke();
            }
            ctx.restore();
        }
        for (let i = 0; i < lines.length; i++) {
            drawRich(ctx, lines[i], leftX + widest / 2,
                     topY + (i + 0.78) * lh,
                     { size, color, align: "center", bold, italic });
        }
    }

    /* legend row centred above the axes: entries = {color, label, lw, dash} */
    legendAbove(entries, { size = 11 * FS, y = null } = {}) {
        const { ctx, rect } = this;
        const yPos = y != null ? y : rect.y - 8;
        const sample = 18, gap = 5, colGap = 14;
        let total = 0;
        for (const e of entries)
            total += sample + gap + measureRich(ctx, e.label, size) + colGap;
        total -= colGap;
        let cx = rect.x + rect.w / 2 - total / 2;
        for (const e of entries) {
            ctx.save();
            ctx.strokeStyle = e.color;
            ctx.lineWidth = e.lw || 2;
            ctx.setLineDash(e.dash || []);
            ctx.beginPath();
            ctx.moveTo(cx, yPos - size * 0.32);
            ctx.lineTo(cx + sample, yPos - size * 0.32);
            ctx.stroke();
            ctx.restore();
            cx += sample + gap;
            drawRich(ctx, e.label, cx, yPos,
                     { size, color: this.theme.fg, align: "left" });
            cx += measureRich(ctx, e.label, size) + colGap;
        }
    }
}

/* ── GridSpec-like cell splitting ────────────────────────────────────────
   Splits [lo, hi] into cells with the given ratios; the gap between cells
   is `space` × (average cell size), like matplotlib's wspace/hspace.     */
function splitCells(lo, hi, ratios, space) {
    const n = ratios.length;
    const R = ratios.reduce((a, b) => a + b, 0);
    const total = hi - lo;
    const c = total / (R * (1 + (n - 1) * space / n));
    const sep = space * (c * R / n);
    const out = [];
    let cur = lo;
    for (let i = 0; i < n; i++) {
        const w = ratios[i] * c;
        out.push([cur, cur + w]);
        cur += w + sep;
    }
    return out;
}
