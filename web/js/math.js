"use strict";
/* math.js — Boltzmannator numerics.
   Faithful port of the pure-numpy helpers and training math from
   Boltzmannator.py.  Everything works on plain Float64Arrays. */

const SQRT2PI = Math.sqrt(2 * Math.PI);
const SQRT2   = Math.sqrt(2);
const SQRT3   = Math.sqrt(3);

/* ── Random numbers ─────────────────────────────────────────────────────── */

function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
        a |= 0; a = (a + 0x6D2B79F5) | 0;
        let t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

/* rng object: seeded (reproducible) if seed given, else Math.random */
function makeRng(seed = null) {
    const u = (seed === null) ? Math.random : mulberry32(seed);
    let spare = null;                         // Box–Muller spare
    return {
        random: () => u(),
        normal(mu = 0, sigma = 1) {
            if (spare !== null) { const v = spare; spare = null; return mu + sigma * v; }
            let a, b, r;
            do { a = 2 * u() - 1; b = 2 * u() - 1; r = a * a + b * b; }
            while (r >= 1 || r === 0);
            const m = Math.sqrt(-2 * Math.log(r) / r);
            spare = b * m;
            return mu + sigma * a * m;
        },
        uniform: (a, b) => a + (b - a) * u(),
        laplace(mu, s) {
            const v = u() - 0.5;
            return mu - s * Math.sign(v) * Math.log(1 - 2 * Math.abs(v));
        },
        cauchy() { return Math.tan(Math.PI * (u() - 0.5)); },
    };
}

/* ── Small array helpers (numpy stand-ins) ──────────────────────────────── */

function linspace(a, b, n) {
    const out = new Float64Array(n);
    if (n === 1) { out[0] = a; return out; }
    const d = (b - a) / (n - 1);
    for (let i = 0; i < n; i++) out[i] = a + d * i;
    return out;
}

function trapz(y, x) {
    let s = 0;
    for (let i = 1; i < y.length; i++)
        s += 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]);
    return s;
}

/* np.interp: piecewise-linear, clamped at the ends; xp must be increasing */
function interp1(xq, xp, fp) {
    const n = xp.length;
    if (xq <= xp[0]) return fp[0];
    if (xq >= xp[n - 1]) return fp[n - 1];
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) {
        const mid = (lo + hi) >> 1;
        if (xp[mid] <= xq) lo = mid; else hi = mid;
    }
    const t = (xq - xp[lo]) / (xp[hi] - xp[lo]);
    return fp[lo] + t * (fp[hi] - fp[lo]);
}

function interpArr(xqArr, xp, fp) {
    const out = new Float64Array(xqArr.length);
    for (let i = 0; i < xqArr.length; i++) out[i] = interp1(xqArr[i], xp, fp);
    return out;
}

/* np.searchsorted(arr, v, side='left') for a sorted array */
function searchsortedLeft(arr, v) {
    let lo = 0, hi = arr.length;
    while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (arr[mid] < v) lo = mid + 1; else hi = mid;
    }
    return lo;
}

function percentile(values, q) {           // linear interpolation, like numpy
    const a = Array.from(values).sort((x, y) => x - y);
    if (a.length === 0) return NaN;
    const idx = (q / 100) * (a.length - 1);
    const lo = Math.floor(idx), hi = Math.ceil(idx);
    if (lo === hi) return a[lo];
    return a[lo] + (a[hi] - a[lo]) * (idx - lo);
}

function arrMin(a) { let m = Infinity;  for (const v of a) if (v < m) m = v; return m; }
function arrMax(a) { let m = -Infinity; for (const v of a) if (v > m) m = v; return m; }

function meanFinite(a) {
    let s = 0, n = 0;
    for (const v of a) if (Number.isFinite(v)) { s += v; n++; }
    return n > 0 ? s / n : NaN;
}

function stdSample(a) {                    // ddof = 1, like data.std(ddof=1)
    const n = a.length;
    if (n < 2) return 0;
    let m = 0; for (const v of a) m += v; m /= n;
    let s = 0; for (const v of a) s += (v - m) * (v - m);
    return Math.sqrt(s / (n - 1));
}

const clampNum = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

/* ── Latent distributions ───────────────────────────────────────────────── */

function gaussianPdf1(x, mu, sigma) {
    const t = (x - mu) / sigma;
    return Math.exp(-0.5 * t * t) / (sigma * SQRT2PI);
}

function latentPdf(z, mu, sigma, name) {
    const n = z.length, out = new Float64Array(n);
    if (name === "Gaussian") {
        for (let i = 0; i < n; i++) out[i] = gaussianPdf1(z[i], mu, sigma);
    } else if (name === "Uniform") {
        const a = mu - sigma * SQRT3, b = mu + sigma * SQRT3, h = 1.0 / (b - a);
        for (let i = 0; i < n; i++) out[i] = (z[i] >= a && z[i] <= b) ? h : 0.0;
    } else if (name === "Laplace") {
        const s = sigma / SQRT2;
        for (let i = 0; i < n; i++)
            out[i] = Math.exp(-Math.abs(z[i] - mu) / s) / (2 * s);
    } else if (name === "Cauchy") {
        for (let i = 0; i < n; i++) {
            const t = (z[i] - mu) / sigma;
            out[i] = 1.0 / (Math.PI * sigma * (1 + t * t));
        }
    } else if (name === "Bimodal") {
        for (let i = 0; i < n; i++)
            out[i] = 0.5 * gaussianPdf1(z[i], -mu, sigma)
                   + 0.5 * gaussianPdf1(z[i],  mu, sigma);
    }
    return out;
}

function sampleLatent(n, mu, sigma, name, rng) {
    const out = new Float64Array(n);
    if (name === "Gaussian") {
        for (let i = 0; i < n; i++) out[i] = rng.normal(mu, sigma);
    } else if (name === "Uniform") {
        const a = mu - sigma * SQRT3, b = mu + sigma * SQRT3;
        for (let i = 0; i < n; i++) out[i] = rng.uniform(a, b);
    } else if (name === "Laplace") {
        for (let i = 0; i < n; i++) out[i] = rng.laplace(mu, sigma / SQRT2);
    } else if (name === "Cauchy") {
        for (let i = 0; i < n; i++) out[i] = mu + sigma * rng.cauchy();
    } else if (name === "Bimodal") {
        for (let i = 0; i < n; i++)
            out[i] = (rng.random() < 0.5) ? rng.normal(-mu, sigma)
                                          : rng.normal( mu, sigma);
    }
    return out;
}

function logLatentPdf(z, mu, sigma, name) {
    const n = z.length, out = new Float64Array(n);
    if (name === "Gaussian") {
        const c = -Math.log(sigma) - 0.5 * Math.log(2 * Math.PI);
        for (let i = 0; i < n; i++) {
            const t = (z[i] - mu) / sigma;
            out[i] = -0.5 * t * t + c;
        }
    } else if (name === "Uniform") {
        const a = mu - sigma * SQRT3, b = mu + sigma * SQRT3, c = -Math.log(b - a);
        for (let i = 0; i < n; i++)
            out[i] = (z[i] >= a && z[i] <= b) ? c : -Infinity;
    } else if (name === "Laplace") {
        const s = sigma / SQRT2, c = -Math.log(2 * s);
        for (let i = 0; i < n; i++) out[i] = -Math.abs(z[i] - mu) / s + c;
    } else if (name === "Cauchy") {
        const c = -(Math.log(Math.PI) + Math.log(sigma));
        for (let i = 0; i < n; i++) {
            const t = (z[i] - mu) / sigma;
            out[i] = c - Math.log1p(t * t);
        }
    } else if (name === "Bimodal") {
        const c = -Math.log(sigma) - 0.5 * Math.log(2 * Math.PI) + Math.log(0.5);
        for (let i = 0; i < n; i++) {
            const t1 = (z[i] + mu) / sigma, t2 = (z[i] - mu) / sigma;
            const lp1 = c - 0.5 * t1 * t1, lp2 = c - 0.5 * t2 * t2;
            const m = Math.max(lp1, lp2);       // logaddexp
            out[i] = m + Math.log(Math.exp(lp1 - m) + Math.exp(lp2 - m));
        }
    } else {
        out.fill(-Infinity);
    }
    return out;
}

/* Gaussian KDE with Scott's bandwidth rule; returns evaluator(xq)->array */
function gaussianKde(data) {
    const bw = Math.pow(data.length, -1 / 5) * stdSample(data);
    return function (xq) {
        const out = new Float64Array(xq.length);
        const c = 1.0 / (data.length * bw * SQRT2PI);
        for (let i = 0; i < xq.length; i++) {
            let s = 0;
            for (let j = 0; j < data.length; j++) {
                const d = (xq[i] - data[j]) / bw;
                s += Math.exp(-0.5 * d * d);
            }
            out[i] = s * c;
        }
        return out;
    };
}

/* ── Rational-quadratic spline ──────────────────────────────────────────── */

function rqsToKnots(params, K) {
    const B = Math.max(params[0], 0.5), W = 2.0 * B;
    const softmax = (raw) => {
        let mx = -Infinity;
        for (const v of raw) if (v > mx) mx = v;
        const e = raw.map(v => Math.exp(v - mx));
        const s = e.reduce((a, b) => a + b, 0);
        return e.map(v => W * v / s);
    };
    const widths  = softmax(Array.from(params.slice(1, K + 1)));
    const heights = softmax(Array.from(params.slice(K + 1, 2 * K + 1)));
    const derivs  = Array.from(params.slice(2 * K + 1, 3 * K + 2))
                         .map(v => Math.exp(clampNum(v, -6.0, 6.0)));
    const xk = new Float64Array(K + 1), yk = new Float64Array(K + 1);
    xk[0] = -B; yk[0] = -B;
    for (let k = 0; k < K; k++) {
        xk[k + 1] = xk[k] + widths[k];
        yk[k + 1] = yk[k] + heights[k];
    }
    return { B, widths, heights, derivs, xk, yk };
}

function rqsForward(z, kn) {
    const { B, widths, heights, derivs, xk, yk } = kn;
    const K = widths.length, n = z.length;
    const x = new Float64Array(n), J = new Float64Array(n);
    const inner = xk.slice(1, K);            // x_knots[1:-1]
    for (let i = 0; i < n; i++) {
        const zi = z[i];
        if (zi <= -B || zi >= B) { x[i] = zi; J[i] = 1.0; continue; }
        const k = clampNum(searchsortedLeft(inner, zi), 0, K - 1);
        const dx = widths[k], dy = heights[k];
        const dk = derivs[k], dk1 = derivs[k + 1], sk = dy / dx;
        const xi = clampNum((zi - xk[k]) / dx, 0.0, 1.0), xi1 = 1.0 - xi;
        const g = dk1 + dk - 2.0 * sk;
        const den = sk + g * xi * xi1;
        x[i] = yk[k] + dy * (sk * xi * xi + dk * xi * xi1) / den;
        J[i] = sk * sk * (dk1 * xi * xi + 2.0 * sk * xi * xi1 + dk * xi1 * xi1)
               / (den * den);
    }
    return { x, J };
}

function rqsInverseAnalytic(xOut, kn) {
    const { B, widths, heights, derivs, xk, yk } = kn;
    const K = widths.length, n = xOut.length;
    const z = new Float64Array(n);
    const innerY = yk.slice(1, K);
    for (let i = 0; i < n; i++) {
        const v = xOut[i];
        if (v <= -B || v >= B) { z[i] = v; continue; }
        const k = clampNum(searchsortedLeft(innerY, v), 0, K - 1);
        const dx = widths[k], dy = heights[k];
        const dk = derivs[k], dk1 = derivs[k + 1], sk = dy / dx;
        const tau = v - yk[k], g = dk1 + dk - 2.0 * sk;
        const a = dy * (sk - dk) + tau * g;
        const b = dy * dk - tau * g;
        const c = -sk * tau;
        const disc = Math.max(b * b - 4.0 * a * c, 0.0), sq = Math.sqrt(disc);
        const aSafe = Math.abs(a) < 1e-9 ? Math.sign(a + 1e-18) * 1e-9 : a;
        let r1 = (-b + sq) / (2.0 * aSafe);
        let r2 = (-b - sq) / (2.0 * aSafe);
        const lin = Math.abs(b) > 1e-9 ? -c / b : 0.0;
        if (Math.abs(a) < 1e-9) { r1 = lin; r2 = lin; }
        const r1ok = (r1 >= -1e-4 && r1 <= 1.0 + 1e-4);
        const xi = clampNum(r1ok ? r1 : r2, 0.0, 1.0);
        z[i] = xk[k] + xi * dx;
    }
    return z;
}

/* ── Transformation evaluation ──────────────────────────────────────────── */

const T_POLY = "Polynomial";
const T_SLP  = "Single layer perceptron";
const T_RQS  = "Rational-quadratic spline";

/* returns {x, J}; params layout matches the Python version */
function evalTransformParams(z, params, ttype, K) {
    const n = z.length;
    if (ttype === T_POLY) {
        const [t0, t1, t2, t3] = params;
        const x = new Float64Array(n), J = new Float64Array(n);
        for (let i = 0; i < n; i++) {
            const zi = z[i];
            x[i] = t0 + t1 * zi + t2 * zi * zi + t3 * zi * zi * zi;
            J[i] = t1 + 2 * t2 * zi + 3 * t3 * zi * zi;
        }
        return { x, J };
    }
    if (ttype === T_RQS) {
        const kn = rqsToKnots(params, K);
        return rqsForward(z, kn);
    }
    // Single layer perceptron
    const off = params[0], slope = params[1];
    const x = new Float64Array(n), J = new Float64Array(n);
    for (let i = 0; i < n; i++) { x[i] = off + slope * z[i]; J[i] = slope; }
    for (let k = 0; k < K; k++) {
        const w = params[2 + 3 * k], c = params[3 + 3 * k];
        const s = Math.max(params[4 + 3 * k], 1e-6);
        for (let i = 0; i < n; i++) {
            const t = clampNum((z[i] - c) / s, -50, 50);
            const sg = 1.0 / (1.0 + Math.exp(-t));
            x[i] += w * sg;
            J[i] += (w / s) * sg * (1.0 - sg);
        }
    }
    return { x, J };
}

/* bisection inverse for poly / SLP (RQS has the analytic inverse) */
function invertTransformParams(xData, params, ttype, K, tol = 1e-7, maxIter = 30) {
    if (ttype === T_RQS) {
        const kn = rqsToKnots(params, K);
        return rqsInverseAnalytic(xData, kn);
    }
    const n = xData.length;
    const zLo = new Float64Array(n).fill(-20.0);
    const zHi = new Float64Array(n).fill( 20.0);
    const zMid = new Float64Array(n);
    for (let it = 0; it < maxIter; it++) {
        for (let i = 0; i < n; i++) zMid[i] = 0.5 * (zLo[i] + zHi[i]);
        const { x: xMid } = evalTransformParams(zMid, params, ttype, K);
        let maxGap = 0;
        for (let i = 0; i < n; i++) {
            if (xMid[i] < xData[i]) zLo[i] = zMid[i]; else zHi[i] = zMid[i];
            const g = zHi[i] - zLo[i];
            if (g > maxGap) maxGap = g;
        }
        if (maxGap < tol) break;
    }
    for (let i = 0; i < n; i++) zMid[i] = 0.5 * (zLo[i] + zHi[i]);
    return zMid;
}

/* ── Losses and gradients ───────────────────────────────────────────────── */

/* target = {kT, u1, u2, u3, u4} */
function potentialU(x, tg) {
    return tg.u1 * x + tg.u2 * x * x + tg.u3 * x * x * x + tg.u4 * x * x * x * x;
}

/* energy-based loss L = <U(f(z))/kT - log|J|> */
function computeLoss(params, zBatch, target, ttype, K, returnComponents = false) {
    const { x, J } = evalTransformParams(zBatch, params, ttype, K);
    const n = zBatch.length;
    let sT = 0, sE = 0, sS = 0, cnt = 0;
    for (let i = 0; i < n; i++) {
        const Jabs = Math.abs(J[i]);
        const logJ = Jabs > 1e-300 ? Math.log(Jabs + 1e-300) : -700.0;
        const energy  = potentialU(x[i], target) / target.kT;
        const entropy = -logJ;
        const v = energy + entropy;
        if (Number.isFinite(v)) { sT += v; sE += energy; sS += entropy; cnt++; }
    }
    const total = cnt > 0 ? sT / cnt : NaN;
    if (returnComponents)
        return [total, cnt > 0 ? sE / cnt : NaN, cnt > 0 ? sS / cnt : NaN];
    return total;
}

/* example-based loss (used by the FD path); latent = {mu, sigma, dist} */
function computeLossExample(params, xData, latent, ttype, K,
                            returnComponents = false) {
    const z = invertTransformParams(xData, params, ttype, K);
    const { J } = evalTransformParams(z, params, ttype, K);
    const logPz = logLatentPdf(z, latent.mu, latent.sigma, latent.dist);
    const n = z.length;
    let sT = 0, sE = 0, sS = 0, cnt = 0;
    for (let i = 0; i < n; i++) {
        const Jabs = Math.abs(J[i]);
        const logJ = Jabs > 1e-300 ? Math.log(Jabs + 1e-300) : -700.0;
        const v = -logPz[i] + logJ;
        if (Number.isFinite(v)) { sT += v; sE += -logPz[i]; sS += logJ; cnt++; }
    }
    const total = cnt > 0 ? sT / cnt : NaN;
    if (returnComponents)
        return [total, cnt > 0 ? sE / cnt : NaN, cnt > 0 ? sS / cnt : NaN];
    return total;
}

/* mean of v over samples where (ok && finite(v)) */
function maskedMean(v, ok) {
    let s = 0, n = 0;
    for (let i = 0; i < v.length; i++)
        if (ok[i] && Number.isFinite(v[i])) { s += v[i]; n++; }
    return n > 0 ? s / n : 0;
}

/* analytic gradient of the energy-based loss (poly and SLP) */
function gradientAnalytic(params, zBatch, target, ttype, K) {
    const z = zBatch, n = z.length;
    const grad = new Float64Array(params.length);
    const { kT, u1, u2, u3, u4 } = target;

    if (ttype === T_POLY) {
        const [t0, t1, t2, t3] = params;
        const x = new Float64Array(n), J = new Float64Array(n),
              dUdx = new Float64Array(n);
        const ok = new Uint8Array(n);
        for (let i = 0; i < n; i++) {
            const zi = z[i];
            x[i] = t0 + t1 * zi + t2 * zi * zi + t3 * zi * zi * zi;
            J[i] = t1 + 2 * t2 * zi + 3 * t3 * zi * zi;
            dUdx[i] = u1 + 2 * u2 * x[i] + 3 * u3 * x[i] * x[i]
                      + 4 * u4 * x[i] * x[i] * x[i];
            ok[i] = (Number.isFinite(J[i]) && Number.isFinite(dUdx[i])
                     && Math.abs(J[i]) > 1e-10) ? 1 : 0;
        }
        const v = new Float64Array(n);
        const derivs = [
            (zi) => [1, 0], (zi) => [zi, 1],
            (zi) => [zi * zi, 2 * zi], (zi) => [zi * zi * zi, 3 * zi * zi]];
        for (let p = 0; p < 4; p++) {
            for (let i = 0; i < n; i++) {
                const [dxdT, dJdT] = derivs[p](z[i]);
                v[i] = dUdx[i] * dxdT / kT - dJdT / J[i];
            }
            grad[p] = maskedMean(v, ok);
        }
        return grad;
    }

    // Single layer perceptron
    const off = params[0], slope = params[1];
    const x = new Float64Array(n), J = new Float64Array(n);
    for (let i = 0; i < n; i++) { x[i] = off + slope * z[i]; J[i] = slope; }
    const sgs = [];                                   // per-k sigmoid arrays
    for (let k = 0; k < K; k++) {
        const w = params[2 + 3 * k], c = params[3 + 3 * k];
        const s = Math.max(params[4 + 3 * k], 1e-6);
        const sg = new Float64Array(n);
        for (let i = 0; i < n; i++) {
            const t = clampNum((z[i] - c) / s, -50, 50);
            sg[i] = 1.0 / (1.0 + Math.exp(-t));
            x[i] += w * sg[i];
            J[i] += w * sg[i] * (1 - sg[i]) / s;
        }
        sgs.push({ w, c, s, sg });
    }
    const dUdx = new Float64Array(n), ok = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
        dUdx[i] = u1 + 2 * u2 * x[i] + 3 * u3 * x[i] * x[i]
                  + 4 * u4 * x[i] * x[i] * x[i];
        ok[i] = (Number.isFinite(J[i]) && Number.isFinite(dUdx[i])
                 && Math.abs(J[i]) > 1e-10) ? 1 : 0;
    }
    const v = new Float64Array(n);
    for (let i = 0; i < n; i++) v[i] = dUdx[i] / kT;
    grad[0] = maskedMean(v, ok);
    for (let i = 0; i < n; i++) v[i] = dUdx[i] * z[i] / kT - 1.0 / J[i];
    grad[1] = maskedMean(v, ok);
    for (let k = 0; k < K; k++) {
        const { w, c, s, sg } = sgs[k];
        for (let i = 0; i < n; i++) {
            const dsg = sg[i] * (1 - sg[i]);
            v[i] = dUdx[i] * sg[i] / kT - dsg / (s * J[i]);
        }
        grad[2 + 3 * k] = maskedMean(v, ok);
        for (let i = 0; i < n; i++) {
            const dsg = sg[i] * (1 - sg[i]);
            const dxDc = -w * dsg / s;
            const dJDc = -w * dsg * (1 - 2 * sg[i]) / (s * s);
            v[i] = dUdx[i] * dxDc / kT - dJDc / J[i];
        }
        grad[3 + 3 * k] = maskedMean(v, ok);
        for (let i = 0; i < n; i++) {
            const dsg = sg[i] * (1 - sg[i]);
            const zc = z[i] - c;
            const dxDs = -w * dsg * zc / (s * s);
            const dJDs = -w * dsg * ((1 - 2 * sg[i]) * zc / s + 1.0) / (s * s);
            v[i] = dUdx[i] * dxDs / kT - dJDs / J[i];
        }
        grad[4 + 3 * k] = maskedMean(v, ok);
    }
    return grad;
}

/* finite-difference gradient of the energy-based loss (used for RQS) */
function gradientFd(params, zBatch, target, ttype, K, eps = 1e-4) {
    const grad = new Float64Array(params.length);
    for (let i = 0; i < params.length; i++) {
        const pp = params.slice(), pm = params.slice();
        pp[i] += eps; pm[i] -= eps;
        grad[i] = (computeLoss(pp, zBatch, target, ttype, K)
                 - computeLoss(pm, zBatch, target, ttype, K)) / (2.0 * eps);
    }
    return grad;
}

/* example-based loss + analytic gradient (poly and SLP), one pass */
function lossAndGradExample(params, xData, latent, ttype, K) {
    const { mu, sigma, dist } = latent;
    const z = invertTransformParams(xData, params, ttype, K);
    const n = z.length;
    let J, dJdz, sgs = null;
    if (ttype === T_POLY) {
        const [, t1, t2, t3] = params;
        J = new Float64Array(n); dJdz = new Float64Array(n);
        for (let i = 0; i < n; i++) {
            J[i] = t1 + 2 * t2 * z[i] + 3 * t3 * z[i] * z[i];
            dJdz[i] = 2 * t2 + 6 * t3 * z[i];
        }
    } else {
        const slope = params[1];
        J = new Float64Array(n).fill(slope);
        dJdz = new Float64Array(n);
        sgs = [];
        for (let k = 0; k < K; k++) {
            const w = params[2 + 3 * k], c = params[3 + 3 * k];
            const s = Math.max(params[4 + 3 * k], 1e-6);
            const sg = new Float64Array(n);
            for (let i = 0; i < n; i++) {
                const t = clampNum((z[i] - c) / s, -50, 50);
                sg[i] = 1.0 / (1.0 + Math.exp(-t));
                J[i] += (w / s) * sg[i] * (1.0 - sg[i]);
            }
            sgs.push({ w, c, s, sg });
        }
        for (let k = 0; k < K; k++) {
            const { w, s, sg } = sgs[k];
            for (let i = 0; i < n; i++)
                dJdz[i] += (w / (s * s)) * sg[i] * (1 - sg[i]) * (1 - 2 * sg[i]);
        }
    }
    const logPz = logLatentPdf(z, mu, sigma, dist);
    const ok = new Uint8Array(n);
    let sT = 0, sE = 0, sS = 0, cnt = 0;
    for (let i = 0; i < n; i++) {
        const Jabs = Math.abs(J[i]);
        ok[i] = Jabs > 1e-10 ? 1 : 0;
        const logJ = Jabs > 1e-300 ? Math.log(Jabs + 1e-300) : -700.0;
        const v = -logPz[i] + logJ;
        if (Number.isFinite(v)) { sT += v; sE += -logPz[i]; sS += logJ; cnt++; }
    }
    if (cnt === 0)
        return { total: NaN, ener: NaN, entr: NaN,
                 grad: new Float64Array(params.length) };
    const total = sT / cnt, ener = sE / cnt, entr = sS / cnt;

    // score = d/dz log p_z(z)   (same per-case expressions as the Python)
    const score = new Float64Array(n);
    if (dist === "Gaussian") {
        for (let i = 0; i < n; i++) score[i] = -(z[i] - mu) / (sigma * sigma);
    } else if (dist === "Laplace") {
        const sl = sigma / SQRT2;
        for (let i = 0; i < n; i++) score[i] = -Math.sign(z[i] - mu) / sl;
    } else if (dist === "Cauchy") {
        for (let i = 0; i < n; i++) {
            const d = z[i] - mu;
            score[i] = -2 * d / (sigma * sigma + d * d);
        }
    } else if (dist === "Bimodal") {
        const sb = sigma * 0.6;   // kept as in the Python source
        for (let i = 0; i < n; i++) {
            const a = (z[i] + mu) / sb, b = (z[i] - mu) / sb;
            const lp1 = -0.5 * a * a, lp2 = -0.5 * b * b;
            const m = Math.max(lp1, lp2);
            const lse = m + Math.log(Math.exp(lp1 - m) + Math.exp(lp2 - m));
            score[i] = -(z[i] + mu) / (sb * sb) * Math.exp(lp1 - lse)
                       - (z[i] - mu) / (sb * sb) * Math.exp(lp2 - lse);
        }
    } else {
        const h = 1e-5;
        const zp = new Float64Array(n), zm = new Float64Array(n);
        for (let i = 0; i < n; i++) { zp[i] = z[i] + h; zm[i] = z[i] - h; }
        const lp = logLatentPdf(zp, mu, sigma, dist);
        const lm = logLatentPdf(zm, mu, sigma, dist);
        for (let i = 0; i < n; i++) score[i] = (lp[i] - lm[i]) / (2 * h);
    }

    const alpha = new Float64Array(n);
    for (let i = 0; i < n; i++)
        alpha[i] = ok[i] ? (score[i] - dJdz[i] / J[i]) / J[i] : 0.0;

    const grad = new Float64Array(params.length);
    const v = new Float64Array(n);
    if (ttype === T_POLY) {
        const derivs = [
            (zi) => [1, 0], (zi) => [zi, 1],
            (zi) => [zi * zi, 2 * zi], (zi) => [zi * zi * zi, 3 * zi * zi]];
        for (let p = 0; p < 4; p++) {
            for (let i = 0; i < n; i++) {
                const [dfdp, dJdp] = derivs[p](z[i]);
                v[i] = alpha[i] * dfdp + (ok[i] ? dJdp / J[i] : 0.0);
            }
            grad[p] = maskedMean(v, ok);
        }
    } else {
        for (let i = 0; i < n; i++) v[i] = alpha[i];
        grad[0] = maskedMean(v, ok);
        for (let i = 0; i < n; i++)
            v[i] = alpha[i] * z[i] + (ok[i] ? 1.0 / J[i] : 0.0);
        grad[1] = maskedMean(v, ok);
        for (let k = 0; k < K; k++) {
            const { w, c, s, sg } = sgs[k];
            for (let i = 0; i < n; i++) {
                const dsg = sg[i] * (1 - sg[i]);
                v[i] = alpha[i] * sg[i] + (ok[i] ? dsg / (s * J[i]) : 0.0);
            }
            grad[2 + 3 * k] = maskedMean(v, ok);
            for (let i = 0; i < n; i++) {
                const dsg = sg[i] * (1 - sg[i]), dsg2 = dsg * (1 - 2 * sg[i]);
                const dxDc = -w * dsg / s, dJDc = -w * dsg2 / (s * s);
                v[i] = alpha[i] * dxDc + (ok[i] ? dJDc / J[i] : 0.0);
            }
            grad[3 + 3 * k] = maskedMean(v, ok);
            for (let i = 0; i < n; i++) {
                const dsg = sg[i] * (1 - sg[i]), dsg2 = dsg * (1 - 2 * sg[i]);
                const zc = z[i] - c;
                const dxDs = -w * dsg * zc / (s * s);
                const dJDs = -(w / (s * s)) * (dsg + zc / s * dsg2);
                v[i] = alpha[i] * dxDs + (ok[i] ? dJDs / J[i] : 0.0);
            }
            grad[4 + 3 * k] = maskedMean(v, ok);
        }
    }
    return { total, ener, entr, grad };
}

/* example-based loss + FD gradient (used for RQS) */
function lossAndGradFdExample(params, xData, latent, ttype, K, eps = 1e-4) {
    const [total, ener, entr] = computeLossExample(
        params, xData, latent, ttype, K, true);
    const grad = new Float64Array(params.length);
    for (let i = 0; i < params.length; i++) {
        const pp = params.slice(), pm = params.slice();
        pp[i] += eps; pm[i] -= eps;
        const lp = computeLossExample(pp, xData, latent, ttype, K);
        const lm = computeLossExample(pm, xData, latent, ttype, K);
        grad[i] = (lp - lm) / (2.0 * eps);
    }
    return { total, ener, entr, grad };
}

function clipParams(params, ttype, K) {
    const out = params.slice();
    if (ttype === T_SLP) {
        for (let k = 0; k < K; k++) {
            out[2 + 3 * k] = Math.max(0.01, out[2 + 3 * k]);
            out[4 + 3 * k] = Math.max(0.05, out[4 + 3 * k]);
        }
    } else if (ttype === T_RQS) {
        out[0] = Math.max(0.5, out[0]);
    }
    return out;
}
