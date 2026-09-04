"use strict";
/* app.js — Boltzmannator application: state, UI, figure, training.
   A faithful JavaScript port of the NiceGUI/Python Boltzmannator; runs
   entirely in the browser with no server. */

/* ── Colours (same palette as the Python version) ───────────────────────── */

const CZ        = "#1565C0";
const CT_POS    = "#C62828";
const CT_NEG    = "#E65100";
const CX        = "#2E7D32";
const CT_TARGET = "#FF8F00";
const CL_ENER   = "#1E88E5";
const CL_ENTR   = "#FB8C00";
const CM        = "#757575";
const FA        = 0.15;

const THEMES = {
    light: { bg: "#ffffff", fg: "#000000", grid: "#b0b0b0",
             phFc: "#f5f5f5", phEc: "#cccccc", lossTotal: "#263238" },
    dark:  { bg: "#23232a", fg: "#e6e6e6", grid: "#5a5a66",
             phFc: "#3a3a42", phEc: "#555555", lossTotal: "#ECEFF1" },
};

/* ── Slider specifications (label, key, min, max, default, step) ────────── */

const TRANSFORM_SLIDERS = [
    ["θ₀", "t0", -3.0, 3.0, 0.0, 0.05],
    ["θ₁", "t1", -3.0, 3.0, 1.0, 0.05],
    ["θ₂", "t2", -2.0, 2.0, 0.0, 0.05],
    ["θ₃", "t3", -0.5, 0.5, 0.0, 0.01],
];
const GAUSSIAN_SLIDERS = [
    ["μ", "mu", -2.0, 2.0, 0.0, 0.05],
    ["σ", "sg",  0.1, 2.0, 1.0, 0.05],
];
const SLP_DEFAULT_C = [-2.5, -1.79, -1.07, -0.36, 0.36, 1.07, 1.79, 2.5];

/* ── State ──────────────────────────────────────────────────────────────── */

const S = {
    vals: {},
    dist: "Gaussian",
    transform: T_SLP,
    K: 3,
    Krqs: 3,

    rescale: true,
    showIW: false,
    showMapLines: false,
    showData: true,
    showTarget: false,
    showExact: false,

    trainMode: "Energy-based",
    optimizer: "Adam",
    resample: true,

    samplesZ: null,
    dataX: null,
    lossHist: [],
    lossEner: [],
    lossEntr: [],
    training: false,
    tgtCacheKey: null,
    tgtCacheVal: null,
    trainParamsPending: null,
    trainingEpoch: 0,
    nEpochsTotal: 1,
    frozenStatic: null,
    trainParamsLive: null,
    trainParamsTarget: null,
    lastTrainedParams: null,
    sliderSnapshotAtEnd: null,
    useTrainedParams: false,
    trainStartingParams: null,
    trainZBatch: null,
    suppressRedraw: false,
    renderDirty: false,
    trainingWasActive: false,
    trainStatus: null,
    epochTimeUs: 0,
    avgEpochUs: 0,
    dark: false,
    progKind: "neutral",

    savedLims: null,      // {mainX, mainY} — for auto-rescale OFF
};

const sliders = {};      // key -> {input, valEl, row}
const UI = {};           // misc widget references

const fmt2 = (v) => (v >= 0 ? "+" : "") + v.toFixed(2);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/* ── DOM helpers ────────────────────────────────────────────────────────── */

function el(tag, attrs = {}, ...children) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
        if (k === "class") e.className = v;
        else if (k === "html") e.innerHTML = v;
        else if (k === "text") e.textContent = v;
        else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
        else e.setAttribute(k, v);
    }
    for (const c of children)
        e.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    return e;
}

function sectTitle(html) { return el("div", { class: "sect-title", html }); }
function sectSub(html)   { return el("div", { class: "sect-sub",   html }); }
function grpLabel(text)  { return el("div", { class: "grp-label",  text }); }
function sep()           { return el("hr", { class: "sep" }); }

function button(label, cls, onClick, tooltip, full = false) {
    const b = el("button", {
        class: `btn ${cls} ${full ? "full" : "grow"}`,
        text: label, onclick: onClick });
    if (tooltip) b.title = tooltip;
    return b;
}

function checkbox(label, checked, onChange) {
    const input = el("input", { type: "checkbox" });
    input.checked = checked;
    input.addEventListener("change", () => onChange(input.checked));
    const lab = el("label", { class: "check" }, input,
                   el("span", { text: label }));
    return { root: lab, input };
}

function select(options, value, onChange) {
    const s = el("select");
    for (const o of options) s.appendChild(el("option", { value: o, text: o }));
    s.value = value;
    s.addEventListener("change", () => onChange(s.value));
    return s;
}

function textInput(value, cls = "num") {
    return el("input", { type: "text", class: cls, value });
}

function radioGroup(name, options, value, onChange) {
    const root = el("div", { class: "radio-group" });
    const inputs = {};
    for (const o of options) {
        const input = el("input", { type: "radio", name, value: String(o) });
        if (String(o) === String(value)) input.checked = true;
        input.addEventListener("change", () => {
            if (input.checked) onChange(o);
        });
        inputs[o] = input;
        root.appendChild(el("label", { class: "radio" }, input,
                            el("span", { text: String(o) })));
    }
    return { root, inputs };
}

/* ── Sliders ────────────────────────────────────────────────────────────── */

function updateSliderFill(input) {
    const lo = parseFloat(input.min), hi = parseFloat(input.max);
    const p = (parseFloat(input.value) - lo) / (hi - lo) * 100;
    input.style.setProperty("--p", `${Math.max(0, Math.min(100, p))}%`);
}

function addSlider(container, label, key, min, max, def, step) {
    S.vals[key] = def;
    const input = el("input", {
        type: "range", min: String(min), max: String(max),
        step: String(step), value: String(def) });
    const valEl = el("span", { class: "val", text: fmt2(def) });
    const row = el("div", { class: "sldrow" },
        el("span", { class: "lbl", text: label }), input, valEl);
    input.addEventListener("input", () => {
        updateSliderFill(input);
        onSliderChange(key, parseFloat(input.value));
    });
    updateSliderFill(input);
    container.appendChild(row);
    sliders[key] = { input, valEl, row };
    return row;
}

function setVal(key, value) {
    value = clampNum(value, -1e6, 1e6);
    S.vals[key] = value;
    const s = sliders[key];
    if (s) {
        const lo = parseFloat(s.input.min), hi = parseFloat(s.input.max);
        s.input.value = String(clampNum(value, lo, hi));
        updateSliderFill(s.input);
        s.valEl.textContent = fmt2(value);
    }
}

function onSliderChange(key, value) {
    S.vals[key] = value;
    const s = sliders[key];
    if (s) s.valEl.textContent = fmt2(value);
    if (S.suppressRedraw) return;
    S.useTrainedParams = false;
    requestRender();
}

function refreshValueLabels() {
    for (const [key, s] of Object.entries(sliders))
        if (key in S.vals) {
            s.valEl.textContent = fmt2(S.vals[key]);
            s.input.value = String(clampNum(
                S.vals[key], parseFloat(s.input.min), parseFloat(s.input.max)));
            updateSliderFill(s.input);
        }
}

/* ── Parameter vector ───────────────────────────────────────────────────── */

function paramKeys() {
    if (S.transform === T_POLY) return ["t0", "t1", "t2", "t3"];
    if (S.transform === T_RQS) {
        const K = S.Krqs, keys = ["rqs_B"];
        for (let k = 0; k < K; k++) keys.push(`rqs_w${k}`);
        for (let k = 0; k < K; k++) keys.push(`rqs_h${k}`);
        for (let k = 0; k <= K; k++) keys.push(`rqs_d${k}`);
        return keys;
    }
    const keys = ["sig_off", "sig_slope"];
    for (let k = 0; k < S.K; k++) keys.push(`w${k}`, `c${k}`, `s${k}`);
    return keys;
}

function getParams() {
    return Float64Array.from(paramKeys().map((k) => S.vals[k]));
}

function setParams(p) {
    const keys = paramKeys();
    for (let i = 0; i < keys.length; i++)
        setVal(keys[i], clampNum(p[i], -1e6, 1e6));
}

function transformK() {
    if (S.transform === T_POLY) return 0;
    return S.transform === T_RQS ? S.Krqs : S.K;
}

/* display-time transform: honours live / trained / slider parameters */
function evalTransformDisplay(z) {
    let live = S.trainParamsLive;
    if (live === null && S.useTrainedParams && S.lastTrainedParams !== null)
        live = S.lastTrainedParams;
    const p = live !== null ? live : getParams();
    const { x, J } = evalTransformParams(z, p, S.transform, transformK());
    let isMonotone = true;
    if (S.transform === T_POLY) {
        let allPos = true, allNeg = true;
        for (const j of J) {
            if (!(j > 1e-7)) allPos = false;
            if (!(j < -1e-7)) allNeg = false;
        }
        isMonotone = allPos || allNeg;
    }
    return { x, J, isMonotone, params: p };
}

/* ── Header pictures ────────────────────────────────────────────────────── */

const PIC_FILES = ["Boltzmannator_title.png", "Boltzmannator_movie.png",
                   "Boltzmann.png", "Boltzmann_bust.png"];
const picDots = [];

function defaultPicIndex() { return S.dark ? 2 : 0; }

function choosePic(i) {
    UI.headerImg.src = `img/${PIC_FILES[i]}`;
    picDots.forEach((d, j) => d.classList.toggle("active", j === i));
}

function buildPicDots() {
    UI.headerImg = document.getElementById("header-img");
    const wrap = document.getElementById("pic-dots");
    for (let i = 0; i < PIC_FILES.length; i++) {
        const dot = el("div", { class: "pic-dot",
                                onclick: () => choosePic(i) });
        wrap.appendChild(dot);
        picDots.push(dot);
    }
    choosePic(defaultPicIndex());
}

/* ── Progress ───────────────────────────────────────────────────────────── */

function progColor(kind) {
    if (kind === "done")  return S.dark ? "#66BB6A" : "#1B5E20";
    if (kind === "error") return S.dark ? "#EF5350" : "#B71C1C";
    return S.dark ? "#e6e6e6" : "#222222";
}

function setProgKind(kind) {
    S.progKind = kind;
    if (UI.progLabel) UI.progLabel.style.color = progColor(kind);
}

function setProgress(frac, epochText, labelText) {
    if (frac !== null) UI.progFill.style.width = `${100 * frac}%`;
    if (epochText !== null) UI.progEpoch.textContent = epochText;
    if (labelText !== null) UI.progLabel.textContent = labelText;
}

/* ── UI construction ────────────────────────────────────────────────────── */

function cbChange(attr, value) {
    S[attr] = value;
    requestRender();
}

function buildDensitiesTab() {
    const t = document.getElementById("tab-dist");
    t.appendChild(sectTitle("Latent p<sub>z</sub>(z)"));
    UI.distSelect = select(["Gaussian", "Uniform", "Laplace", "Bimodal"],
                           "Gaussian", onDistChange);
    t.appendChild(UI.distSelect);
    for (const spec of GAUSSIAN_SLIDERS) addSlider(t, ...spec);

    t.appendChild(sep());
    t.appendChild(sectTitle("Target p<sup>*</sup>(x)"));
    t.appendChild(sectSub(
        "p<sup>*</sup>(x) &prop; exp(&minus;U(x)/kT),&ensp;" +
        "U = u₁x+u₂x²+u₃x³+u₄x⁴"));

    UI.showTargetCb = checkbox("Show target", false,
        (v) => cbChange("showTarget", v));
    t.appendChild(UI.showTargetCb.root);
    UI.showExactCb = checkbox("Show exact transformation", false,
        (v) => cbChange("showExact", v));
    t.appendChild(UI.showExactCb.root);

    addSlider(t, "kT", "kT", 0.1, 3.0, 1.0, 0.05);
    addSlider(t, "u₁", "u1", -2.0, 2.0, 0.0, 0.10);
    addSlider(t, "u₂", "u2", -2.0, 2.0, 1.0, 0.10);
    addSlider(t, "u₃", "u3", -1.0, 1.0, 0.0, 0.05);
    addSlider(t, "u₄", "u4",  0.05, 1.5, 0.1, 0.05);
}

function buildMapTab() {
    const t = document.getElementById("tab-map");
    t.appendChild(sectTitle("Transformation f<sub>θ</sub>(z)"));
    UI.transformSelect = select([T_POLY, T_SLP, T_RQS], T_SLP,
                                onTransformChange);
    t.appendChild(UI.transformSelect);

    // Polynomial
    UI.polySection = el("div", { class: "tabpanel" });
    UI.polySection.appendChild(sectSub(
        "x = θ₀+θ₁z+θ₂z²+θ₃z³"));
    for (const spec of TRANSFORM_SLIDERS) addSlider(UI.polySection, ...spec);
    t.appendChild(UI.polySection);

    // Single layer perceptron
    UI.sigSection = el("div", { class: "tabpanel" });
    UI.sigSection.appendChild(sectSub(
        "x = a + bz + Σwₖσ((z−cₖ)/sₖ)"));
    const kRow = el("div", { class: "row" },
                    el("span", { text: "K =" }));
    UI.kRadio = radioGroup("k-slp", [1, 2, 3, 4, 5, 6, 7, 8], 3, onKChange);
    kRow.appendChild(UI.kRadio.root);
    UI.sigSection.appendChild(kRow);

    addSlider(UI.sigSection, "a", "sig_off",   -3.0, 3.0, 0.0, 0.05);
    addSlider(UI.sigSection, "b", "sig_slope", -3.0, 3.0, 1.0, 0.05);

    UI.sigWRows = []; UI.sigCRows = []; UI.sigSRows = [];
    UI.sigSection.appendChild(grpLabel("weights  wₖ"));
    for (let k = 0; k < 8; k++)
        UI.sigWRows.push(addSlider(UI.sigSection, `w${k + 1}`, `w${k}`,
                                   0.01, 3.0, 0.01, 0.05));
    UI.sigSection.appendChild(grpLabel("centres  cₖ"));
    for (let k = 0; k < 8; k++)
        UI.sigCRows.push(addSlider(UI.sigSection, `c${k + 1}`, `c${k}`,
                                   -4.0, 4.0, SLP_DEFAULT_C[k], 0.1));
    UI.sigSection.appendChild(grpLabel("scales  sₖ"));
    for (let k = 0; k < 8; k++)
        UI.sigSRows.push(addSlider(UI.sigSection, `s${k + 1}`, `s${k}`,
                                   0.05, 3.0, 1.0, 0.05));
    t.appendChild(UI.sigSection);

    // Rational-quadratic spline
    UI.rqsSection = el("div", { class: "tabpanel" });
    UI.rqsSection.appendChild(sectSub(
        "Monotone spline on [−B, B], linear outside"));
    const kRow2 = el("div", { class: "row" }, el("span", { text: "K =" }));
    UI.kRqsRadio = radioGroup("k-rqs", [2, 3, 4], 3, onKRqsChange);
    kRow2.appendChild(UI.kRqsRadio.root);
    UI.rqsSection.appendChild(kRow2);
    addSlider(UI.rqsSection, "B", "rqs_B", 1.0, 6.0, 3.0, 0.1);
    UI.rqsWRows = []; UI.rqsHRows = []; UI.rqsDRows = [];
    UI.rqsSection.appendChild(grpLabel("Bin widths"));
    for (let k = 0; k < 4; k++)
        UI.rqsWRows.push(addSlider(UI.rqsSection, `w${k + 1}`, `rqs_w${k}`,
                                   -3.0, 3.0, 0.0, 0.05));
    UI.rqsSection.appendChild(grpLabel("Bin heights"));
    for (let k = 0; k < 4; k++)
        UI.rqsHRows.push(addSlider(UI.rqsSection, `h${k + 1}`, `rqs_h${k}`,
                                   -3.0, 3.0, 0.0, 0.05));
    UI.rqsSection.appendChild(grpLabel("Knot derivatives"));
    for (let k = 0; k <= 4; k++)
        UI.rqsDRows.push(addSlider(UI.rqsSection, `d${k}`, `rqs_d${k}`,
                                   -3.0, 3.0, 0.0, 0.05));
    t.appendChild(UI.rqsSection);

    t.appendChild(sep());
    t.appendChild(el("div", { class: "row gap" },
        button("Defaults", "btn-bluegrey", resetMapParams,
               "Reset the transformation parameters to their defaults"),
        button("Randomize", "btn-purple", randomizeMapParams,
               "Set random transformation parameters")));

    t.appendChild(sep());
    UI.showMapCb = checkbox("Show mapping lines", false,
        (v) => cbChange("showMapLines", v));
    UI.nMapInput = textInput("50", "num-sm");
    UI.nMapInput.addEventListener("change", requestRender);
    t.appendChild(el("div", { class: "row" },
        UI.showMapCb.root, el("span", { text: "N =" }), UI.nMapInput));
}

function buildTrainingTab() {
    const t = document.getElementById("tab-train");
    t.appendChild(sectTitle("Sampling"));
    UI.nEntryInput = textInput("1000");
    t.appendChild(el("div", { class: "row" },
        el("span", { text: "N =" }), UI.nEntryInput));
    t.appendChild(el("div", { class: "row gap" },
        button("Sample!", "btn-primary", doSampling,
               "Draw N samples from the latent distribution and push them " +
               "through the transformation"),
        button("Data", "btn-teal", doGenerateData,
               "Generate N example data points from the target distribution")));
    UI.showDataCb = checkbox("Show target data", true,
        (v) => cbChange("showData", v));
    t.appendChild(UI.showDataCb.root);
    UI.showIwCb = checkbox("Show importance weights", false,
        (v) => cbChange("showIW", v));
    t.appendChild(UI.showIwCb.root);

    t.appendChild(sep());
    t.appendChild(sectTitle("Training"));
    UI.modeRadio = radioGroup("train-mode", ["Energy-based", "Example-based"],
        "Energy-based", (v) => { S.trainMode = v; });
    t.appendChild(UI.modeRadio.root);
    UI.optSelect = select(["Adam", "SGD", "SGD+momentum", "RMSprop"], "Adam",
        (v) => { S.optimizer = v; });
    t.appendChild(UI.optSelect);

    UI.nEpochsInput = textInput("500");
    UI.lrInput      = textInput("0.01");
    t.appendChild(el("div", { class: "row" },
        el("span", { text: "Epochs =" }), UI.nEpochsInput,
        el("span", { text: "lr =" }), UI.lrInput));

    UI.nBatchInput = textInput("1000");
    UI.resampleCb = checkbox("resample", true, (v) => { S.resample = v; });
    t.appendChild(el("div", { class: "row" },
        el("span", { text: "N batch =" }), UI.nBatchInput,
        UI.resampleCb.root));

    UI.strideInput = textInput("10", "num-sm");
    UI.delayInput  = textInput("20", "num-sm");
    t.appendChild(el("div", { class: "row" },
        el("span", { text: "Every =" }), UI.strideInput,
        el("span", { text: "epochs" }), el("span", { class: "spacer" }),
        el("span", { text: "Delay =" }), UI.delayInput,
        el("span", { text: "ms" })));

    t.appendChild(el("div", { class: "row gap" },
        button("Train!", "btn-positive", doTraining,
               "Optimise the transformation parameters"),
        button("Stop", "btn-negative", stopTraining,
               "Stop the running training")));

    const prog = el("div", { id: "prog-wrap" },
        el("div", { id: "prog-bar" },
            el("div", { id: "prog-fill" }),
            el("div", { id: "prog-epoch", text: "0 / 0" })),
        el("div", { id: "prog-label" }));
    t.appendChild(prog);
    UI.progFill  = prog.querySelector("#prog-fill");
    UI.progEpoch = prog.querySelector("#prog-epoch");
    UI.progLabel = prog.querySelector("#prog-label");

    t.appendChild(button("Reset training", "btn-orange", resetTraining,
        "Clear loss history and reset the transformation to the identity",
        true));
}

function buildTabs() {
    const tabs = document.querySelectorAll("#tabs .tab");
    tabs.forEach((btn) => {
        btn.addEventListener("click", () => {
            tabs.forEach((b) => b.classList.toggle("active", b === btn));
            for (const name of ["dist", "map", "train"])
                document.getElementById(`tab-${name}`).classList
                        .toggle("hidden", name !== btn.dataset.tab);
        });
    });
}

/* ── Distribution / transform switching ─────────────────────────────────── */

function configureLatentSliders(dist) {
    const mu = sliders["mu"];
    S.suppressRedraw = true;
    try {
        if (dist === "Bimodal") {
            if (mu) { mu.input.min = "0"; mu.input.max = "2"; }
            setVal("mu", 1.0);
            setVal("sg", 0.5);
        } else if (mu) {
            mu.input.min = "-2"; mu.input.max = "2";
        }
    } finally {
        S.suppressRedraw = false;
    }
    refreshValueLabels();
}

function onDistChange(val) {
    S.dist = val;
    configureLatentSliders(val);
    requestRender();
}

function onTransformChange(val) {
    if (val !== undefined) S.transform = val;
    UI.polySection.classList.toggle("hidden", S.transform !== T_POLY);
    UI.sigSection.classList.toggle("hidden",  S.transform !== T_SLP);
    UI.rqsSection.classList.toggle("hidden",  S.transform !== T_RQS);
    if (S.transform === T_RQS) updateRqsRows(); else updateSigRows();
    S.useTrainedParams = false;
    requestRender();
}

function onKChange(k) {
    S.K = k;
    updateSigRows();
    S.useTrainedParams = false;
    requestRender();
}

function onKRqsChange(k) {
    S.Krqs = k;
    updateRqsRows();
    S.useTrainedParams = false;
    requestRender();
}

function updateSigRows() {
    for (const rows of [UI.sigWRows, UI.sigCRows, UI.sigSRows])
        rows.forEach((r, i) => r.classList.toggle("hidden", i >= S.K));
}

function updateRqsRows() {
    UI.rqsWRows.forEach((r, i) => r.classList.toggle("hidden", i >= S.Krqs));
    UI.rqsHRows.forEach((r, i) => r.classList.toggle("hidden", i >= S.Krqs));
    UI.rqsDRows.forEach((r, i) => r.classList.toggle("hidden", i > S.Krqs));
}

/* ── Reset / randomize ──────────────────────────────────────────────────── */

function resetSlpDefaults() {
    setVal("sig_off", 0.0); setVal("sig_slope", 1.0);
    for (let k = 0; k < 8; k++) {
        setVal(`w${k}`, 0.01);
        setVal(`c${k}`, SLP_DEFAULT_C[k]);
        setVal(`s${k}`, 1.0);
    }
}

function resetApp() {
    S.dist = "Gaussian";
    UI.distSelect.value = "Gaussian";
    configureLatentSliders("Gaussian");
    for (const [, key, , , def] of GAUSSIAN_SLIDERS) setVal(key, def);
    for (const [, key, , , def] of TRANSFORM_SLIDERS) setVal(key, def);
    S.transform = T_SLP;
    UI.transformSelect.value = T_SLP;
    resetSlpDefaults();
    S.K = 3;
    UI.kRadio.inputs[3].checked = true;
    onTransformChange();
    for (const [key, val] of [["kT", 1.0], ["u1", 0.0], ["u2", 1.0],
                              ["u3", 0.0], ["u4", 0.1]])
        setVal(key, val);
    S.showTarget = false;
    S.showExact = false;
    S.tgtCacheKey = null;
    UI.showTargetCb.input.checked = false;
    UI.showExactCb.input.checked = false;
    UI.showDataCb.input.checked = true;
    UI.showIwCb.input.checked = false;
    UI.showMapCb.input.checked = false;
    UI.rescaleCb.checked = true;
    UI.nEntryInput.value = "1000";
    UI.nMapInput.value = "50";
    S.showData = true;
    S.showIW = false;
    S.showMapLines = false;
    S.rescale = true;
    S.lossHist = []; S.lossEner = []; S.lossEntr = [];
    S.lastTrainedParams = null;
    S.sliderSnapshotAtEnd = null;
    S.useTrainedParams = false;
    S.samplesZ = null;
    S.dataX = null;
    S.savedLims = null;
    setProgress(0, "0 / 0", "");
    setProgKind("neutral");
    refreshValueLabels();
    requestRender();
}

function resetTraining() {
    S.training = false;
    S.lossHist = []; S.lossEner = []; S.lossEntr = [];
    S.trainParamsPending = null;
    S.trainParamsLive = null;
    S.trainParamsTarget = null;
    S.trainZBatch = null;
    S.lastTrainedParams = null;
    S.sliderSnapshotAtEnd = null;
    S.useTrainedParams = false;
    S.trainingEpoch = 0;
    S.frozenStatic = null;
    S.trainingWasActive = false;
    setProgress(0, "0 / 0", "");
    setProgKind("neutral");
    S.suppressRedraw = true;
    try {
        if (S.transform === T_POLY) {
            setVal("t0", 0.0); setVal("t1", 1.0);
            setVal("t2", 0.0); setVal("t3", 0.0);
        } else if (S.transform === T_RQS) {
            setVal("rqs_B", 3.0);
            for (let k = 0; k < S.Krqs; k++) {
                setVal(`rqs_w${k}`, 0.0); setVal(`rqs_h${k}`, 0.0);
            }
            for (let k = 0; k <= S.Krqs; k++) setVal(`rqs_d${k}`, 0.0);
        } else {
            resetSlpDefaults();
        }
    } finally {
        S.suppressRedraw = false;
    }
    refreshValueLabels();
    requestRender();
}

function resetMapParams() {
    S.useTrainedParams = false;
    S.suppressRedraw = true;
    try {
        if (S.transform === T_POLY) {
            for (const [, key, , , def] of TRANSFORM_SLIDERS) setVal(key, def);
        } else if (S.transform === T_RQS) {
            setVal("rqs_B", 3.0);
            for (let k = 0; k < S.Krqs; k++) {
                setVal(`rqs_w${k}`, 0.0); setVal(`rqs_h${k}`, 0.0);
            }
            for (let k = 0; k <= S.Krqs; k++) setVal(`rqs_d${k}`, 0.0);
        } else {
            resetSlpDefaults();
            S.K = 3;
            UI.kRadio.inputs[3].checked = true;
            updateSigRows();
        }
    } finally {
        S.suppressRedraw = false;
    }
    refreshValueLabels();
    requestRender();
}

function randomizeMapParams() {
    S.useTrainedParams = false;
    const rng = makeRng(null);
    S.suppressRedraw = true;
    try {
        if (S.transform === T_POLY) {
            for (const [, key, lo, hi] of TRANSFORM_SLIDERS)
                setVal(key, rng.uniform(lo, hi));
        } else if (S.transform === T_RQS) {
            setVal("rqs_B", rng.uniform(1.5, 5.0));
            for (let k = 0; k < S.Krqs; k++) {
                setVal(`rqs_w${k}`, rng.uniform(-2.0, 2.0));
                setVal(`rqs_h${k}`, rng.uniform(-2.0, 2.0));
            }
            for (let k = 0; k <= S.Krqs; k++)
                setVal(`rqs_d${k}`, rng.uniform(-1.5, 1.5));
        } else {
            setVal("sig_off",   rng.uniform(-2.0, 2.0));
            setVal("sig_slope", rng.uniform(0.2, 2.0));
            for (let k = 0; k < S.K; k++) {
                setVal(`w${k}`, rng.uniform(0.1, 2.0));
                setVal(`c${k}`, rng.uniform(-3.0, 3.0));
                setVal(`s${k}`, rng.uniform(0.1, 2.0));
            }
        }
    } finally {
        S.suppressRedraw = false;
    }
    refreshValueLabels();
    requestRender();
}

/* ── Target cache / sampling / data ─────────────────────────────────────── */

function targetVals() {
    return { kT: S.vals["kT"], u1: S.vals["u1"], u2: S.vals["u2"],
             u3: S.vals["u3"], u4: S.vals["u4"] };
}

function targetCache() {
    const tg = targetVals();
    const key = [tg.kT, tg.u1, tg.u2, tg.u3, tg.u4]
        .map((v) => Math.round(v * 1000) / 1000).join(",");
    if (key === S.tgtCacheKey) return S.tgtCacheVal;
    const xWide = linspace(-15, 15, 2000);
    const logP = new Float64Array(2000);
    let shift = -Infinity;
    for (let i = 0; i < 2000; i++) {
        logP[i] = -potentialU(xWide[i], tg) / tg.kT;
        if (logP[i] > shift) shift = logP[i];
    }
    const pWide = new Float64Array(2000);
    for (let i = 0; i < 2000; i++) pWide[i] = Math.exp(logP[i] - shift);
    const Z = trapz(pWide, xWide);
    S.tgtCacheKey = key;
    S.tgtCacheVal = { xWide, pWide, Z, shift };
    return S.tgtCacheVal;
}

function doSampling() {
    let N = parseInt(UI.nEntryInput.value, 10);
    if (!Number.isFinite(N) || N < 1) N = 1000;
    S.samplesZ = sampleLatent(N, S.vals["mu"], S.vals["sg"], S.dist,
                              makeRng(null));
    requestRender();
}

function doGenerateData() {
    let N = parseInt(UI.nEntryInput.value, 10);
    if (!Number.isFinite(N) || N < 1) N = 1000;
    const { xWide, pWide, Z } = targetCache();
    if (Z <= 0) return;
    const dx = xWide[1] - xWide[0];
    const cdfX = new Float64Array(xWide.length);
    let acc = 0;
    for (let i = 0; i < xWide.length; i++) {
        acc += pWide[i] * dx / Z;
        cdfX[i] = clampNum(acc, 0.0, 1.0);
    }
    const mu = S.vals["mu"], sg = S.vals["sg"];
    const rng = makeRng(null);
    const zSamp = sampleLatent(N, mu, sg, S.dist, rng);
    const zGrid = linspace(arrMin(zSamp) - 3 * sg, arrMax(zSamp) + 3 * sg, 4000);
    const pGrid = latentPdf(zGrid, mu, sg, S.dist);
    const dz = zGrid[1] - zGrid[0];
    const cdfZ = new Float64Array(4000);
    let acc2 = 0;
    for (let i = 0; i < 4000; i++) { acc2 += pGrid[i] * dz; cdfZ[i] = acc2; }
    const norm = Math.max(cdfZ[3999], 1e-12);
    for (let i = 0; i < 4000; i++) cdfZ[i] = clampNum(cdfZ[i] / norm, 0, 1);
    const uVals = interpArr(zSamp, zGrid, cdfZ);
    S.dataX = interpArr(uVals, cdfX, xWide);
    requestRender();
}

/* ── z-grid construction (matches the Python thresholding) ──────────────── */

function makeZGrid(mu, sg, dist, N, probeN) {
    let zProbe;
    if (dist === "Bimodal") {
        const half = Math.abs(mu) + 15.0 * sg;
        zProbe = linspace(-half, half, probeN);
    } else {
        zProbe = linspace(mu - 15.0 * sg, mu + 15.0 * sg, probeN);
    }
    const pProbe = latentPdf(zProbe, mu, sg, dist);
    const peak = arrMax(pProbe) || 1.0;
    const thr = 0.005;
    let lo = null, hi = null;
    for (let i = 0; i < probeN; i++)
        if (pProbe[i] > thr * peak) {
            if (lo === null) lo = zProbe[i];
            hi = zProbe[i];
        }
    if (lo === null) {
        if (dist === "Bimodal") {
            lo = -(Math.abs(mu) + 4.5 * sg); hi = Math.abs(mu) + 4.5 * sg;
        } else {
            lo = mu - 4.5 * sg; hi = mu + 4.5 * sg;
        }
    }
    const pad = (hi - lo) * 0.12;
    const z = linspace(lo - pad, hi + pad, N);
    return { z, pZ: latentPdf(z, mu, sg, dist) };
}

/* ── Figure drawing ─────────────────────────────────────────────────────── */

const LOGICAL_W = 1250, LOGICAL_H = 920;
let canvas, ctx;

function requestRender() { S.renderDirty = true; }

function drawFigure() {
    const theme = S.dark ? THEMES.dark : THEMES.light;
    const dpr = window.devicePixelRatio || 1;
    const cw = canvas.clientWidth, ch = canvas.clientHeight;
    if (cw === 0 || ch === 0) return;
    if (canvas.width !== Math.round(cw * dpr) ||
        canvas.height !== Math.round(ch * dpr)) {
        canvas.width = Math.round(cw * dpr);
        canvas.height = Math.round(ch * dpr);
    }
    const scale = Math.min(cw / LOGICAL_W, ch / LOGICAL_H);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.setTransform(dpr * scale, 0, 0, dpr * scale,
                     dpr * (cw - LOGICAL_W * scale) / 2,
                     dpr * (ch - LOGICAL_H * scale) / 2);

    /* — data preparation — */
    let mu, sg, dist, z, pZ;
    if (S.frozenStatic) {
        ({ mu, sg, dist, z, pZ } = S.frozenStatic);
    } else {
        mu = S.vals["mu"]; sg = S.vals["sg"]; dist = S.dist;
        ({ z, pZ } = makeZGrid(mu, sg, dist, 500, 800));
    }
    const nZ = z.length;
    const { x, J, isMonotone } = evalTransformDisplay(z);
    const invJabs = new Float64Array(nZ), pxCurve = new Float64Array(nZ);
    for (let i = 0; i < nZ; i++) {
        const a = Math.abs(J[i]);
        invJabs[i] = a > 1e-9 ? 1.0 / a : NaN;
        pxCurve[i] = a > 1e-9 ? pZ[i] / a : 0.0;
    }
    const order = Array.from({ length: nZ }, (_, i) => i)
                       .sort((a, b) => x[a] - x[b]);
    const xSorted = Float64Array.from(order, (i) => x[i]);
    const pxSorted = Float64Array.from(order, (i) => pxCurve[i]);

    /* — target curve (cached) — */
    const tg = targetVals();
    let pTgtMax = 0, xTgtLo = null, xTgtHi = null;
    let xDispTgt = null, pTgtCurve = null, tgtCdf = null, tgtXWide = null;
    if (S.showTarget || S.showExact) {
        const { xWide, pWide, Z, shift } = targetCache();
        if (Z > 0) {
            const dx = xWide[1] - xWide[0];
            const cdf = new Float64Array(xWide.length);
            let acc = 0;
            for (let i = 0; i < xWide.length; i++) {
                acc += pWide[i] * dx / Z;
                cdf[i] = acc;
            }
            tgtCdf = cdf; tgtXWide = xWide;
            const thr = 0.001;
            let loIdx = searchsortedLeft(cdf, thr);
            let hiIdx = Math.min(xWide.length - 1,
                                 searchsortedLeft(cdf, 1.0 - thr));
            loIdx = Math.max(0, loIdx);
            xTgtLo = xWide[loIdx]; xTgtHi = xWide[hiIdx];
            const margin = (xTgtHi - xTgtLo) * 0.10;
            xTgtLo -= margin; xTgtHi += margin;
            const xDisp = linspace(xTgtLo, xTgtHi, 500);
            const pTgt = new Float64Array(500);
            for (let i = 0; i < 500; i++) {
                pTgt[i] = Math.exp(-potentialU(xDisp[i], tg) / tg.kT - shift) / Z;
                if (pTgt[i] > pTgtMax) pTgtMax = pTgt[i];
            }
            if (S.showTarget) { xDispTgt = xDisp; pTgtCurve = pTgt; }
        }
    }

    /* — axis ranges (with auto-rescale lock) — */
    const xCurveLo = arrMin(x), xCurveHi = arrMax(x);
    const span = (xCurveHi - xCurveLo) || 1.0;
    let yLoMain = xCurveLo - 0.12 * span;
    let yHiMain = xCurveHi + 0.12 * span;
    if (S.showTarget && xTgtLo !== null) {
        yLoMain = Math.min(yLoMain, xTgtLo);
        yHiMain = Math.max(yHiMain, xTgtHi);
    }
    let mainX = [z[0], z[nZ - 1]], mainY = [yLoMain, yHiMain];
    if (!S.rescale && S.savedLims) {
        mainX = S.savedLims.mainX;
        mainY = S.savedLims.mainY;
    } else {
        S.savedLims = { mainX: mainX.slice(), mainY: mainY.slice() };
    }

    /* — layout — */
    const xf = (f) => f * LOGICAL_W, yf = (f) => f * LOGICAL_H;
    const rect = (cx, cy) => ({
        x: xf(cx[0]), y: yf(cy[0]),
        w: xf(cx[1]) - xf(cx[0]), h: yf(cy[1]) - yf(cy[0]) });
    const outerCols = splitCells(0.055, 0.975, [3.5, 2], 0.22);
    const leftRows  = splitCells(0.045, 0.935, [1, 3, 1], 0.28);
    const leftCols  = splitCells(outerCols[0][0], outerCols[0][1], [3, 1], 0.06);
    const rightRows = splitCells(0.045, 0.935, [1, 1, 1], 0.55);

    const axTop   = new Axes(ctx, rect(leftCols[0], leftRows[0]), theme);
    const axMain  = new Axes(ctx, rect(leftCols[0], leftRows[1]), theme);
    const axLogj  = new Axes(ctx, rect(leftCols[0], leftRows[2]), theme);
    const axRight = new Axes(ctx, rect(leftCols[1], leftRows[1]), theme);
    const axHistZ = new Axes(ctx, rect(outerCols[1], rightRows[0]), theme);
    const axHistX = new Axes(ctx, rect(outerCols[1], rightRows[1]), theme);
    const axLoss  = new Axes(ctx, rect(outerCols[1], rightRows[2]), theme);

    const showGrid = !S.showMapLines;

    /* — top panel: latent density — */
    axTop.setXlim(mainX[0], mainX[1]);
    axTop.setYlim(0, (arrMax(pZ) || 1) * 1.05);
    axTop.computeTicks({ nx: 6, ny: 3 });
    if (showGrid) axTop.grid();
    axTop.fillUnder(z, pZ, { color: CZ, alpha: FA });
    axTop.line(z, pZ, { color: CZ, lw: 2.4 });
    axTop.frame({ spines: { top: false, right: false, bottom: false,
                            left: true }, ticksX: false, ticksY: true });
    axTop.ylabel(["p", ["z", "sub"], "(z)"]);

    /* — main panel: transformation — */
    axMain.setXlim(mainX[0], mainX[1]);
    axMain.setYlim(mainY[0], mainY[1]);
    axMain.computeTicks({ nx: 6, ny: 6 });
    if (showGrid) axMain.grid();
    if (!S.showMapLines) {
        axMain.hline(0, { color: "#999", lw: 0.9 });
        axMain.vline(0, { color: "#999", lw: 0.9 });
    }
    /* mapping lines */
    if (S.showMapLines) {
        let nMl = parseInt(UI.nMapInput.value, 10);
        if (!Number.isFinite(nMl)) nMl = 0;
        if (nMl > 100) { nMl = 100; UI.nMapInput.value = "100"; }
        nMl = Math.max(0, nMl);
        if (nMl > 0) {
            const zMl = linspace(z[0], z[nZ - 1], nMl);
            const { x: xMl } = evalTransformDisplay(zMl);
            for (let i = 0; i < nMl; i++) {
                axMain.line([zMl[i], zMl[i]], [mainY[1], xMl[i]],
                            { color: CM, lw: 0.9, alpha: 0.6 });
                axMain.line([zMl[i], mainX[1]], [xMl[i], xMl[i]],
                            { color: CM, lw: 0.9, alpha: 0.6 });
            }
        }
    }
    /* transformation curve, split into monotone runs by sign(J) */
    const drawRuns = (mask, color, ax, ysrc) => {
        let start = null;
        for (let i = 0; i <= nZ; i++) {
            const on = i < nZ && mask(i);
            if (on && start === null) start = i;
            if (!on && start !== null) {
                if (i - start >= 2)
                    ax.line(z.slice(start, i), ysrc.slice(start, i),
                            { color, lw: 2.4 });
                start = null;
            }
        }
    };
    drawRuns((i) => J[i] >= 0, CT_POS, axMain, x);
    drawRuns((i) => J[i] < 0,  CT_NEG, axMain, x);
    /* exact transformation x*(z) */
    if (S.showExact && tgtCdf !== null) {
        const dzg = z[1] - z[0];
        const cdfZ = new Float64Array(nZ);
        let acc = 0;
        for (let i = 0; i < nZ; i++) { acc += pZ[i] * dzg; cdfZ[i] = acc; }
        if (cdfZ[nZ - 1] > 0)
            for (let i = 0; i < nZ; i++) cdfZ[i] /= cdfZ[nZ - 1];
        const xStar = interpArr(cdfZ, tgtCdf, tgtXWide);
        axMain.line(z, xStar, { color: CT_TARGET, lw: 1.9 });
    }
    if (!isMonotone) {
        for (let i = 0; i < nZ - 1; i++) {
            if (Math.sign(J[i]) !== Math.sign(J[i + 1])) {
                const zt = (z[i] + z[i + 1]) / 2;
                const { x: xt } = evalTransformDisplay(new Float64Array([zt]));
                axMain.marker(zt, xt[0], { r: 5.5, color: "#FF6D00" });
            }
        }
        axMain.textAxes(0.03, 0.955, "Not invertible ✗",
            { size: 10 * FS, color: "#B71C1C", ha: "left", va: "top",
              bold: true,
              bbox: { fc: "#FFEBEE", ec: "#B71C1C", lw: 1.2 } });
    }
    axMain.frame({ spines: { top: true, right: true, bottom: true,
                             left: true }, ticksX: false, ticksY: true });
    axMain.ylabel(["x = f", ["θ", "sub"], "(z)"]);

    /* — Jacobian panel — */
    axLogj.setXlim(mainX[0], mainX[1]);
    const iJf = Array.from(invJabs).filter(Number.isFinite);
    const iJTop = iJf.length > 0 ? percentile(iJf, 99) * 1.15 : 2.0;
    axLogj.setYlim(0, Math.max(iJTop, 0.1));
    axLogj.computeTicks({ nx: 6, ny: 3 });
    if (showGrid) axLogj.grid();
    axLogj.hline(1, { color: "#999", lw: 1, dash: [6, 4] });
    drawRuns((i) => J[i] >= 0, CT_POS, axLogj, invJabs);
    drawRuns((i) => J[i] < 0,  CT_NEG, axLogj, invJabs);
    axLogj.frame({ spines: { top: false, right: false, bottom: true,
                             left: true } });
    axLogj.xlabel("z");
    axLogj.ylabel(["|J|", ["−1", "sup"]]);

    /* — right panel: pushed-forward density — */
    axRight.setYlim(mainY[0], mainY[1]);
    let pxMax = 0;
    if (isMonotone) {
        pxMax = arrMax(pxSorted);
    }
    let kdeX = null, kdeP = null;
    if (!isMonotone) {
        try {
            const rng0 = makeRng(0);
            const zS = sampleLatent(5000, mu, sg, dist, rng0);
            const { x: xS } = evalTransformDisplay(zS);
            const kde = gaussianKde(xS);
            kdeX = linspace(arrMin(xS), arrMax(xS), 400);
            kdeP = kde(kdeX);
            pxMax = arrMax(kdeP);
        } catch (e) { /* ignore */ }
    }
    axRight.setXlim(0, Math.max(pxMax, pTgtMax) * 1.25 || 1.0);
    axRight.computeTicks({ nx: 3, ny: 6 });
    if (showGrid) axRight.grid();
    if (isMonotone) {
        axRight.fillLeft(pxSorted, xSorted, { color: CX, alpha: FA });
        axRight.line(pxSorted, xSorted, { color: CX, lw: 2.4 });
    } else if (kdeX !== null) {
        axRight.fillLeft(kdeP, kdeX, { color: CX, alpha: FA });
        axRight.line(kdeP, kdeX, { color: CX, lw: 2.4, dash: [7, 5] });
        axRight.textAxes(0.93, 0.03, "KDE",
            { size: 9 * FS, color: "#B71C1C", ha: "right", va: "bottom",
              bbox: { fc: "#FFEBEE", ec: "#B71C1C", lw: 1.2, pad: 5 } });
    }
    if (S.showTarget && xDispTgt !== null) {
        axRight.fillLeft(pTgtCurve, xDispTgt, { color: CT_TARGET, alpha: FA });
        axRight.line(pTgtCurve, xDispTgt, { color: CT_TARGET, lw: 2.4 });
    }
    axRight.frame({ spines: { top: false, right: false, bottom: true,
                              left: false }, ticksY: false });
    axRight.xlabel(["p", ["x", "sub"], "(x)"]);

    /* — histogram of latent samples — */
    const trainBatch = S.training ? S.trainZBatch : null;
    const displaySamples = trainBatch !== null ? trainBatch : S.samplesZ;
    const ph = { fc: theme.phFc, ec: theme.phEc, alpha: 0.9 };

    if (displaySamples !== null) {
        axHistZ.setXlim(mainX[0], mainX[1]);
        axHistZ.setYlim(0, (arrMax(pZ) || 1) * 1.3);
        axHistZ.computeTicks({ nx: 5, ny: 4 });
        axHistZ.grid();
        axHistZ.hist(displaySamples, { bins: 40, color: CZ, alpha: 0.4 });
        axHistZ.line(z, pZ, { color: CZ, lw: 2.4 });
        axHistZ.frame({ spines: { top: false, right: false, bottom: true,
                                  left: true } });
        axHistZ.xlabel("z");
        axHistZ.ylabel("density");
        axHistZ.title("Latent z");
    } else {
        axHistZ.title("Latent z");
        axHistZ.xlabel("z", { offset: 20 });
        axHistZ.ylabel("density", { offset: 14 });
        axHistZ.frame({ spines: { top: false, right: false, bottom: true,
                                  left: true },
                        ticksX: false, ticksY: false });
        axHistZ.textAxes(0.5, 0.5, "Press  'Sample!'\nto generate points",
            { size: 9 * FS, color: "#888", bbox: ph });
    }

    /* — histogram of transformed samples — */
    if (displaySamples !== null) {
        const { x: samplesX } = evalTransformDisplay(displaySamples);
        axHistX.setXlim(mainY[0], mainY[1]);
        let yTop = 0;
        if (isMonotone) yTop = arrMax(pxSorted);
        const histMax = axHistX.histMaxDensity(samplesX, 40);
        if (pTgtCurve !== null) yTop = Math.max(yTop, arrMax(pTgtCurve));
        let dataHistMax = 0;
        if (S.dataX !== null && S.showData)
            dataHistMax = axHistX.histMaxDensity(S.dataX, 40);
        yTop = Math.max(yTop, dataHistMax);
        let kdeP2 = null, kdeX2 = null;
        if (!isMonotone) {
            try {
                const kde2 = gaussianKde(samplesX);
                kdeX2 = linspace(arrMin(samplesX), arrMax(samplesX), 400);
                kdeP2 = kde2(kdeX2);
                yTop = Math.max(yTop, arrMax(kdeP2));
            } catch (e) { /* ignore */ }
        }
        const yLimTop = yTop > 0 ? yTop * 1.3 : histMax * 1.3 || 1.0;
        axHistX.setYlim(0, yLimTop);
        axHistX.computeTicks({ nx: 5, ny: 4 });
        axHistX.grid();
        axHistX.hist(samplesX, { bins: 40, color: CX, alpha: 0.4 });
        if (isMonotone)
            axHistX.line(xSorted, pxSorted, { color: CX, lw: 2.4 });
        else if (kdeX2 !== null)
            axHistX.line(kdeX2, kdeP2, { color: CX, lw: 2.4, dash: [7, 5] });
        if (pTgtCurve !== null) {
            axHistX.fillUnder(xDispTgt, pTgtCurve,
                              { color: CT_TARGET, alpha: FA });
            axHistX.line(xDispTgt, pTgtCurve, { color: CT_TARGET, lw: 2.4 });
        }
        if (S.dataX !== null && S.showData)
            axHistX.hist(S.dataX, { bins: 40, color: CT_TARGET, alpha: 0.35 });
        axHistX.frame({ spines: { top: false, right: false, bottom: true,
                                  left: true } });
        axHistX.xlabel("x");
        axHistX.ylabel("density");
        axHistX.title("Transformed x");

        /* importance weights on a twin axis */
        if (S.showTarget && isMonotone && S.showIW &&
            S.tgtCacheVal !== null && pTgtCurve !== null) {
            drawImportanceWeights(axHistX, displaySamples, xSorted, pxSorted,
                                  mu, sg, dist, tg, theme);
        }
    } else {
        axHistX.title("Transformed x");
        axHistX.xlabel("x", { offset: 20 });
        axHistX.ylabel("density", { offset: 14 });
        axHistX.frame({ spines: { top: false, right: false, bottom: true,
                                  left: true },
                        ticksX: false, ticksY: false });
        axHistX.textAxes(0.5, 0.5, "Press  'Sample!'\nto generate points",
            { size: 9 * FS, color: "#888", bbox: ph });
    }

    /* — training loss — */
    axLoss.title("Training loss", { pad: 30 });
    if (S.lossHist.length > 0) {
        const nPts = S.lossHist.length;
        const MAX_PLOT = 100;
        const nTotal = Math.max(nPts, S.nEpochsTotal);
        let idx;
        if (nTotal > MAX_PLOT) {
            const step = Math.max(1, Math.floor(nTotal / MAX_PLOT));
            let maxI = 0;
            for (let i = 1; i < nPts; i++)
                if (S.lossHist[i] > S.lossHist[maxI]) maxI = i;
            const set = new Set();
            for (let i = 0; i < nPts; i += step) set.add(i);
            set.add(maxI);
            set.add(nPts - 1);
            idx = Array.from(set).sort((a, b) => a - b);
        } else {
            idx = Array.from({ length: nPts }, (_, i) => i);
        }
        const ep = idx.map((i) => i + 1);
        const tot = idx.map((i) => S.lossHist[i]);
        const ener = idx.map((i) => S.lossEner[i]);
        const entr = idx.map((i) => S.lossEntr[i]);
        const all = [];
        for (const arr of [S.lossHist, S.lossEner, S.lossEntr])
            for (const v of arr) if (Number.isFinite(v)) all.push(v);
        let yLo = 0, yHi = 1;
        if (all.length > 0) {
            yLo = Math.min(...all); yHi = Math.max(...all);
            const pad = (yHi - yLo) * 0.08 || Math.abs(yHi) * 0.08 || 0.1;
            yLo -= pad; yHi += pad;
        }
        axLoss.setXlim(1, Math.max(nPts, 2));
        axLoss.setYlim(yLo, yHi);
        axLoss.computeTicks({ nx: 4, ny: 4, integerX: true });
        axLoss.grid();
        const mode = S.trainMode;
        const lbl1 = mode === "Energy-based"
            ? ["⟨U⟩/k", ["B", "sub"], "T"]
            : ["−⟨log p", ["z", "sub"], "⟩"];
        const lbl2 = mode === "Energy-based"
            ? "−⟨log|J|⟩" : "⟨log|J|⟩";
        const lblT = mode === "Example-based" ? "NLL" : "total";
        axLoss.line(ep, ener, { color: CL_ENER, lw: 1.6 });
        axLoss.line(ep, entr, { color: CL_ENTR, lw: 1.6 });
        axLoss.line(ep, tot,  { color: theme.lossTotal, lw: 1.6 });
        axLoss.legendAbove([
            { color: CL_ENER, label: lbl1, lw: 2 },
            { color: CL_ENTR, label: lbl2, lw: 2 },
            { color: theme.lossTotal, label: lblT, lw: 2 }]);
        axLoss.frame({ spines: { top: false, right: false, bottom: true,
                                 left: true } });
        axLoss.xlabel("epoch");
        axLoss.ylabel("loss");
    } else {
        axLoss.xlabel("epoch", { offset: 20 });
        axLoss.ylabel("loss", { offset: 14 });
        axLoss.frame({ spines: { top: false, right: false, bottom: true,
                                 left: true },
                       ticksX: false, ticksY: false });
        axLoss.textAxes(0.5, 0.5, "Press  'Train!'\nto start training",
            { size: 9 * FS, color: "#888", bbox: ph });
    }
}

function drawImportanceWeights(axHistX, displaySamples, xSorted, pxSorted,
                               mu, sg, dist, tg, theme) {
    const { Z, shift } = S.tgtCacheVal;
    if (Z <= 0) return;
    const CW = S.dark ? "#C9A27A" : "#6D4C41";
    const n = xSorted.length;
    const wSmooth = new Float64Array(n).fill(NaN);
    for (let i = 0; i < n; i++) {
        if (pxSorted[i] > 1e-10) {
            const pstar = Math.exp(-potentialU(xSorted[i], tg) / tg.kT - shift) / Z;
            wSmooth[i] = pstar / pxSorted[i];
        }
    }
    const { x: sx, J: jSamp } = evalTransformDisplay(displaySamples);
    const pzSamp = latentPdf(displaySamples, mu, sg, dist);
    let sumW = 0, sumW2 = 0, nTot = 0;
    for (let i = 0; i < displaySamples.length; i++) {
        const jA = Math.abs(jSamp[i]);
        const pxS = jA > 1e-9 ? pzSamp[i] / jA : 0.0;
        const pstarS = Math.exp(-potentialU(sx[i], tg) / tg.kT - shift) / Z;
        if (pxS > 1e-10 && Number.isFinite(pstarS)) {
            const w = pstarS / pxS;
            sumW += w; sumW2 += w * w; nTot++;
        }
    }
    const wFin = Array.from(wSmooth).filter(Number.isFinite);
    if (nTot === 0 || wFin.length === 0) return;
    const nEff = (sumW * sumW) / sumW2;
    const wHi = Math.max(percentile(wFin, 99) * 1.25, 1.5);

    /* twin axes sharing the rect, own y-scale */
    const tw = new Axes(axHistX.ctx, axHistX.rect, theme);
    tw.setXlim(axHistX.xlim[0], axHistX.xlim[1]);
    tw.setYlim(0, wHi);
    tw.line(xSorted, wSmooth, { color: CW, lw: 1.7, dash: [7, 5] });
    tw.hline(1.0, { color: CW, lw: 1, dash: [2, 4], alpha: 0.6 });
    /* right spine + ticks in the weights colour */
    const { ctx: c, rect: r } = tw;
    c.save();
    c.strokeStyle = CW;
    c.lineWidth = 1.1;
    c.beginPath();
    c.moveTo(r.x + r.w, r.y); c.lineTo(r.x + r.w, r.y + r.h);
    c.stroke();
    const yt = niceTicks(0, wHi, 3);
    for (const t of yt) {
        const y = tw.py(t);
        c.beginPath();
        c.moveTo(r.x + r.w, y); c.lineTo(r.x + r.w + 5, y);
        c.stroke();
        drawRich(c, tickLabel(t, yt.length > 1 ? yt[1] - yt[0] : 1),
                 r.x + r.w + 8, y,
                 { size: 8 * FS, color: CW, align: "left", baseline: "middle" });
    }
    c.restore();
    tw.ylabelRight("w(x)", { color: CW, offset: 40 });
    tw.textAxes(0.97, 0.97,
        ["N", ["eff", "sub"], ` = ${(100 * nEff / nTot).toFixed(1)}%`],
        { size: 8.5 * FS, color: CW, ha: "right", va: "top" });
}

/* ── Training ───────────────────────────────────────────────────────────── */

function freezeStaticForTraining() {
    const mu = S.vals["mu"], sg = S.vals["sg"], dist = S.dist;
    const { z, pZ } = makeZGrid(mu, sg, dist, 800, 2000);
    S.frozenStatic = { mu, sg, dist, z, pZ };
}

function doTraining() {
    if (S.training) return;
    S.trainParamsPending = null;
    S.trainingEpoch = 0;
    let nTot = parseInt(UI.nEpochsInput.value, 10);
    if (!Number.isFinite(nTot) || nTot < 1) nTot = 0;
    setProgress(0, `0 / ${nTot}`, "");
    setProgKind("neutral");
    /* make the target visible so convergence is observable */
    S.showTarget = true;
    UI.showTargetCb.input.checked = true;
    const current = getParams();
    if (S.lastTrainedParams !== null && S.sliderSnapshotAtEnd !== null &&
        current.length === S.sliderSnapshotAtEnd.length &&
        current.every((v, i) => Math.abs(v - S.sliderSnapshotAtEnd[i]) < 1e-6)) {
        S.trainStartingParams = S.lastTrainedParams.slice();
    } else {
        S.trainStartingParams = current;
    }
    S.training = true;
    S.useTrainedParams = false;
    S.trainingWasActive = true;
    S.frozenStatic = null;
    drawFigure();
    freezeStaticForTraining();
    trainLoop();          // async fire-and-forget (the JS "thread")
}

function stopTraining() { S.training = false; }

async function trainLoop() {
    let nEpochs = parseInt(UI.nEpochsInput.value, 10);
    if (!Number.isFinite(nEpochs) || nEpochs < 1) nEpochs = 200;
    S.nEpochsTotal = nEpochs;
    let lr = parseFloat(UI.lrInput.value);
    if (!Number.isFinite(lr)) lr = 0.01;
    const mu = S.vals["mu"], sg = S.vals["sg"], dist = S.dist;
    let nBatch = parseInt(UI.nBatchInput.value, 10);
    if (!Number.isFinite(nBatch)) nBatch = 1000;
    nBatch = Math.max(10, nBatch);
    const trainMode = S.trainMode;
    const resampleEachEpoch = S.resample;
    const rng = makeRng(42);
    let zBatch = sampleLatent(nBatch, mu, sg, dist, rng);
    S.trainZBatch = zBatch;
    const xData = S.dataX;
    if (trainMode === "Example-based" && (xData === null || xData.length === 0)) {
        S.trainStatus = "No data — click 'Data' first";
        S.training = false;
        return;
    }
    const latent = { mu, sigma: sg, dist };
    let stride = parseInt(UI.strideInput.value, 10);
    if (!Number.isFinite(stride)) stride = 10;
    stride = Math.max(1, stride);
    let delayMs = parseFloat(UI.delayInput.value);
    if (!Number.isFinite(delayMs)) delayMs = 50;
    delayMs = Math.max(0, delayMs);
    let params = S.trainStartingParams !== null
        ? S.trainStartingParams.slice() : getParams();
    const optimizer = S.optimizer;
    const target = targetVals();
    const ttype = S.transform;
    const K = transformK();
    const P = params.length;
    let m = new Float64Array(P), v = new Float64Array(P);
    const b1 = 0.9, b2 = 0.999, epsOpt = 1e-8;
    S.epochTimeUs = 0;
    const emaAlpha = 0.1;
    let epochUsSum = 0;
    S.avgEpochUs = 0;
    let bestLoss = Infinity;
    let bestParams = params.slice();

    for (let t = 1; t <= nEpochs; t++) {
        if (!S.training) { S.trainStatus = "Stopped"; return; }
        const t0 = performance.now();
        let loss, ener, entr, grad;
        if (trainMode === "Energy-based") {
            if (resampleEachEpoch) {
                zBatch = sampleLatent(nBatch, mu, sg, dist, rng);
                S.trainZBatch = zBatch;
            }
            [loss, ener, entr] = computeLoss(params, zBatch, target,
                                             ttype, K, true);
            if (!Number.isFinite(loss)) {
                S.trainStatus = "Loss diverged — reduce lr";
                S.training = false;
                return;
            }
            grad = (ttype === T_RQS)
                ? gradientFd(params, zBatch, target, ttype, K)
                : gradientAnalytic(params, zBatch, target, ttype, K);
        } else {
            const res = (ttype === T_RQS)
                ? lossAndGradFdExample(params, xData, latent, ttype, K)
                : lossAndGradExample(params, xData, latent, ttype, K);
            loss = res.total; ener = res.ener; entr = res.entr; grad = res.grad;
            if (!Number.isFinite(loss)) {
                S.trainStatus = "Loss diverged — reduce lr";
                S.training = false;
                return;
            }
        }
        S.lossHist.push(loss);
        S.lossEner.push(ener);
        S.lossEntr.push(entr);
        if (loss < bestLoss) { bestLoss = loss; bestParams = params.slice(); }

        if (optimizer === "Adam") {
            for (let i = 0; i < P; i++) {
                m[i] = b1 * m[i] + (1 - b1) * grad[i];
                v[i] = b2 * v[i] + (1 - b2) * grad[i] * grad[i];
                const mh = m[i] / (1 - Math.pow(b1, t));
                const vh = v[i] / (1 - Math.pow(b2, t));
                params[i] -= lr * mh / (Math.sqrt(vh) + epsOpt);
            }
        } else if (optimizer === "SGD") {
            for (let i = 0; i < P; i++) params[i] -= lr * grad[i];
        } else if (optimizer === "SGD+momentum") {
            for (let i = 0; i < P; i++) {
                m[i] = b1 * m[i] + grad[i];
                params[i] -= lr * m[i];
            }
        } else if (optimizer === "RMSprop") {
            for (let i = 0; i < P; i++) {
                v[i] = b2 * v[i] + (1 - b2) * grad[i] * grad[i];
                params[i] -= lr * grad[i] / (Math.sqrt(v[i]) + epsOpt);
            }
        }
        params = clipParams(params, ttype, K);
        const dtUs = (performance.now() - t0) * 1000;
        epochUsSum += dtUs;
        S.epochTimeUs = (t === 1) ? dtUs
            : (1 - emaAlpha) * S.epochTimeUs + emaAlpha * dtUs;

        if (t % stride === 0 || t === nEpochs) {
            S.trainParamsPending = params.slice();
            S.trainingEpoch = t;
            await sleep(delayMs);
        }
    }
    S.avgEpochUs = epochUsSum / Math.max(nEpochs, 1);
    /* finish on the best-loss parameters */
    S.trainParamsPending = bestParams.slice();
    S.trainingEpoch = nEpochs;
    S.training = false;
}

function doEndOfTraining() {
    const final = S.trainParamsTarget;
    if (final !== null) {
        S.suppressRedraw = true;
        try { setParams(final); }
        finally { S.suppressRedraw = false; }
        refreshValueLabels();
        S.lastTrainedParams = Float64Array.from(final);
        S.sliderSnapshotAtEnd = getParams();
    }
    S.trainParamsTarget = null;
    if (S.trainZBatch !== null) S.samplesZ = S.trainZBatch.slice();
    S.trainZBatch = null;
    S.frozenStatic = null;
    S.trainParamsLive = null;
    S.useTrainedParams = true;
    S.renderDirty = true;
    const n = S.nEpochsTotal;
    let msg;
    if (S.trainStatus !== null) {
        msg = S.trainStatus;
        S.trainStatus = null;
        setProgKind("error");
    } else {
        const avg = S.avgEpochUs;
        const avgStr = avg >= 1000
            ? `${(avg / 1000).toFixed(1)} ms/epoch`
            : `${avg.toFixed(0)} μs/epoch`;
        msg = `Done — ${n} epochs  ·  avg ${avgStr}`;
        setProgKind("done");
    }
    setProgress(1, `${n} / ${n}`, msg);
}

/* ── 30 ms tick: live-parameter blending, progress, render dispatch ─────── */

function tick() {
    if (S.training || S.trainingWasActive) {
        const pending = S.trainParamsPending;
        if (pending !== null) {
            S.trainParamsPending = null;
            S.trainParamsTarget = pending;
            if (S.trainParamsLive === null)
                S.trainParamsLive = pending.slice();
        }
        if (S.trainParamsTarget !== null && S.trainParamsLive !== null) {
            if (!S.training) {
                S.trainParamsLive = S.trainParamsTarget.slice();
                if (S.trainZBatch !== null)
                    S.samplesZ = S.trainZBatch.slice();
                S.renderDirty = true;
            } else {
                let maxDiff = 0;
                for (let i = 0; i < S.trainParamsLive.length; i++) {
                    const d = Math.abs(
                        S.trainParamsTarget[i] - S.trainParamsLive[i]);
                    if (d > maxDiff) maxDiff = d;
                }
                if (maxDiff > 1e-5) {
                    for (let i = 0; i < S.trainParamsLive.length; i++)
                        S.trainParamsLive[i] += 0.25 *
                            (S.trainParamsTarget[i] - S.trainParamsLive[i]);
                    S.renderDirty = true;
                }
            }
        }
        const t = S.trainingEpoch, n = S.nEpochsTotal;
        const loss = S.lossHist.length > 0
            ? S.lossHist[S.lossHist.length - 1] : NaN;
        if (S.training) {
            setProgress(t / Math.max(n, 1), `${t} / ${n}`,
                `loss ${loss.toFixed(3)}  ·  ` +
                `${S.epochTimeUs.toFixed(0)} μs`);
        }
        if (!S.training) {
            if (S.trainingWasActive) {
                S.trainingWasActive = false;
                doEndOfTraining();
            }
        } else {
            S.trainingWasActive = true;
        }
    }
    if (S.renderDirty) {
        S.renderDirty = false;
        drawFigure();
    }
}

/* ── Dark mode ──────────────────────────────────────────────────────────── */

function toggleDark(value) {
    S.dark = value;
    document.documentElement.classList.toggle("dark", value);
    setProgKind(S.progKind);
    choosePic(defaultPicIndex());
    requestRender();
}

/* ── Boot ───────────────────────────────────────────────────────────────── */

function init() {
    canvas = document.getElementById("plot");
    ctx = canvas.getContext("2d");

    buildPicDots();
    buildTabs();
    buildDensitiesTab();
    buildMapTab();
    buildTrainingTab();

    UI.rescaleCb = document.getElementById("cb-rescale");
    UI.rescaleCb.addEventListener("change",
        () => cbChange("rescale", UI.rescaleCb.checked));
    UI.darkCb = document.getElementById("cb-dark");
    UI.darkCb.addEventListener("change", () => toggleDark(UI.darkCb.checked));

    document.getElementById("btn-reset").addEventListener("click", resetApp);
    document.getElementById("btn-exit").addEventListener("click", () => {
        window.close();
        setTimeout(() => { window.location.href = "about:blank"; }, 150);
    });

    onTransformChange();

    new ResizeObserver(() => requestRender())
        .observe(document.getElementById("plotcard"));

    // On iOS an orientation change can fire the resize observer with stale
    // dimensions, leaving the plot band mis-sized until the next interaction.
    // Force redraws once the new layout has settled.
    const refitAfterRotate = () => {
        requestRender();
        setTimeout(requestRender, 200);
        setTimeout(requestRender, 500);
    };
    window.addEventListener("orientationchange", refitAfterRotate);
    window.addEventListener("resize", refitAfterRotate);

    setupFullscreenButton();

    drawFigure();
    setInterval(tick, 30);
}

/* Full-screen toggle over the plot.  Only wired up (and shown) where the
   Fullscreen API is available — Android Chrome, desktop — so on iOS Safari,
   which does not support it for regular pages, no dead button appears. */
function setupFullscreenButton() {
    const ICON_EXPAND =
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
        '<path d="M8 3H5a2 2 0 0 0-2 2v3M16 3h3a2 2 0 0 1 2 2v3' +
        'M8 21H5a2 2 0 0 1-2-2v-3M16 21h3a2 2 0 0 0 2-2v-3"/></svg>';
    const ICON_COLLAPSE =
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
        '<path d="M8 3v3a2 2 0 0 1-2 2H3M16 3v3a2 2 0 0 0 2 2h3' +
        'M8 21v-3a2 2 0 0 0-2-2H3M16 21v-3a2 2 0 0 1 2-2h3"/></svg>';
    const btn = document.createElement("button");
    btn.id = "btn-fs";
    btn.setAttribute("aria-label", "Toggle full screen");
    btn.title = "Full screen";
    btn.innerHTML = ICON_EXPAND;
    document.getElementById("plotwrap").appendChild(btn);

    const supported = !!(document.fullscreenEnabled ||
                         document.webkitFullscreenEnabled);
    if (!supported) return;                 // stays hidden (no .avail)
    btn.classList.add("avail");

    const isFs = () => document.fullscreenElement ||
                       document.webkitFullscreenElement;
    btn.addEventListener("click", () => {
        if (isFs()) {
            (document.exitFullscreen || document.webkitExitFullscreen)
                .call(document);
        } else {
            const el = document.documentElement;
            (el.requestFullscreen || el.webkitRequestFullscreen)
                .call(el).catch(() => {});
        }
    });
    const onChange = () => {
        btn.innerHTML = isFs() ? ICON_COLLAPSE : ICON_EXPAND;
        requestRender();                    // re-fit the canvas to the new size
    };
    document.addEventListener("fullscreenchange", onChange);
    document.addEventListener("webkitfullscreenchange", onChange);
}

document.addEventListener("DOMContentLoaded", init);
