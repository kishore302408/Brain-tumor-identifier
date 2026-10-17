import os, json
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="NeuroScan Pro", page_icon="🧠", layout="wide")

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
NAMES = ["cnn", "resnet18", "mobilenet_v2"]
MODEL_CHOICES = ["resnet18 (best)", "mobilenet_v2", "cnn", "All + Ensemble"]
RESAMPLE = getattr(Image, "Resampling", Image).BICUBIC

st.title("🧠 NeuroScan Pro — Brain Tumor MRI Classification")
st.caption("MED-02 Medical AI • Upload MRI scans → AI prediction with Grad-CAM, confidence & clinical metrics")

# ---------------- Metrics (pure JSON, renders instantly — no torch needed) ----------------
@st.cache_data
def load_metrics():
    met = {}
    for n in NAMES:
        p = os.path.join("metrics", f"{n}.json")
        if os.path.exists(p):
            with open(p) as f:
                met[n] = json.load(f)
    return met

metrics = load_metrics()

# ---------------- Lazy heavy backend: torch imported only on first analysis ----------------
@st.cache_resource
def get_backend():
    import torch
    from nets import tf, build, target_layer
    return torch, tf, build, target_layer

@st.cache_resource
def load_model(name):
    torch, tf, build, _ = get_backend()
    path = os.path.join("weights", f"{name}.pt")
    if not os.path.exists(path):
        return None
    m = build(name)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval()

def gradcam(m, name, x):
    torch, _, _, target_layer = get_backend()
    holder = {}
    def fh(mod, inp, out):
        holder["a"] = out
        out.register_hook(lambda g: holder.__setitem__("g", g))
    h = target_layer(m, name).register_forward_hook(fh)
    m.zero_grad()
    out = m(x)
    p = out.softmax(1)[0]
    out[0, int(p.argmax())].backward()
    h.remove()
    cam = torch.relu((holder["g"].mean((2, 3), keepdim=True) * holder["a"]).sum(1))[0].detach().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return p.detach(), cam

def overlay(img, cam):
    a = np.asarray(Image.fromarray((cam * 255).astype("uint8")).resize(img.size, RESAMPLE)) / 255.0
    heat = np.stack([np.clip(1.5 - np.abs(4 * a - 3), 0, 1),
                     np.clip(1.5 - np.abs(4 * a - 2), 0, 1),
                     np.clip(1.5 - np.abs(4 * a - 1), 0, 1)], -1) * 255
    al = (0.65 * a)[..., None]
    out = np.asarray(img) * (1 - al) + heat * al
    return Image.fromarray(out.astype("uint8"))

def seg_overlay(img, cam):
    """Approximate tumor localisation: top-quartile Grad-CAM activation as a mask."""
    a = np.asarray(Image.fromarray((cam * 255).astype("uint8")).resize(img.size, RESAMPLE)) / 255.0
    thr = max(float(np.percentile(a, 75)), 0.30)
    mask = (a >= thr).astype(np.float32)
    arr = np.asarray(img).astype(np.float32)
    green = np.zeros_like(arr)
    green[..., 1] = 255
    out = arr * (1 - 0.45 * mask[..., None]) + green * (0.45 * mask[..., None])
    return Image.fromarray(out.astype("uint8")), float(mask.mean())

def parametric_features(img):
    """Lightweight parametric/structural features computed directly from the scan."""
    g = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    gx = float(np.abs(np.diff(g, axis=1)).mean())
    gy = float(np.abs(np.diff(g, axis=0)).mean())
    return {
        "mean_intensity": float(g.mean()),
        "std_contrast": float(g.std()),
        "edge_density": (gx + gy) / 2,
        "bright_region_ratio": float((g > 0.7).mean()),
    }

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.selectbox("Model", MODEL_CHOICES, index=0,
                                help="ResNet18 achieved 95.13% test accuracy")
    show_features = st.toggle("Show parametric features", value=True)
    show_seg = st.toggle("Show tumor localisation mask (approx.)", value=False)
    uploaded = st.file_uploader("Upload MRI images", type=["jpg", "jpeg", "png"],
                                accept_multiple_files=True)
    analyze = st.button("🔍 Analyze scans", type="primary",
                        disabled=not uploaded, use_container_width=True)
    st.caption("ℹ️ Free tier: the first analysis may take 1–2 minutes while the model loads.")

# ---------------- Model performance (instant) ----------------
st.subheader("📊 Model Performance")
if metrics:
    avail = [n for n in NAMES if n in metrics]
    tabs = st.tabs(avail)
    for t, n in zip(tabs, avail):
        with t:
            m = metrics[n]
            rep = m["report"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{m['accuracy']:.2%}")
            c2.metric("Macro F1", f"{rep['macro avg']['f1-score']:.3f}")
            c3.metric("Macro Precision", f"{rep['macro avg']['precision']:.3f}")
            c4.metric("Macro Recall", f"{rep['macro avg']['recall']:.3f}")
            cc1, cc2 = st.columns([1, 1])
            with cc1:
                st.markdown("**Confusion matrix**")
                cm = pd.DataFrame(m["confusion"], index=CLASSES, columns=CLASSES)
                try:
                    st.dataframe(cm.style.background_gradient(cmap="Oranges").format("{:g}"))
                except Exception:
                    st.dataframe(cm)  # fallback if matplotlib is unavailable
            with cc2:
                st.markdown("**Classification report**")
                rows = [c for c in CLASSES if c in rep] + ["macro avg", "weighted avg"]
                rep_df = pd.DataFrame(
                    {c: {k: rep[c][k] for k in ("precision", "recall", "f1-score", "support")}
                     for c in rows}).T
                st.dataframe(rep_df.style.format({"precision": "{:.3f}", "recall": "{:.3f}",
                                                  "f1-score": "{:.3f}", "support": "{:.0f}"}))
else:
    st.info("Run train.py first to generate metrics.")

# ---------------- Analysis ----------------
if analyze and uploaded:
    selected = NAMES if model_choice == "All + Ensemble" else [model_choice.split(" ")[0]]
    progress = st.progress(0, text="Loading backend…")
    results = []
    total = len(uploaded)
    for ui, f in enumerate(uploaded):
        torch, tf, _, _ = get_backend()
        img = Image.open(f).convert("RGB")
        x = tf(img).unsqueeze(0)
        per_model, cams = {}, {}
        for mi, n in enumerate(selected):
            progress.progress((ui + mi / max(len(selected), 1)) / total,
                              text=f"Running {n} on {f.name}…")
            m = load_model(n)
            if m is None:
                st.warning(f"No trained weights found for {n} — skipped.")
                continue
            p, cam = gradcam(m, n, x)
            per_model[n] = {"label": CLASSES[int(p.argmax())], "conf": float(p.max()),
                            "probs": {c: float(v) for c, v in zip(CLASSES, p.tolist())}}
            cams[n] = cam
        entry = {"name": f.name, "original": img, "models": per_model, "cams": cams}
        if len(per_model) > 1:
            avg = np.mean([list(v["probs"].values()) for v in per_model.values()], axis=0)
            entry["ensemble"] = {"label": CLASSES[int(avg.argmax())], "conf": float(avg.max()),
                                 "probs": {c: float(v) for c, v in zip(CLASSES, avg.tolist())}}
        results.append(entry)
    progress.progress(1.0, text="Done")
    st.session_state["results"] = results

# ---------------- Results ----------------
if "results" in st.session_state and st.session_state["results"]:
    st.subheader("🔬 Predictions")
    for r in st.session_state["results"]:
        st.markdown(f"**{r['name']}**")
        if "ensemble" in r:
            e = r["ensemble"]
            st.success(f"🧮 Ensemble verdict: **{e['label']}** — {e['conf']:.1%} confidence")
        n_mod = len(r["models"])
        cols = st.columns(1 + n_mod)
        with cols[0]:
            st.image(r["original"], caption="Original scan")
        for i, (n, mres) in enumerate(r["models"].items()):
            with cols[i + 1]:
                st.image(overlay(r["original"], r["cams"][n]),
                         caption=f"{n}: {mres['label']} · {mres['conf']:.1%}")
                st.bar_chart(pd.Series(mres["probs"], name="probability"))
        if show_seg and r["cams"]:
            best = max(r["models"], key=lambda n: r["models"][n]["conf"])
            seg_img, cov = seg_overlay(r["original"], r["cams"][best])
            st.image(seg_img,
                     caption=f"Approx. localisation ({best}, top-quartile activation, "
                             f"{cov:.1%} of image highlighted)")
        if show_features:
            with st.expander("Parametric / structural features"):
                feats = parametric_features(r["original"])
                st.json(feats)

    st.subheader("📋 Comparison & Report")
    rows = []
    for r in st.session_state["results"]:
        if "ensemble" in r:
            rows.append({"Image": r["name"], "Model": "ensemble",
                         "Prediction": r["ensemble"]["label"],
                         "Confidence": r["ensemble"]["conf"]})
        for n, mres in r["models"].items():
            rows.append({"Image": r["name"], "Model": n, "Prediction": mres["label"],
                         "Confidence": mres["conf"]})
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df)
        st.download_button("⬇️ Download CSV report", df.to_csv(index=False).encode(),
                           "mri_report.csv", "text/csv", use_container_width=True)

    st.markdown("---")
    st.caption("⚠️ Educational tool for the MED-02 challenge — not a medical device and not "
               "for diagnostic use.")
