import os, io, json, base64, torch, numpy as np
from PIL import Image
import streamlit as st
from torchvision import transforms as T
from nets import CLASSES, NAMES, tf, build, target_layer

st.set_page_config(page_title="NeuroScan Pro", page_icon="🧠", layout="wide")

@st.cache_resource
def load_models():
    M = {}
    os.makedirs("weights", exist_ok=True)
    for n in NAMES:
        if os.path.exists(f"weights/{n}.pt"):
            m = build(n)
            m.load_state_dict(torch.load(f"weights/{n}.pt", map_location="cpu"))
            M[n] = m.eval()
    return M

@st.cache_data
def load_metrics():
    met = {}
    for n in NAMES:
        p = f"metrics/{n}.json"
        if os.path.exists(p):
            met[n] = json.load(open(p))
    return met

M = load_models()
metrics = load_metrics()

def gradcam(m, name, x):
    st_dict = {}
    def fh(mod, i, o):
        st_dict["a"] = o
        o.register_hook(lambda g: st_dict.__setitem__("g", g))
    h = target_layer(m, name).register_forward_hook(fh)
    m.zero_grad()
    out = m(x)
    p = out.softmax(1)[0]
    out[0, int(p.argmax())].backward()
    h.remove()
    cam = torch.relu((st_dict["g"].mean((2,3), keepdim=True) * st_dict["a"]).sum(1))[0].detach().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return p.detach(), cam

def overlay(img, cam):
    a = np.asarray(Image.fromarray((cam*255).astype("uint8")).resize(img.size, Image.BICUBIC)) / 255.0
    heat = np.stack([np.clip(1.5 - abs(4*a - 3),0,1),
                     np.clip(1.5 - abs(4*a - 2),0,1),
                     np.clip(1.5 - abs(4*a - 1),0,1)], -1) * 255
    al = (0.65*a)[...,None]
    out = np.asarray(img)*(1-al) + heat*al
    return Image.fromarray(out.astype("uint8"))

st.title("🧠 NeuroScan Pro — Brain Tumor MRI Classification")
st.caption("MED-02 Medical AI • Upload MRI → AI prediction with Grad-CAM, confidence & clinical metrics")

col1, col2 = st.columns([1,2])
with col1:
    st.subheader("Settings")
    model_choice = st.selectbox("Model", ["All + Ensemble"] + NAMES)
    enable_features = st.checkbox("Show parametric features", value=True)
    enable_seg = st.checkbox("Show segmentation mask overlay (simulated)", value=False)
    uploaded = st.file_uploader("Upload MRI images", type=["jpg","jpeg","png"], accept_multiple_files=True)

with col2:
    st.subheader("Model Performance")
    if metrics:
        tabs = st.tabs(NAMES)
        for i,n in enumerate(NAMES):
            with tabs[i]:
                m = metrics.get(n)
                if not m:
                    st.write("No metrics")
                    continue
                st.metric("Accuracy", f"{m['accuracy']:.2%}")
                st.dataframe(m['report'])
                # confusion matrix heatmap
                import pandas as pd
                cm = np.array(m['confusion'])
                df = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
                st.dataframe(df.style.background_gradient(cmap="Oranges"))
    else:
        st.info("Run train.py first to generate metrics.")

if uploaded:
    st.subheader("Predictions")
    results = []
    for f in uploaded:
        img = Image.open(f).convert("RGB")
        x = tf(img).unsqueeze(0)
        st.markdown(f"**{f.name}**")
        cols = st.columns(2 + (1 if model_choice=="All + Ensemble" else 0))
        with cols[0]:
            st.image(img, caption="Original", use_column_width=True)
        model_res = {}
        probs_all = []
        for n in M:
            if model_choice!= "All + Ensemble" and n!=model_choice:
                continue
            p, cam = gradcam(M[n], n, x)
            label = CLASSES[int(p.argmax())]
            conf = float(p.max())
            model_res[n] = {"label":label, "conf":conf, "probs":p.tolist(), "cam":cam}
            probs_all.append(p.tolist())
            with cols[1] if model_choice=="All + Ensemble" else cols[0]:
                st.image(overlay(img, cam), caption=f"{n}: {label} {conf:.1%}", use_column_width=True)
        if model_choice=="All + Ensemble" and len(probs_all)>1:
            avg = np.mean(probs_all,0)
            ens_label = CLASSES[int(avg.argmax())]
            ens_conf = float(avg.max())
            st.success(f"Ensemble → {ens_label} {ens_conf:.1%}")
            results.append({"name":f.name, "ensemble":ens_label, "conf":ens_conf, "models":model_res})
        else:
            results.append({"name":f.name, "models":model_res})

    st.subheader("Comparison Table")
    import pandas as pd
    rows = []
    for r in results:
        for n, m in r.get("models",{}).items():
            rows.append({"Image":r["name"],"Model":n,"Prediction":m["label"],"Confidence":m["conf"]})
    if rows:
        st.dataframe(pd.DataFrame(rows))
        csv = pd.DataFrame(rows).to_csv(index=False).encode()
        st.download_button("Download CSV Report", csv, "mri_report.csv", "text/csv")

    if enable_features:
        st.subheader("Parametric Features")
        st.info("Texture, shape and fractal dimension features are computed from ROI. Demo values shown.")
        st.json({"mean_intensity": 0.42, "contrast": 0.31, "fractal_dim": 1.68})

    if enable_seg:
        st.subheader("Segmentation")
        st.warning("Segmentation is simulated in this demo. Full annotation data required for clinical segmentation.")
