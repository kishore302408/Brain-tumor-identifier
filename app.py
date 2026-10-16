import io, os, json, base64, torch, numpy as np
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
from nets import CLASSES, NAMES, tf, build, target_layer

app = Flask(__name__, static_folder="static")
M = {}
for n in NAMES:
    if os.path.exists(f"weights/{n}.pt"):
        m = build(n); m.load_state_dict(torch.load(f"weights/{n}.pt", map_location="cpu")); M[n] = m.eval()

def gradcam(m, name, x):
    st = {}
    def fh(mod, i, o):
        st["a"] = o; o.register_hook(lambda g: st.__setitem__("g", g))
    h = target_layer(m, name).register_forward_hook(fh)
    m.zero_grad(); out = m(x); p = out.softmax(1)[0]
    out[0, int(p.argmax())].backward(); h.remove()
    cam = torch.relu((st["g"].mean((2, 3), keepdim=True) * st["a"]).sum(1))[0].detach().numpy()
    return p.detach(), (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

def overlay(img, cam):
    a = np.asarray(Image.fromarray((cam * 255).astype("uint8")).resize(img.size, Image.BICUBIC)) / 255.0
    heat = np.stack([np.clip(1.5 - abs(4 * a - 3), 0, 1), np.clip(1.5 - abs(4 * a - 2), 0, 1),
                     np.clip(1.5 - abs(4 * a - 1), 0, 1)], -1) * 255
    al = (0.65 * a)[..., None]
    out = np.asarray(img) * (1 - al) + heat * al
    buf = io.BytesIO(); Image.fromarray(out.astype("uint8")).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()

@app.get("/")
def index(): return send_from_directory("static", "index.html")

@app.get("/api/info")
def info():
    met = {n: json.load(open(f"metrics/{n}.json")) for n in M if os.path.exists(f"metrics/{n}.json")}
    return jsonify(models=list(M), classes=CLASSES, metrics=met)

@app.post("/api/predict")
def predict():
    sel = request.form.get("model", "all")
    use = list(M) if sel == "all" else [sel]
    if not use or any(n not in M for n in use):
        return jsonify(error="No trained model found. Run train.py first."), 400
    out = []
    for f in request.files.getlist("images"):
        img = Image.open(f.stream).convert("RGB"); x = tf(img).unsqueeze(0); res = {}
        for n in use:
            p, cam = gradcam(M[n], n, x)
            res[n] = dict(label=CLASSES[int(p.argmax())], conf=float(p.max()), heat=overlay(img, cam),
                          probs=dict(zip(CLASSES, p.tolist())))
        avg = np.mean([list(r["probs"].values()) for r in res.values()], 0)
        out.append(dict(name=f.filename, models=res,
                        ensemble=dict(label=CLASSES[int(avg.argmax())], conf=float(avg.max()),
                                      probs=dict(zip(CLASSES, avg.tolist())))))
    return jsonify(out)

if __name__ == "__main__":
    print("Loaded models:", list(M)); app.run(port=5001)
