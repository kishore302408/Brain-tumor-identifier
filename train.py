"""Usage: python train.py <dataset_dir> [epochs]
dataset_dir must contain Training/ and Testing/ with glioma, meningioma, notumor, pituitary folders."""
import sys, os, json, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T
from sklearn.metrics import confusion_matrix, classification_report
from nets import CLASSES, NAMES, NORM, tf, build

def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    EP = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    aug = T.Compose([T.Resize((224, 224)), T.RandomHorizontalFlip(), T.RandomRotation(10),
                     T.ColorJitter(.2, .2), T.ToTensor(), NORM])
    tr = datasets.ImageFolder(f"{root}/Training", aug)
    te = datasets.ImageFolder(f"{root}/Testing", tf)
    assert tr.classes == CLASSES, tr.classes
    L = DataLoader(tr, 128, shuffle=True, num_workers=0, pin_memory=False)
    V = DataLoader(te, 256, num_workers=0, pin_memory=False)
    os.makedirs("weights", exist_ok=True); os.makedirs("metrics", exist_ok=True)
    for name in NAMES:
        m = build(name, True).to(dev)
        opt = torch.optim.AdamW(m.parameters(), 1e-3 if name == "cnn" else 3e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EP)
        for e in range(EP):
            m.train()
            for x, y in L:
                x, y = x.to(dev), y.to(dev)
                opt.zero_grad(); nn.functional.cross_entropy(m(x), y).backward(); opt.step()
            sch.step(); print(f"{name} epoch {e+1}/{EP}", flush=True)
        m.eval(); P, Y = [], []
        with torch.no_grad():
            for x, y in V:
                P += m(x.to(dev)).argmax(1).cpu().tolist(); Y += y.tolist()
        rep = classification_report(Y, P, target_names=CLASSES, output_dict=True)
        json.dump({"accuracy": rep["accuracy"], "confusion": confusion_matrix(Y, P).tolist(), "report": rep},
                  open(f"metrics/{name}.json", "w"))
        torch.save(m.cpu().state_dict(), f"weights/{name}.pt")
        print(f"{name}: test accuracy {rep['accuracy']:.4f}")

if __name__ == "__main__":
    main()
