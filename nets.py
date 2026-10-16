import torch.nn as nn
from torchvision import models, transforms as T

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
NAMES = ["cnn", "resnet18", "mobilenet_v2"]
NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), NORM])

class CNN(nn.Module):
    def __init__(s, n=4):
        super().__init__()
        def b(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        s.features = nn.Sequential(b(3, 32), b(32, 64), b(64, 128), b(128, 256))
        s.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.4), nn.Linear(256, n))
    def forward(s, x): return s.head(s.features(x))

def build(name, pretrained=False):
    w = "DEFAULT" if pretrained else None
    if name == "cnn": return CNN()
    if name == "resnet18":
        m = models.resnet18(weights=w); m.fc = nn.Linear(512, 4); return m
    m = models.mobilenet_v2(weights=w)
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, 4)); return m

def target_layer(m, name):  # last conv block, used by Grad-CAM
    return {"cnn": lambda: m.features[-1][2], "resnet18": lambda: m.layer4[-1],
            "mobilenet_v2": lambda: m.features[-1]}[name]()
