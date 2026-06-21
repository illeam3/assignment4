import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class IrisMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )

    def forward(self, x):
        return self.net(x)


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    iris = load_iris()
    X = iris.data.astype(np.float32)
    y = iris.target.astype(np.int64)

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    model = IrisMLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(1000):
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(X_test_t).argmax(dim=1)
        acc = (pred == y_test_t).float().mean().item()

    print(f"Test accuracy: {acc:.4f}")

    torch.save(model.state_dict(), "models/iris_mlp.pth")

    dummy = torch.zeros(1, 4, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        "models/iris_mlp.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,
        opset_version=12,
        dynamo=False,
    )

    np.savez(
        "data/iris_test_sample.npz",
        x=X_test[0].astype(np.float32),
        y=np.array(y_test[0], dtype=np.int64),
        mean=scaler.mean_.astype(np.float32),
        scale=scaler.scale_.astype(np.float32),
        test_accuracy=np.array(acc, dtype=np.float32),
    )

    print("Saved models/iris_mlp.onnx")
    print("Saved data/iris_test_sample.npz")
    print("Selected test label:", y_test[0])
    print("Selected test input:", X_test[0])


if __name__ == "__main__":
    main()
