import numpy as np
import onnxruntime as ort
from pathlib import Path

sample = np.load("data/iris_test_sample.npz")
x = sample["x"].astype(np.float32)
saved_label = int(sample["y"])

sess = ort.InferenceSession("models/iris_mlp.onnx", providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
logits = sess.run([output_name], {input_name: x.reshape(1, 4)})[0][0]
pred = int(np.argmax(logits))

# vnnlib에서는 "unsafe/adversarial condition"을 적는다.
# 즉, 어떤 다른 클래스가 현재 예측 클래스 이상이 되는 경우가 존재하는지를 찾게 한다.
label = pred
eps = 0.05

out_path = Path("specs/iris_sample_eps005.vnnlib")
out_path.parent.mkdir(exist_ok=True)

lines = []

for i in range(4):
    lines.append(f"(declare-const X_{i} Real)")
for i in range(3):
    lines.append(f"(declare-const Y_{i} Real)")

lines.append("")

for i, value in enumerate(x):
    lb = float(value - eps)
    ub = float(value + eps)
    lines.append(f"(assert (>= X_{i} {lb:.8f}))")
    lines.append(f"(assert (<= X_{i} {ub:.8f}))")

lines.append("")

bad_classes = [j for j in range(3) if j != label]

if len(bad_classes) == 1:
    j = bad_classes[0]
    lines.append(f"(assert (>= Y_{j} Y_{label}))")
else:
    lines.append("(assert (or")
    for j in bad_classes:
        lines.append(f"  (and (>= Y_{j} Y_{label}))")
    lines.append("))")

out_path.write_text("\n".join(lines) + "\n")

print("Saved", out_path)
print("saved dataset label:", saved_label)
print("onnx predicted label:", pred)
print("logits:", logits)
print("epsilon:", eps)
print("input:", x)
