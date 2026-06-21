# Assignment 4: Neural Network Verification with alpha-beta-CROWN

This repository contains my Assignment 4 implementation for neural network verification using `alpha-beta-CROWN`.

## Project Overview

The goal of this assignment is to explore the structure of `alpha-beta-CROWN`, prepare a verification configuration, and run a verification query on an external neural network model.

For the external model, I used a small Multi-Layer Perceptron trained on the Iris dataset. The model was exported to ONNX and verified with `alpha-beta-CROWN` using a VNNLIB property.

## Model and Dataset

- Dataset: Iris dataset from scikit-learn
- Model: MLP with input dimension 4 and output dimension 3
- Architecture:
  - Linear(4, 16)
  - ReLU
  - Linear(16, 16)
  - ReLU
  - Linear(16, 3)
- Export format: ONNX
- Model file: `models/iris_mlp.onnx`

## Verification Property

The verification property checks local robustness for one selected Iris test sample.

The input is allowed to vary within an L-infinity perturbation box with epsilon = 0.05. The VNNLIB file encodes the unsafe condition: another class output becomes greater than or equal to the originally predicted class output.

If `alpha-beta-CROWN` returns `unsat`, this means no unsafe input exists inside the perturbation region, so the model is verified as locally robust for that sample and epsilon.

## Important Files

- `create_iris_model.py`: trains the Iris MLP and exports it to ONNX
- `make_iris_vnnlib.py`: creates the VNNLIB property file
- `models/iris_mlp.onnx`: exported external model
- `specs/iris_sample_eps005.vnnlib`: verification property
- `configs/iris_mlp.yaml`: alpha-beta-CROWN configuration file
- `test.py`: runs alpha-beta-CROWN on the Iris model and VNNLIB specification
- `results/iris_out.txt`: summarized alpha-beta-CROWN result
- `report.pdf`: short report for the assignment

## Setup

Clone `alpha-beta-CROWN` into the `external` directory.

```bash
mkdir -p external
cd external
git clone --recursive https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
cd alpha-beta-CROWN
uv sync
source .venv/bin/activate
```

Return to the project root.

```bash
cd ../../
pip install -r requirements.txt
```

## Reproduce the Model and Property

```bash
python create_iris_model.py
python make_iris_vnnlib.py
```

## Run Verification

```bash
python test.py
```

Alternatively, run `alpha-beta-CROWN` directly.

```bash
cd external/alpha-beta-CROWN/complete_verifier
python abcrown.py --config ../../../configs/iris_mlp.yaml
```

## Result

For the selected Iris sample and epsilon = 0.05, `alpha-beta-CROWN` returned:

```text
Result: unsat
```

This means the unsafe adversarial condition is unsatisfiable. Therefore, no adversarial example exists within the specified L-infinity perturbation box for this sample, and the model is locally robust under the tested condition.

## Comparison with Marabou

Compared with Marabou, `alpha-beta-CROWN` uses bound propagation methods such as CROWN, alpha-CROWN, beta-CROWN, and branch-and-bound to efficiently reason about neural network robustness. Marabou is based more directly on SMT-style solving and piecewise-linear constraint reasoning.

In this assignment, `alpha-beta-CROWN` was fast and convenient for ONNX + VNNLIB robustness verification, while Marabou provided a more explicit logical view of the verification problem.
