from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT

doc = SimpleDocTemplate(
    "report.pdf",
    pagesize=A4,
    rightMargin=0.65 * inch,
    leftMargin=0.65 * inch,
    topMargin=0.65 * inch,
    bottomMargin=0.65 * inch,
)

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="MyTitle",
    parent=styles["Title"],
    fontSize=16,
    leading=20,
    alignment=TA_LEFT,
    spaceAfter=10,
))
styles.add(ParagraphStyle(
    name="MyHeading",
    parent=styles["Heading2"],
    fontSize=11.5,
    leading=14,
    spaceBefore=8,
    spaceAfter=4,
))
styles.add(ParagraphStyle(
    name="MyBody",
    parent=styles["BodyText"],
    fontSize=9.5,
    leading=12.5,
    spaceAfter=6,
))
styles.add(ParagraphStyle(
    name="MyCode",
    parent=styles["BodyText"],
    fontName="Courier",
    fontSize=8.5,
    leading=10.5,
    leftIndent=12,
    spaceAfter=6,
))

story = []

def h(text):
    story.append(Paragraph(text, styles["MyHeading"]))

def p(text):
    story.append(Paragraph(text, styles["MyBody"]))

def code(text):
    story.append(Paragraph(text.replace("\n", "<br/>"), styles["MyCode"]))

story.append(Paragraph("Assignment 4 Report: Neural Network Verification with alpha-beta-CROWN", styles["MyTitle"]))

h("1. Overview")
p("In this assignment, I explored alpha-beta-CROWN and used it to verify a neural network robustness property. The project includes an external neural network model, an ONNX model file, a VNNLIB specification, a YAML configuration file, a reproducible test script, and verification results. The main goal was to check whether a small neural network remains locally robust under a bounded input perturbation.")

h("2. alpha-beta-CROWN Structure")
p("The alpha-beta-CROWN project is organized around a complete verifier. The main executable used in this assignment is abcrown.py inside complete_verifier. Verification settings are controlled through YAML files. Important YAML sections include general, model, data, specification, solver, attack, and bab. The model section points to a PyTorch or ONNX model, the specification section points to the VNNLIB property, and solver/bab configure alpha-CROWN, beta-CROWN, and branch-and-bound behavior.")

p("I first tested the provided CIFAR convolutional example to confirm that the verifier could load a model, read a configuration file, run PGD attack, and produce verification statuses such as unsafe-pgd and safe-incomplete. Since the built-in CIFAR configuration targets many samples and is expensive on CPU, I used it only as an installation and output-format check.")

h("3. External Model and Dataset")
p("For the external model, I used the Iris dataset from scikit-learn. The Iris dataset has four numerical input features and three output classes. I trained a small multi-layer perceptron and exported it to ONNX. The model architecture is:")

code("Linear(4, 16)\nReLU\nLinear(16, 16)\nReLU\nLinear(16, 3)")

p("The model was trained in create_iris_model.py. The final ONNX model is saved as models/iris_mlp.onnx. The selected test sample and preprocessing information are saved in data/iris_test_sample.npz. The ONNX model was checked successfully with ONNX checker and uses opset 12.")

h("4. Verification Property")
p("The verification property checks local robustness for one selected Iris test sample. The input is allowed to vary inside an L-infinity perturbation box with epsilon = 0.05. The VNNLIB file encodes the unsafe condition: another class output becomes greater than or equal to the originally predicted class output. Therefore, if the verifier returns UNSAT, it means that no adversarial input exists inside the perturbation region.")

p("The unsafe condition was encoded as a disjunction over the non-predicted classes. For example, if class 0 is the predicted class, the VNNLIB property checks whether either Y_1 >= Y_0 or Y_2 >= Y_0 can occur within the input bounds. This follows the standard adversarial specification style: the verifier searches for a counterexample, and UNSAT means the robustness property holds.")

h("5. Verification Configuration")
p("The YAML configuration file is configs/iris_mlp.yaml. It uses CPU execution, the ONNX model path, the VNNLIB specification path, alpha-CROWN bound propagation, beta-CROWN optimization, and branch-and-bound as the complete verifier. The run can be reproduced by executing test.py or by directly running abcrown.py with the YAML configuration.")

code("python test.py\n\ncd external/alpha-beta-CROWN/complete_verifier\npython abcrown.py --config ../../../configs/iris_mlp.yaml")

h("6. Result and Interpretation")
p("For the selected Iris sample with epsilon = 0.05, alpha-beta-CROWN returned:")

code("PGD attack failed\nTotal number of violation: 0\nVerified with initial CROWN!\nResult: unsat")

p("The result UNSAT means that the unsafe adversarial condition is unsatisfiable. In other words, within the specified L-infinity perturbation box, there is no input that makes another class output greater than or equal to the predicted class output. Therefore, the model is verified to be locally robust for this sample and epsilon value. The verification finished quickly on CPU because the Iris MLP is small and only one input instance was verified.")

h("7. Comparison with Marabou")
p("Compared with Marabou, alpha-beta-CROWN is more specialized for neural network robustness verification using bound propagation and branch-and-bound. It supports methods such as CROWN, alpha-CROWN, beta-CROWN, PGD attack, and BaB splitting. This makes it convenient for ONNX plus VNNLIB robustness experiments. Marabou, on the other hand, provides a more explicit SMT-style view of neural network verification by reasoning over piecewise-linear constraints.")

p("In my previous Marabou-based workflow, the verification problem felt closer to manually constructing input bounds and output constraints. In alpha-beta-CROWN, the YAML and VNNLIB format made the experiment more standardized. However, alpha-beta-CROWN was also sensitive to configuration paths and VNNLIB syntax, so debugging required careful checking of relative paths, output-condition direction, and parser-compatible specification format.")

h("8. Strengths and Limitations")
p("The main strength of alpha-beta-CROWN is that it can combine fast incomplete methods, adversarial search, and complete branch-and-bound verification. In this experiment, PGD failed to find a violation, and the initial CROWN bound was already strong enough to prove the property. This shows that alpha-beta-CROWN can efficiently certify local robustness for small neural networks.")

p("The main limitation is that verification can become expensive for larger models, larger input dimensions, or larger perturbation radii. The built-in CIFAR example was much slower on CPU, especially when many samples were selected. Another limitation is that a single verified sample does not prove global robustness of the entire model. It only proves local robustness for the selected input and epsilon. For stronger evidence, more samples and multiple epsilon values should be verified.")

h("9. Conclusion")
p("This assignment successfully verified an external Iris MLP model using alpha-beta-CROWN. The model was exported to ONNX, the robustness property was encoded in VNNLIB, and the verification was executed through a YAML configuration and test.py. The final result was UNSAT for the unsafe condition at L-infinity epsilon = 0.05, meaning the selected sample is locally robust under the tested perturbation bound.")

doc.build(story)
print("Saved report.pdf")
