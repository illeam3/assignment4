"""
test.py

Run alpha-beta-CROWN on the external Iris MLP model.
The model is stored in ONNX format, and the verification property is written in VNNLIB.
"""

import subprocess
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parent
    verifier_dir = project_root / "external" / "alpha-beta-CROWN" / "complete_verifier"
    config_path = project_root / "configs" / "iris_mlp.yaml"
    log_path = project_root / "results" / "iris_verification.log"

    if not verifier_dir.exists():
        raise FileNotFoundError(
            "alpha-beta-CROWN was not found at external/alpha-beta-CROWN. "
            "Please clone it before running this script."
        )

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "abcrown.py",
        "--config",
        "../../../configs/iris_mlp.yaml",
    ]

    print("Running alpha-beta-CROWN verification...")
    print("Verifier directory:", verifier_dir)
    print("Config file:", config_path)
    print("Log file:", log_path)
    print("Command:", " ".join(cmd))

    result = subprocess.run(
        cmd,
        cwd=verifier_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    print(result.stdout)
    log_path.write_text(result.stdout, encoding="utf-8")

    if result.returncode != 0:
        raise RuntimeError(f"alpha-beta-CROWN failed with return code {result.returncode}")

    if "Result: unsat" in result.stdout:
        print("Final verification result: UNSAT")
        print("No unsafe adversarial example exists in the specified perturbation region.")
    elif "Result: sat" in result.stdout:
        print("Final verification result: SAT")
        print("An unsafe adversarial example exists.")
    else:
        print("Verification finished. Please inspect results/iris_verification.log.")


if __name__ == "__main__":
    main()
