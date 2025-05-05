import subprocess
import sys
import os
from pathlib import Path
from tqdm import tqdm

# Constants
VENV_DIR = "venv"
BENCHMARK_SCRIPT = "src/benchmark/benchmark_model.py"
GOLD_STANDARD = "./data/annotations_15_04_2025.json"
MASKED_DIR = "./output/predictions/"
BENCHMARK_DIR = "./output/benchmarks/"
MODELS = ["DaAnonymization", "DaAnonymization_FG", "Gemma"]
USE_BERT_WEIGHTING = True

def activate_virtualenv():
    activate_script = Path(VENV_DIR) / "bin" / "activate_this.py"
    if not activate_script.exists():
        print(f"❌ Virtualenv activation script not found at: {activate_script}")
        sys.exit(1)
    exec(activate_script.read_text(), dict(__file__=str(activate_script)))

def run_model(model_name):
    command = [
        sys.executable,  # uses current Python interpreter
        BENCHMARK_SCRIPT,
        "--gold_standard_file", GOLD_STANDARD,
        "--masked_output_dir", MASKED_DIR,
        "--benchmark_output_dir", BENCHMARK_DIR,
        "--model", model_name,
    ]
    if USE_BERT_WEIGHTING:
        command.append("--bert_weighting")

    subprocess.run(command, check=True)

def main():
    # Check that virtualenv is activated
    if os.environ.get("VIRTUAL_ENV") is None:
        print("⚠️  Virtual environment is not active. Please activate it manually.")
        sys.exit(1)

    print("🔬 Benchmarking models...\n")

    for model in tqdm(MODELS, desc="Running benchmarks"):
        try:
            run_model(model)
        except subprocess.CalledProcessError as e:
            print(f"❌ Benchmark for model '{model}' failed:\n{e}")
            sys.exit(1)

    print("\n✅ All benchmarks completed.")

if __name__ == "__main__":
    main()
