"""
Modal app for running mini-vllm experiments on cloud GPUs.
Optimized for minimal cost and exact local environment replication.

Usage:
    modal run modal_app.py --gpu a10      # Run on A10 GPU
    modal run modal_app.py --gpu a100     # Run on A100 GPU
    modal run modal_app.py --gpu h100     # Run on H100 GPU
    modal run modal_app.py --experiment scheduler  # Run specific experiment
"""

import modal

# GPU configurations (updated to use string syntax)
GPU_CONFIG = {
    "a10": "A10G",
    "a100": "A100-40GB",
    "h100": "H100",
}

# Cost optimization: Build image with exact dependencies from pyproject.toml
# - Image is cached and only rebuilt when dependencies or code changes
# - No volume storage costs
# - Faster cold starts (everything pre-installed)
image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("uv")  # Install uv package manager
    .add_local_file("pyproject.toml", "/root/pyproject.toml", copy=True)
    .add_local_file("uv.lock", "/root/uv.lock", copy=True)  # Lock for reproducibility
    .workdir("/root")
    .run_commands(
        # Install exact dependencies from lockfile with GPU support
        "uv pip install --system torch>=2.8.0 transformers>=4.57.0",
        gpu="A10G",  # Ensures correct CUDA/PyTorch version
    )
    # Copy all code files (cached layer, rebuilds only if code changes)
    .add_local_dir("mini_vllm", "/root/mini_vllm", copy=True)
    .add_local_dir("experiments", "/root/experiments", copy=True)
)

app = modal.App("mini-vllm", image=image)


def _run_script(script_path: str, gpu_type: str):
    """Shared logic to run experiment scripts."""
    import subprocess
    import sys
    from pathlib import Path

    # Code is in /root (working directory)
    code_dir = Path("/root")
    sys.path.insert(0, str(code_dir))

    script_full_path = code_dir / script_path

    if not script_full_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    print(f"🚀 Running {script_path} on {gpu_type.upper()} GPU")
    print("=" * 60)

    # Run the script using Python from the container
    result = subprocess.run(
        [sys.executable, str(script_full_path)],
        cwd=str(code_dir),
        capture_output=False,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Script failed with code {result.returncode}")

    print("=" * 60)
    print("✓ Experiment completed successfully")


@app.function(gpu="A10G", timeout=1800)
def run_on_a10(script_path: str, gpu_type: str):
    """Run experiment on A10 GPU."""
    _run_script(script_path, gpu_type)


@app.function(gpu="A100-40GB", timeout=1800)
def run_on_a100(script_path: str, gpu_type: str):
    """Run experiment on A100 GPU."""
    _run_script(script_path, gpu_type)


@app.function(gpu="H100", timeout=1800)
def run_on_h100(script_path: str, gpu_type: str):
    """Run experiment on H100 GPU."""
    _run_script(script_path, gpu_type)


@app.local_entrypoint()
def main(
    gpu: str = "a10",
    script: str = None,
    experiment: str = None,
):
    """
    Run Python scripts on Modal GPUs.

    Args:
        gpu: GPU type (a10, a100, h100)
        script: Path to Python script to run (e.g., "experiments/my_script.py")
        experiment: Named experiment shortcut (hf_baseline_cache, scheduler, compare_scheduler)
                   Used only if --script is not provided
    """
    if gpu not in GPU_CONFIG:
        print(f"❌ Invalid GPU type: {gpu}")
        print(f"Available options: {', '.join(GPU_CONFIG.keys())}")
        return

    # Determine script path
    if script:
        # Direct script path provided
        script_path = script
        label = script_path
    elif experiment:
        # Named experiment shortcut
        experiments = {
            "hf_baseline_cache": "experiments/hf_baseline_cache.py",
            "scheduler": "mini_vllm/engine/scheduler.py",
            "compare_scheduler": "experiments/compare_scheduler.py",
        }
        if experiment not in experiments:
            print(f"❌ Unknown experiment: {experiment}")
            print(f"Available experiments: {', '.join(experiments.keys())}")
            return
        script_path = experiments[experiment]
        label = f"experiment '{experiment}'"
    else:
        # Default to hf_baseline_cache
        script_path = "experiments/hf_baseline_cache.py"
        label = "experiment 'hf_baseline_cache' (default)"

    print(f"\n🎯 Running: {label}")
    print(f"💻 GPU: {gpu.upper()}")
    print(f"📄 Script: {script_path}\n")

    # Select appropriate function based on GPU
    gpu_functions = {
        "a10": run_on_a10,
        "a100": run_on_a100,
        "h100": run_on_h100,
    }

    gpu_functions[gpu].remote(script_path, gpu)
