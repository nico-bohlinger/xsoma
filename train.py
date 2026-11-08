"""Train XSoma on Holosoma's G1 velocity-tracking task.

    python train.py --log_dir runs/xsoma            # 15 minutes of training (--minutes), then every checkpoint is evaluated
    python train.py --layout AL --log_dir runs/x4   # 2 or more GPUs: GPU 0 collects, the others update the policy (xsoma/multi_gpu.py)

Every option of xsoma/train.py's Config is a flag (--name value); the defaults are the published H100 configuration.
Checkpoints, log.jsonl, config.json and eval.jsonl go to --log_dir.
"""

import os


def runtime_defaults():
    """Environment variables that JAX reads when it starts, set unless the user already set them: JAX reserves 90 % of the GPU
    memory up front, and compiled programs carry no Python tracebacks (so the programs in the compilation cache are smaller)."""
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
    os.environ.setdefault("JAX_INCLUDE_FULL_TRACEBACKS_IN_LOCATIONS", "false")


if __name__ == "__main__":
    runtime_defaults()
    from xsoma.train import main  # imported only now, so that JAX starts with the settings above

    main()
