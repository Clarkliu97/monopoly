# Development Guide

This file is the short contributor-oriented companion to [README.md](README.md). It focuses on the commands and checks you are likely to use while changing code.

## Environment Setup

1. Create and activate a Python 3.11+ environment.
2. Install PyTorch first so you can choose the correct CPU or CUDA build.
3. Install the remaining dependencies with `requirements.txt`.

CPU-only example:

```bash
python -m pip install torch
python -m pip install -r requirements.txt
```

CUDA example:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

## Common Commands

Launch the GUI:

```bash
python main.py
```

Run the full test suite with coverage:

```bash
python run_tests.py
```

Run all pytest tests directly:

```bash
python -m pytest
```

Run focused gameplay tests:

```bash
python -m pytest tests/test_game.py tests/test_interactive.py
```

Run focused agent tests:

```bash
python -m pytest tests/test_agent.py -k action_space
python -m pytest tests/test_environment.py tests/test_worker_pool.py
```

Train from scratch on CPU:

```bash
python train_agent.py --iterations 20 --device cpu --threads 2 --plain_output
```

Resume training:

```bash
python train_agent.py --resume .checkpoints/latest.pt --iterations 20 --plain_output
```

Run a long training job under Ubuntu with `nohup`:

```bash
nohup python train_agent.py --iterations 200 --device cpu --threads 4 --plain_output > train.log 2>&1 &
tail -f train.log
```

Evaluate a checkpoint:

```bash
python evaluate_agent.py .checkpoints/latest.pt --games 8 --players 2 --device cpu
```

Run a checkpoint tournament:

```bash
python tournament_checkpoints.py .checkpoints/iteration_0020.pt .checkpoints/iteration_0040.pt .checkpoints/latest.pt
```

## Debugging Tips

- Set `DEBUG_MODE = True` in `main.py` to enable the debug editor in the GUI.
- Frontend and backend logs are written to `logs/frontend.log` and `logs/backend.log`.
- Use `MONOPOLY_LOG_LEVEL=DEBUG` when you need detailed socket, backend, or GUI logging.
- Saved checkpoints are written under `.checkpoints/` unless overridden with `--checkpoint-dir`.
- Use `--plain_output` for non-interactive training runs so stdout stays line-oriented and `tqdm` does not clutter log files.

## High-Value Test Targets

- `tests/test_game.py`: engine turn flow, serialization, and trade bookkeeping
- `tests/test_interactive.py`: pending actions and interactive execution paths
- `tests/test_api.py`: serialized state contract
- `tests/test_agent.py`: RL pipeline integration and CLI entry points
- `tests/test_environment.py`: self-play environment helpers
- `tests/test_worker_pool.py`: rollout worker orchestration
- `tests/test_pygame_frontend.py`: frontend controller and UI support behavior

## Generated Artifacts

These are local-only outputs and are already ignored by `.gitignore`:

- `.checkpoints/`
- `logs/`
- `.coverage`
- `.pytest_cache/`
- `__pycache__/`

## Architecture Quick Reference

- `src/monopoly/game.py`: authoritative game state machine
- `src/monopoly/gui/backend_process.py`: backend runtime and online lobby logic
- `src/monopoly/gui/transport.py`: socket messaging layer
- `src/monopoly/gui/rendezvous.py`: lobby discovery service
- `src/monopoly/gui/pygame_frontend/`: GUI application and rendering
- `src/monopoly/agent/`: RL training, evaluation, scripted agents, and checkpoints