from __future__ import annotations

import argparse
from collections import deque
import os
import shutil
import sys
from pathlib import Path

if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from tqdm import tqdm

from monopoly.agent import MonopolyActionSpace, ObservationEncoder, ParallelSelfPlayTrainer, PolicyConfig, TorchPolicyModel, TrainingConfig


DEFAULT_TRAINING_VALUES = {
    "iterations": 50,
    "episodes_per_thread": 16, 
    "max_steps": 400,
    "max_actions": 20000,
    "players": 4, 
    "learning_rate": 3e-4,
    "minimum_learning_rate": 3e-5,
    "gamma": 0.97,
    "gae_lambda": 0.95,
    "entropy_weight": 0.003,
    "model_type": "transformer",
    "transformer_embedding_size": 128,
    "transformer_heads": 8,
    "transformer_layers": 2,
    "clip_ratio": 0.2, 
    "ppo_epochs": 4, 
    "minibatch_size": 512, 
    "advantage_clip": 5.0,
    "hidden_size": [512, 512, 256, 256],
    "checkpoint_dir": ".checkpoints",
    "checkpoint_interval": 5,
    "seed": 7,
    "use_league_self_play": False,
    "league_snapshot_interval": 10,
    "league_recent_snapshot_count": 4,
    "league_use_heuristic_baseline": False,
    "league_use_scripted_opponents": True,
    "league_scripted_variants": [
        "conservative_liquidity_manager",
        "auction_value_shark",
        "expansionist_builder",
        "monopoly_denial_disruptor",
    ],
    "benchmark_interval": 10,
    "benchmark_games": 4,
    "benchmark_seed": 10000,
    "benchmark_seed_step": 1,
    "benchmark_max_steps": 400,
    "benchmark_players_per_game": 2,
}


class _TrainingDisplay:
    _PHASE_WEIGHTS = {
        "rollout": 55.0,
        "update": 35.0,
        "benchmark": 7.0,
        "checkpoint": 3.0,
    }
    _PHASE_ORDER = ("rollout", "update", "benchmark", "checkpoint")

    def __init__(self, total_iterations: int) -> None:
        self.total_iterations = total_iterations
        self.overall_bar: tqdm | None = None
        self.iteration_bar: tqdm | None = None
        self.summary_lines: list[tqdm] = []
        self._current_iteration_index: int | None = None
        self._recent_stats: deque = deque(maxlen=5)
        self._color = _AnsiColor(enabled=sys.stdout.isatty())

    def __enter__(self) -> _TrainingDisplay:
        self.overall_bar = tqdm(
            total=self.total_iterations,
            desc=self._color.cyan("Training"),
            unit="iter",
            position=0,
            dynamic_ncols=True,
        )
        self.iteration_bar = tqdm(
            total=100,
            desc=self._color.cyan("Iteration"),
            unit="%",
            position=1,
            leave=False,
            dynamic_ncols=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:.0f}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
        )
        for position in range(2, 15):
            line = tqdm(total=0, position=position, leave=True, bar_format="{desc}", dynamic_ncols=True)
            line.set_description_str("")
            self.summary_lines.append(line)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for line in self.summary_lines:
            line.close()
        if self.iteration_bar is not None:
            self.iteration_bar.close()
        if self.overall_bar is not None:
            self.overall_bar.close()

    def print_config(self, trainer: ParallelSelfPlayTrainer, iterations: int) -> None:
        config_rows = [
            ("Device", trainer.policy_config.device),
            ("Workers", str(trainer.training_config.worker_count)),
            ("Players", str(trainer.training_config.players_per_game)),
            ("Iterations", str(iterations)),
            ("Episodes/Worker", str(trainer.training_config.episodes_per_worker)),
            ("Max Steps", str(trainer.training_config.max_steps_per_episode)),
            ("Max Actions", str(trainer.training_config.max_actions_per_episode)),
            ("Model", trainer.policy_config.model_type),
            ("Auction Macro", str(trainer.training_config.use_auction_macro_steps)),
            ("Heuristics", str(trainer.policy_config.use_heuristic_bias)),
            ("League", str(trainer.training_config.use_league_self_play)),
            ("Benchmark Int", str(trainer.training_config.benchmark_interval)),
            ("Heuristic Scale", f"{trainer.current_heuristic_scale():.2f}"),
            ("Heuristic Sched", trainer.policy_config.heuristic_anneal_schedule),
            ("GAE", f"{trainer.policy_config.gae_lambda:.2f}"),
            ("Bootstrap", str(trainer.policy_config.bootstrap_truncated_episodes)),
        ]
        self.write(_render_key_value_table("Training Config", config_rows, columns=2))
        self._refresh_summary_block()

    def handle_status(self, update) -> None:
        if self.iteration_bar is None:
            return
        if self._current_iteration_index != update.iteration_index:
            self._current_iteration_index = update.iteration_index
            self.iteration_bar.reset(total=100)
            self.iteration_bar.set_description(self._color.cyan(f"Iteration {update.iteration_index + 1}/{update.total_iterations}"))
        progress = self._phase_progress(update.phase, update.completed, update.total)
        self.iteration_bar.n = progress
        self.iteration_bar.set_postfix_str(self._color.dim(update.message), refresh=False)
        self.iteration_bar.refresh()

    def complete_iteration(self, stat) -> None:
        if self.overall_bar is not None:
            self.overall_bar.update(1)
        if self.iteration_bar is not None:
            self.iteration_bar.n = self.iteration_bar.total
            self.iteration_bar.set_postfix_str(self._color.green("done"), refresh=False)
            self.iteration_bar.refresh()
        self._recent_stats.append(stat)
        self._refresh_summary_block()

    def write(self, message: str) -> None:
        if self.overall_bar is not None:
            self.overall_bar.write(message)
        else:
            print(message)

    def _phase_progress(self, phase: str, completed: int, total: int) -> float:
        completed_progress = 0.0
        for phase_name in self._PHASE_ORDER:
            weight = self._PHASE_WEIGHTS[phase_name]
            if phase_name == phase:
                ratio = 1.0 if total <= 0 else max(0.0, min(1.0, completed / float(total)))
                return completed_progress + weight * ratio
            completed_progress += weight
        return completed_progress

    def _refresh_summary_block(self) -> None:
        if not self.summary_lines:
            return
        lines = _render_rolling_summary_block(tuple(self._recent_stats), self._color)
        padded_lines = list(lines)
        while len(padded_lines) < len(self.summary_lines):
            padded_lines.append("")
        for line_bar, text in zip(self.summary_lines, padded_lines, strict=True):
            line_bar.set_description_str(text)
            line_bar.refresh()


class _AnsiColor:
    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        if not self.enabled:
            return text
        return f"\033[{code}m{text}\033[0m"

    def cyan(self, text: str) -> str:
        return self._wrap("36;1", text)

    def green(self, text: str) -> str:
        return self._wrap("32;1", text)

    def yellow(self, text: str) -> str:
        return self._wrap("33;1", text)

    def red(self, text: str) -> str:
        return self._wrap("31;1", text)

    def magenta(self, text: str) -> str:
        return self._wrap("35;1", text)

    def dim(self, text: str) -> str:
        return self._wrap("2", text)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Monopoly RL agent with parallel threaded self-play.")
    parser.add_argument("--iterations", type=int, default=DEFAULT_TRAINING_VALUES["iterations"])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--episodes-per-thread", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-actions", type=int, default=None)
    parser.add_argument("--players", type=int, default=DEFAULT_TRAINING_VALUES["players"])
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--minimum-learning-rate", type=float, default=None)
    parser.add_argument("--learning-rate-schedule", type=str, default="none", choices=("linear", "none"))
    parser.add_argument("--model-type", type=str, default=None, choices=("mlp", "transformer"))
    parser.add_argument("--transformer-embedding-size", type=int, default=None)
    parser.add_argument("--transformer-heads", type=int, default=None)
    parser.add_argument("--transformer-layers", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--gae-lambda", type=float, default=None)
    parser.add_argument("--bootstrap-truncated-episodes", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--entropy-weight", type=float, default=None)
    parser.add_argument("--heuristic-bias", action=argparse.BooleanOptionalAction, dest="use_heuristic_bias", default=None)
    parser.add_argument("--heuristic-anneal-schedule", type=str, default=None, choices=("linear", "none"))
    parser.add_argument("--heuristic-bias-start", type=float, default=None)
    parser.add_argument("--heuristic-bias-end", type=float, default=None)
    parser.add_argument("--heuristic-anneal-iterations", type=int, default=None)
    parser.add_argument("--clip-ratio", type=float, default=None)
    parser.add_argument("--ppo-epochs", type=int, default=None)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--advantage-clip", type=float, default=None)
    parser.add_argument("--hidden-size", type=int, nargs="+", default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--auction-macro-steps", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--league-self-play", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--league-snapshot-interval", type=int, default=None)
    parser.add_argument("--league-recent-snapshot-count", type=int, default=None)
    parser.add_argument("--league-use-heuristic-baseline", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--league-scripted-opponents", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--league-scripted-variants", nargs="+", default=None)
    parser.add_argument("--benchmark-interval", type=int, default=None)
    parser.add_argument("--benchmark-games", type=int, default=None)
    parser.add_argument("--benchmark-seed", type=int, default=None)
    parser.add_argument("--benchmark-seed-step", type=int, default=None)
    parser.add_argument("--benchmark-max-steps", type=int, default=None)
    parser.add_argument("--benchmark-players", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    encoder = ObservationEncoder()
    action_space = MonopolyActionSpace()
    resolved_device = args.device or _detect_default_device()
    resolved_threads = args.threads if args.threads is not None else _detect_default_worker_count(resolved_device)
    if args.resume is not None:
        trainer = ParallelSelfPlayTrainer.load_checkpoint(args.resume, device_override=resolved_device)
        if trainer.policy_model.observation_size != encoder.observation_size:
            raise ValueError(
                f"Checkpoint observation size {trainer.policy_model.observation_size} does not match the current encoder size {encoder.observation_size}. Start a fresh training run with the new schema."
            )
        if trainer.policy_model.action_count != action_space.action_count:
            raise ValueError(
                f"Checkpoint action count {trainer.policy_model.action_count} does not match the current action space size {action_space.action_count}. Start a fresh training run with the new schema."
            )
        if args.auction_macro_steps is not None and args.auction_macro_steps != trainer.training_config.use_auction_macro_steps:
            raise ValueError(
                "Cannot resume a checkpoint with a different auction macro-step setting. Start a fresh training run or resume with the original rollout mode."
            )
        _apply_runtime_defaults(trainer, worker_count=resolved_threads, device=resolved_device)
        _apply_resume_overrides(trainer, args)
        if args.hidden_size is not None and tuple(args.hidden_size) != trainer.policy_model.hidden_sizes:
            raise ValueError("Cannot change hidden layer sizes when resuming from an existing checkpoint.")
        if args.model_type is not None and args.model_type != trainer.policy_config.model_type:
            raise ValueError("Cannot change model type when resuming from an existing checkpoint.")
        if args.checkpoint_dir is not None:
            trainer.training_config.output_directory = args.checkpoint_dir
        print(f"loaded_checkpoint={Path(args.resume)}")
    else:
        policy_config = PolicyConfig(
            learning_rate=_value_or_default(args.learning_rate, "learning_rate"),
            minimum_learning_rate=_value_or_default(args.minimum_learning_rate, "minimum_learning_rate"),
            learning_rate_schedule=_value_or_default(args.learning_rate_schedule, "learning_rate_schedule"),
            model_type=_value_or_default(args.model_type, "model_type"),
            discount_gamma=_value_or_default(args.gamma, "gamma"),
            gae_lambda=_value_or_default(args.gae_lambda, "gae_lambda"),
            bootstrap_truncated_episodes=True if args.bootstrap_truncated_episodes is None else args.bootstrap_truncated_episodes,
            use_heuristic_bias=False if args.use_heuristic_bias is None else args.use_heuristic_bias,
            heuristic_anneal_schedule="linear" if args.heuristic_anneal_schedule is None else args.heuristic_anneal_schedule,
            heuristic_bias_start=1.0 if args.heuristic_bias_start is None else args.heuristic_bias_start,
            heuristic_bias_end=0.0 if args.heuristic_bias_end is None else args.heuristic_bias_end,
            heuristic_anneal_iterations=100 if args.heuristic_anneal_iterations is None else args.heuristic_anneal_iterations,
            entropy_weight=_value_or_default(args.entropy_weight, "entropy_weight"),
            advantage_clip=_value_or_default(args.advantage_clip, "advantage_clip"),
            seed=_value_or_default(args.seed, "seed"),
            hidden_sizes=tuple(int(size) for size in _value_or_default(args.hidden_size, "hidden_size")),
            transformer_embedding_size=_value_or_default(args.transformer_embedding_size, "transformer_embedding_size"),
            transformer_heads=_value_or_default(args.transformer_heads, "transformer_heads"),
            transformer_layers=_value_or_default(args.transformer_layers, "transformer_layers"),
            ppo_clip_ratio=_value_or_default(args.clip_ratio, "clip_ratio"),
            ppo_epochs=_value_or_default(args.ppo_epochs, "ppo_epochs"),
            minibatch_size=_value_or_default(args.minibatch_size, "minibatch_size"),
            device=resolved_device,
        )
        training_config = TrainingConfig(
            worker_count=resolved_threads,
            episodes_per_worker=_value_or_default(args.episodes_per_thread, "episodes_per_thread"),
            max_steps_per_episode=_value_or_default(args.max_steps, "max_steps"),
            max_actions_per_episode=_value_or_default(args.max_actions, "max_actions"),
            players_per_game=_value_or_default(args.players, "players"),
            checkpoint_interval=_value_or_default(args.checkpoint_interval, "checkpoint_interval"),
            output_directory=_value_or_default(args.checkpoint_dir, "checkpoint_dir"),
            use_auction_macro_steps=True if args.auction_macro_steps is None else args.auction_macro_steps,
            use_league_self_play=_value_or_default(args.league_self_play, "use_league_self_play"),
            league_snapshot_interval=_value_or_default(args.league_snapshot_interval, "league_snapshot_interval"),
            league_recent_snapshot_count=_value_or_default(args.league_recent_snapshot_count, "league_recent_snapshot_count"),
            league_use_heuristic_baseline=_value_or_default(args.league_use_heuristic_baseline, "league_use_heuristic_baseline"),
            league_use_scripted_opponents=_value_or_default(args.league_scripted_opponents, "league_use_scripted_opponents"),
            league_scripted_variants=tuple(str(value) for value in _value_or_default(args.league_scripted_variants, "league_scripted_variants")),
            benchmark_interval=_value_or_default(args.benchmark_interval, "benchmark_interval"),
            benchmark_games=_value_or_default(args.benchmark_games, "benchmark_games"),
            benchmark_seed=_value_or_default(args.benchmark_seed, "benchmark_seed"),
            benchmark_seed_step=_value_or_default(args.benchmark_seed_step, "benchmark_seed_step"),
            benchmark_max_steps=_value_or_default(args.benchmark_max_steps, "benchmark_max_steps"),
            benchmark_players_per_game=_value_or_default(args.benchmark_players, "benchmark_players_per_game"),
        )
        model = TorchPolicyModel(
            observation_size=encoder.observation_size,
            action_count=action_space.action_count,
            seed=policy_config.seed,
            hidden_sizes=policy_config.hidden_sizes,
            device=policy_config.device,
            model_type=policy_config.model_type,
            transformer_embedding_size=policy_config.transformer_embedding_size,
            transformer_heads=policy_config.transformer_heads,
            transformer_layers=policy_config.transformer_layers,
            input_layout=encoder.observation_layout,
        )
        trainer = ParallelSelfPlayTrainer(
            policy_model=model,
            training_config=training_config,
            policy_config=policy_config,
        )

    stats: list = []
    with _TrainingDisplay(args.iterations) as display:
        display.print_config(trainer, args.iterations)

        def _on_iteration_complete(stat) -> None:
            stats.append(stat)
            display.complete_iteration(stat)

        trainer.train(
            args.iterations,
            progress_callback=_on_iteration_complete,
            status_callback=display.handle_status,
            show_update_progress=False,
        )

    checkpoint_dir = Path(trainer.training_config.output_directory)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_checkpoint = checkpoint_dir / "latest.pt"
    trainer.save_checkpoint(final_checkpoint)

    print(_render_key_value_table("Run Complete", [("Iterations", str(len(stats))), ("Checkpoint", str(final_checkpoint))], columns=1))
    print(f"saved_checkpoint={final_checkpoint}")
    return 0


def _value_or_default(value: object, key: str):
    if value is not None:
        return value
    return DEFAULT_TRAINING_VALUES[key]


def _format_source_mix(source_weights: dict[str, float]) -> str:
    if not source_weights:
        return "none"
    return "/".join(
        (
            f"b:{source_weights.get('best', 0.0):.2f}",
            f"r:{source_weights.get('recent', 0.0):.2f}",
            f"s:{source_weights.get('scripted', 0.0):.2f}",
        )
    )


def _render_key_value_table(title: str, rows: list[tuple[str, str]], *, columns: int = 2) -> str:
    del columns
    terminal_width = min(max(72, shutil.get_terminal_size((96, 24)).columns), 96)
    inner_width = terminal_width - 4
    key_width = min(18, max(10, inner_width // 4))
    value_width = max(12, inner_width - key_width - 3)
    lines = [f"+{'-' * (terminal_width - 2)}+", f"| {title:<{inner_width}} |", f"+{'-' * (terminal_width - 2)}+"]
    for key, value in rows:
        rendered_value = str(value)
        if len(rendered_value) > value_width:
            rendered_value = rendered_value[: value_width - 3] + "..."
        lines.append(f"| {key:<{key_width}} {rendered_value:<{value_width}} |")
    lines.append(f"+{'-' * (terminal_width - 2)}+")
    return "\n".join(lines)


def _render_rolling_summary_block(stats: tuple, color: _AnsiColor) -> list[str]:
    border = color.dim("+----+----+------+--------+--------+-------+-------+")
    header = color.cyan("|Itr | Ep |  Ex  | Reward | St/Act | Roll  |  Upd  |")
    lines = [color.magenta("Recent Iterations (last 5)"), border, header, border]
    if not stats:
        lines.append(color.dim("No completed iterations yet."))
        return lines
    for stat in stats:
        reward = _color_metric(f"{stat.average_total_reward:.3f}", stat.average_total_reward, color)
        primary = (
            f"|{stat.iteration_index + 1:>4}|{stat.episode_count:>4}|{stat.example_count:>6}|"
            f"{reward:>8}|{f'{stat.average_steps:.0f}/{stat.average_raw_actions:.0f}':>8}|"
            f"{f'{stat.rollout_seconds:.2f}s':>7}|{f'{stat.update_seconds:.2f}s':>7}|"
        )
        bench_wr = _color_metric(f"{stat.benchmark_current_win_rate:.1%}", stat.benchmark_current_win_rate, color, neutral_threshold=0.001)
        elo = _color_metric(f"{stat.benchmark_current_elo:.0f}", stat.benchmark_current_elo - 1000.0, color, neutral_threshold=1.0)
        secondary = (
            f"  {color.dim('bench')} {bench_wr:<8}  {color.dim('elo')} {elo:<7}  "
            f"{color.dim('src')} {_color_source_mix(_format_source_mix(stat.league_source_weights), color):<18}  "
            f"{color.dim('denial')} {stat.average_monopoly_denial_events:.2f}  "
            f"{color.dim('bid')} {stat.average_auction_bid_quality:.2f}"
        )
        lines.append(primary)
        lines.append(secondary)
    return lines


def _color_metric(text: str, value: float, color: _AnsiColor, *, neutral_threshold: float = 0.0) -> str:
    if value > neutral_threshold:
        return color.green(text)
    if value < -neutral_threshold:
        return color.red(text)
    return color.yellow(text)


def _color_source_mix(source_mix: str, color: _AnsiColor) -> str:
    if source_mix == "none":
        return color.dim(source_mix)
    return color.cyan(source_mix)


def _detect_default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _detect_default_worker_count(device: str) -> int:
    logical_cores = max(1, os.cpu_count() or 1)
    normalized_device = device.lower()

    if logical_cores <= 2:
        return 1

    if normalized_device == "cuda":
        if logical_cores <= 6:
            return max(1, logical_cores - 2)
        return max(2, min(8, (logical_cores * 2) // 3))

    if logical_cores <= 8:
        return max(1, logical_cores - 1)
    return max(2, min(12, logical_cores - max(2, logical_cores // 4)))


def _apply_runtime_defaults(trainer: ParallelSelfPlayTrainer, *, worker_count: int, device: str) -> None:
    trainer.training_config.worker_count = worker_count
    trainer.policy_config.device = device


def _apply_resume_overrides(trainer: ParallelSelfPlayTrainer, args: argparse.Namespace) -> None:
    if args.threads is not None:
        trainer.training_config.worker_count = args.threads
    if args.episodes_per_thread is not None:
        trainer.training_config.episodes_per_worker = args.episodes_per_thread
    if args.max_steps is not None:
        trainer.training_config.max_steps_per_episode = args.max_steps
    if args.max_actions is not None:
        trainer.training_config.max_actions_per_episode = args.max_actions
    if args.players is not None:
        trainer.training_config.players_per_game = args.players
    if args.checkpoint_interval is not None:
        trainer.training_config.checkpoint_interval = args.checkpoint_interval
    if args.learning_rate is not None:
        trainer.policy_config.learning_rate = args.learning_rate
        for param_group in trainer._optimizer.param_groups:
            param_group["lr"] = args.learning_rate
    if args.gamma is not None:
        trainer.policy_config.discount_gamma = args.gamma
    if args.gae_lambda is not None:
        trainer.policy_config.gae_lambda = args.gae_lambda
    if args.bootstrap_truncated_episodes is not None:
        trainer.policy_config.bootstrap_truncated_episodes = args.bootstrap_truncated_episodes
    if args.minimum_learning_rate is not None:
        trainer.policy_config.minimum_learning_rate = args.minimum_learning_rate
    if args.learning_rate_schedule is not None:
        trainer.policy_config.learning_rate_schedule = args.learning_rate_schedule
    if args.model_type is not None:
        trainer.policy_config.model_type = args.model_type
    if args.transformer_embedding_size is not None:
        trainer.policy_config.transformer_embedding_size = args.transformer_embedding_size
    if args.transformer_heads is not None:
        trainer.policy_config.transformer_heads = args.transformer_heads
    if args.transformer_layers is not None:
        trainer.policy_config.transformer_layers = args.transformer_layers
    if args.entropy_weight is not None:
        trainer.policy_config.entropy_weight = args.entropy_weight
    if args.use_heuristic_bias is not None:
        trainer.policy_config.use_heuristic_bias = args.use_heuristic_bias
    if args.heuristic_anneal_schedule is not None:
        trainer.policy_config.heuristic_anneal_schedule = args.heuristic_anneal_schedule
    if args.heuristic_bias_start is not None:
        trainer.policy_config.heuristic_bias_start = args.heuristic_bias_start
    if args.heuristic_bias_end is not None:
        trainer.policy_config.heuristic_bias_end = args.heuristic_bias_end
    if args.heuristic_anneal_iterations is not None:
        trainer.policy_config.heuristic_anneal_iterations = args.heuristic_anneal_iterations
    if args.advantage_clip is not None:
        trainer.policy_config.advantage_clip = args.advantage_clip
    if args.clip_ratio is not None:
        trainer.policy_config.ppo_clip_ratio = args.clip_ratio
    if args.ppo_epochs is not None:
        trainer.policy_config.ppo_epochs = args.ppo_epochs
    if args.minibatch_size is not None:
        trainer.policy_config.minibatch_size = args.minibatch_size
    if args.device is not None:
        trainer.policy_config.device = args.device
    if args.seed is not None:
        trainer.policy_config.seed = args.seed
    if args.auction_macro_steps is not None:
        trainer.training_config.use_auction_macro_steps = args.auction_macro_steps
    if args.league_self_play is not None:
        trainer.training_config.use_league_self_play = args.league_self_play
    if args.league_snapshot_interval is not None:
        trainer.training_config.league_snapshot_interval = args.league_snapshot_interval
    if args.league_recent_snapshot_count is not None:
        trainer.training_config.league_recent_snapshot_count = args.league_recent_snapshot_count
    if args.league_use_heuristic_baseline is not None:
        trainer.training_config.league_use_heuristic_baseline = args.league_use_heuristic_baseline
    if args.league_scripted_opponents is not None:
        trainer.training_config.league_use_scripted_opponents = args.league_scripted_opponents
    if args.league_scripted_variants is not None:
        trainer.training_config.league_scripted_variants = tuple(str(value) for value in args.league_scripted_variants)
    if args.benchmark_interval is not None:
        trainer.training_config.benchmark_interval = args.benchmark_interval
    if args.benchmark_games is not None:
        trainer.training_config.benchmark_games = args.benchmark_games
    if args.benchmark_seed is not None:
        trainer.training_config.benchmark_seed = args.benchmark_seed
    if args.benchmark_seed_step is not None:
        trainer.training_config.benchmark_seed_step = args.benchmark_seed_step
    if args.benchmark_max_steps is not None:
        trainer.training_config.benchmark_max_steps = args.benchmark_max_steps
    if args.benchmark_players is not None:
        trainer.training_config.benchmark_players_per_game = args.benchmark_players
    trainer.refresh_league_manager()


if __name__ == "__main__":
    raise SystemExit(main())