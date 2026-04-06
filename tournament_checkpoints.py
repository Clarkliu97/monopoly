from __future__ import annotations

import argparse

from monopoly.agent import CheckpointEvaluator


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a fixed-seed tournament across multiple Monopoly checkpoints.")
    parser.add_argument("checkpoints", nargs="+", type=str)
    parser.add_argument("--games", type=int, default=6)
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--seed-step", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    seeds = [args.seed + index * args.seed_step for index in range(args.games)]
    evaluator = CheckpointEvaluator(device=args.device)
    summary = evaluator.run_tournament(
        args.checkpoints,
        seeds=seeds,
        players_per_game=args.players,
        max_steps=args.max_steps,
    )
    print(
        f"games={summary.game_count} draws={summary.draw_count} avg_steps={summary.average_steps:.2f} "
        f"wins={summary.win_counts} avg_assets={summary.average_assets}"
    )
    for result in summary.results:
        print(
            f"seed={result.seed} lineup={result.lineup_labels} winner={result.winner_label or 'draw'} "
            f"steps={result.step_count} assets={result.player_assets}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())