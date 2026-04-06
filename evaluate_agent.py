from __future__ import annotations

import argparse

from monopoly.agent import CheckpointEvaluator


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate one Monopoly checkpoint with deterministic fixed-seed self-play.")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--benchmark-opponents", nargs="*", default=None)
    parser.add_argument("--games", type=int, default=8)
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
    if args.benchmark_opponents:
        summary = evaluator.run_benchmark_suite(
            [args.checkpoint, *args.benchmark_opponents],
            seeds=seeds,
            players_per_game=args.players,
            max_steps=args.max_steps,
        )
        print(
            f"benchmark_games={summary.game_count} draws={summary.draw_count} avg_steps={summary.average_steps:.2f} "
            f"elo={summary.elo_ratings} cross_play={summary.cross_play_win_rates}"
        )
        for participant in summary.participants:
            print(
                f"label={participant.label} win_rate={participant.win_rate:.2%} placement={participant.average_placement:.2f} "
                f"avg_assets={participant.average_assets:.2f} asset_diff={participant.average_asset_differential:.2f} "
                f"rent_trend={participant.average_rent_potential_trend:.2f} denial={participant.average_monopoly_denial_events:.2f} "
                f"board_trend={participant.average_board_strength_trend:.2f} bid_quality={participant.average_auction_bid_quality:.2f} elo={participant.elo_rating:.2f}"
            )
        return 0
    summary = evaluator.evaluate_checkpoint(
        args.checkpoint,
        seeds=seeds,
        players_per_game=args.players,
        max_steps=args.max_steps,
    )
    print(
        f"checkpoint={summary.checkpoint_path} games={summary.game_count} wins={summary.win_count} "
        f"draws={summary.draw_count} avg_steps={summary.average_steps:.2f} avg_assets={summary.average_assets:.2f} "
        f"rent_trend={summary.average_rent_potential_trend:.2f} denial={summary.average_monopoly_denial_events:.2f} "
        f"board_trend={summary.average_board_strength_trend:.2f} bid_quality={summary.average_auction_bid_quality:.2f}"
    )
    for result in summary.results:
        print(
            f"seed={result.seed} winner={result.winner_label or 'draw'} steps={result.step_count} "
            f"assets={result.player_assets} rent_trend={result.rent_potential_trend:.2f} "
            f"denial={result.monopoly_denial_events:.2f} board_trend={result.board_strength_trend:.2f} "
            f"bid_quality={0.0 if result.auction_bid_quality_count <= 0 else result.auction_bid_quality_sum / float(result.auction_bid_quality_count):.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())