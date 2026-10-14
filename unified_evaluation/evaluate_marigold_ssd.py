"""Run Marigold-SSD through the unified evaluation pipeline."""

from unified_evaluation import runner
from unified_evaluation.adapters import marigold_ssd


def main():
    parser = runner.build_eval_parser(marigold_ssd)
    runner.run_eval(marigold_ssd, parser.parse_args())


if __name__ == "__main__":
    main()
