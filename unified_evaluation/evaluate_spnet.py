"""Run SPNet through the unified evaluation pipeline."""

from unified_evaluation import runner
from unified_evaluation.adapters import spnet


def main():
    parser = runner.build_eval_parser(spnet)
    runner.run_eval(spnet, parser.parse_args())


if __name__ == "__main__":
    main()
