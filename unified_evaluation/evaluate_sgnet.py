"""Run SGNet through the unified evaluation pipeline."""

from unified_evaluation import runner
from unified_evaluation.adapters import sgnet


def main():
    parser = runner.build_eval_parser(sgnet)
    runner.run_eval(sgnet, parser.parse_args())


if __name__ == "__main__":
    main()
