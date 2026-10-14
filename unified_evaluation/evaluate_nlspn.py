"""Run NLSPN through the unified evaluation pipeline."""

from unified_evaluation import runner
from unified_evaluation.adapters import nlspn


def main():
    parser = runner.build_eval_parser(nlspn)
    runner.run_eval(nlspn, parser.parse_args())


if __name__ == "__main__":
    main()
