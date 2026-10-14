"""Run PSD-D through the unified evaluation pipeline."""

from unified_evaluation import runner
from unified_evaluation.adapters import psd


def main():
    parser = runner.build_eval_parser(psd)
    runner.run_eval(psd, parser.parse_args())


if __name__ == "__main__":
    main()
