"""Run CompletionFormer through the unified evaluation pipeline."""

from unified_evaluation import runner
from unified_evaluation.adapters import completionformer


def main():
    parser = runner.build_eval_parser(completionformer)
    runner.run_eval(completionformer, parser.parse_args())


if __name__ == "__main__":
    main()
