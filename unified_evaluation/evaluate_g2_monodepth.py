"""Run G2-MonoDepth through the unified evaluation pipeline."""

from unified_evaluation import runner
from unified_evaluation.adapters import g2_monodepth


def main():
    parser = runner.build_eval_parser(g2_monodepth)
    runner.run_eval(g2_monodepth, parser.parse_args())


if __name__ == "__main__":
    main()
