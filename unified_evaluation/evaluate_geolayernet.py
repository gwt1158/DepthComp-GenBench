"""Run GeoLayerNet through the unified evaluation pipeline."""

from unified_evaluation import runner
from unified_evaluation.adapters import geolayernet


def main():
    parser = runner.build_eval_parser(geolayernet)
    runner.run_eval(geolayernet, parser.parse_args())


if __name__ == "__main__":
    main()
