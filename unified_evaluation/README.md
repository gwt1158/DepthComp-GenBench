# DepthComp-GenBench unified evaluation

This package is the shared evaluation component of DepthComp-GenBench. It
contains the dataset loader, metrics, visualization, method adapters,
in-domain evaluation entries, and a canonical OOD-16 runner for the eight
retained depth completion methods.

Use an individual method entry with `--help`:

```bash
python -m unified_evaluation.evaluate_geolayernet --help
python -m unified_evaluation.evaluate_psd --help
```

Run the complete OOD protocol with:

```bash
python -m unified_evaluation.eval_ood16 --help
```

`eval_ood16` is the canonical entry. `eval_ood15` remains a compatibility entry
for explicitly labeled OOD-15 partial runs when ScanNet access has not been
approved; those results must never be reported as Macro-16. See the repository
root README for dataset downloads, directory layout, training, evaluation, and
acknowledgements.
