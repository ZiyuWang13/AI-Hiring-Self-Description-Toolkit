# Synthetic example

Run:

```bash
python scripts/run_synthetic_demo.py
```

The script generates deterministic artificial profile wages, constructs
within-category contrasts, performs paired category-level inference and
bootstrap estimation, applies BH-FDR, and writes a schema-controlled table to
`examples/output/synthetic_inference.csv`.

The example does not contain or approximate empirical study results.
