# Software architecture

The toolkit separates design validation, outcome construction, statistical
inference, and presentation schemas.

```text
synthetic or approved derived records
                |
                v
        validation.py
                |
       +--------+---------+
       |                  |
       v                  v
   salary.py          overlap.py
       |                  |
       +--------+---------+
                v
          contrasts.py
                |
                v
          inference.py
                |
       +--------+---------+
       |                  |
       v                  v
 multiplicity.py    decomposition.py
       |                  |
       +--------+---------+
                v
        result_tables.py
```

`derived_adapter.py` is intentionally separate. It only accepts exact,
hash-pinned, schema-approved CSV inputs and rejects restricted column names,
symlinks, path traversal, and incomplete pairing.

No module performs network access or launches a model experiment.
