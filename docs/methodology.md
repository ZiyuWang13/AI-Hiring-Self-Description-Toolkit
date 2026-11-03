# Methodological conventions

## Matched inference

The toolkit assumes that profiles within a source category are matched. Profile
contrasts are formed within each category, producing one effect per category.
Inference is performed on those category-level effects.

This avoids treating job recommendations or the 15 profile pairs as independent
inferential observations.

## Interaction directions

The fixed interaction definitions are:

```text
Hardworking × Gender = (FH − FB) − (MH − MB)
Talent × Gender      = (FT − FB) − (MT − MB)
```

A positive interaction means that the corresponding self-description effect is
more positive for women than for men. It does not imply that either component
effect is itself positive or statistically distinguishable from zero.

## Bootstrap and multiplicity

The bootstrap resamples complete category-level effects. Deterministic seeds are
supported for reproducibility. Benjamini–Hochberg adjustments are applied only
inside hypothesis families that callers define explicitly.

## Wages and decomposition

Annual wage is the monthly salary-range midpoint multiplied by the observed
salary-month factor. A blank factor uses 12 months; unparseable values remain
missing. Missing wages are not imputed.

The accounting identity is:

```text
Final Top10 wage = 60-job PoolValue + selection premium
```

This is a descriptive decomposition and should not be described as causal
mediation.

## Overlap

Cross-profile recommendation overlap uses stable original job identities. Local
J01–J60 display identifiers are not comparable across different pools.
