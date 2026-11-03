# AI Hiring Self-Description Toolkit

A privacy-preserving Python toolkit for auditing category-making and inequality
in multi-stage AI recruitment systems.

This repository is a public, synthetic-data-only software companion to the PhD
project:

> *Who Fits Where? Algorithmic Category-Making and Inequality in AI-Mediated
> Recruitment*

It demonstrates the reusable analysis and validation code without publishing
candidate records, job advertisements, model responses, frozen job pools, or
unpublished empirical results.

## What the toolkit demonstrates

- fail-closed validation of ranked Top-6 occupations and Top-10 jobs;
- matched source-category contrasts rather than job-level pseudo-replication;
- fixed Hardworking × Gender and Talent × Gender interaction definitions;
- paired t-tests, Wilcoxon tests, deterministic category-level bootstrap CIs,
  and Cohen's *d*<sub>z</sub>;
- Benjamini–Hochberg correction within explicitly defined hypothesis families;
- recommendation overlap using stable original job identities;
- missing-preserving annual-wage construction; and
- descriptive PoolValue–selection accounting with reconstruction checks.

The code also includes a deny-by-default adapter for hash-pinned, schema-approved
derived tables. It does not discover or load research files automatically.

## Research design represented in code

The synthetic fixtures encode six profiles:

| Code | Gender | Self-description |
|---|---|---|
| MB | Male | Baseline |
| MH | Male | Hardworking |
| MT | Male | Talented |
| FB | Female | Baseline |
| FH | Female | Hardworking |
| FT | Female | Talented |

They also encode four two-stage information architectures:

| Condition | Treatment enters Stage 1 | Structured résumé enters Stage 2 |
|---|---:|---:|
| 00 | No | No |
| 01 | No | Yes |
| 10 | Yes | No |
| 11 | Yes | Yes |

The toolkit preserves two interaction directions:

```text
Hardworking × Gender = (FH − FB) − (MH − MB)
Talent × Gender      = (FT − FB) − (MT − MB)
```

## Quick start

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -p 'test_*.py'
python scripts/run_synthetic_demo.py
```

The final command creates a deterministic, entirely synthetic inference table at
`examples/output/synthetic_inference.csv`.

Run the repository safety check with:

```bash
python scripts/check_public_repository.py
```

## Repository map

| Location | Purpose |
|---|---|
| `src/who_fits_where/` | Reusable validation and statistical modules |
| `tests/` | Unit tests and fully synthetic fixtures |
| `scripts/run_synthetic_demo.py` | Small end-to-end synthetic analysis |
| `scripts/check_public_repository.py` | Public-release safety checks |
| `docs/` | Method, architecture, and data-protection notes |
| `config/` | Empty, deny-by-default derived-data manifest example |

## Inferential safeguards

The empirical studies use matched source categories as the inferential unit.
Profile contrasts are constructed within category before aggregation. Job rows
and profile-pair rows are not treated as independent observations.

Missing wages remain missing. PoolValue–selection decomposition is descriptive
accounting, not causal mediation. A nonsignificant estimate is not interpreted
as proof of no effect.

## Data availability and scope

No real research data are included. In particular, this repository contains no:

- résumés, names, contact details, or candidate-level free text;
- raw recruitment-platform data or job advertisements;
- company names, job URLs, or original job descriptions;
- raw model prompts, requests, responses, or API logs;
- populated Stage 2 job pools; or
- unpublished result tables and Golden References.

The public toolkit is not an end-to-end replication package for the empirical
studies. It is a reviewed software excerpt designed to make the computational
methods inspectable and testable with synthetic inputs.

## Author

Developed by Ziyu Wang as part of doctoral research on AI-mediated recruitment.

## License

The source code and repository documentation are released under the MIT License.
No rights to restricted research data or third-party platform content are
granted or implied.
