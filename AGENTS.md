# AGENTS.md

These rules apply to all human and AI-assisted work in this public toolkit.

## Public-data boundary

- Use only synthetic fixtures or explicitly approved public data.
- Never add résumés, personal identifiers, job advertisements, company names,
  platform URLs, raw model requests/responses, populated job pools, credentials,
  unpublished results, or local research-archive paths.
- Do not create symlinks to private research archives.
- The default derived-data manifest must remain deny-by-default.

## Research definitions

- Preserve the six profiles MB, MH, MT, FB, FH, and FT.
- Preserve conditions 00, 01, 10, and 11.
- Construct profile contrasts inside matched source categories before inference.
- Hardworking × Gender is `(FH - FB) - (MH - MB)`.
- Talent × Gender is `(FT - FB) - (MT - MB)`.
- Never impute missing wages in the approved salary helpers.
- Treat PoolValue–selection decomposition as descriptive accounting, not causal
  mediation.

## Engineering workflow

- Inspect the existing implementation and tests before changing definitions.
- Add or update synthetic tests for every behavioral change.
- Keep functions deterministic where a seed is supplied.
- Do not add API clients, network calls, experiment runners, or hidden data
  downloads.
- Run the unit tests, synthetic demo, and public-repository safety check before
  committing.
- Do not change definitions or expected values merely to make tests pass.
