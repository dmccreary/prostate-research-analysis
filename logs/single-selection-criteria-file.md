# Log: Consolidate Selection Criteria into a Single File

**Date:** 2026-09-08

## Summary

Consolidated the three separate selection-criteria documents into a single [selection-criteria.md](../docs/selection-criteria.md) file.

## Changes

- Created `docs/selection-criteria.md`, merging content from:
  - `docs/summary-criteria.md` (Treatment Modality Key, Acceptance/Rejection Key Code)
  - `docs/inclusion-criteria.md` (Study Inclusion Criteria)
  - `docs/exclusion-criteria.md` (Study Exclusion Criteria)
- Added a new "Abstract Pre-Filtering" section documenting the abstract-level text filters:
  - Excludes abstracts containing "palliative", "metastatic", or "hormone resistant"
  - Requires "prostate cancer", "prostate neoplasia", or "prostate carcinoma" to appear
- Removed the old files: `docs/exclusion-criteria.md`, `docs/inclusion-criteria.md`, `docs/summary-criteria.md`
- Updated `mkdocs.yml` nav: replaced the three separate nav entries ("Summary Criteria", "Exclusion Criteria", "Inclusion Criteria") with a single "Selection Criteria: selection-criteria.md" entry
- Updated cross-references to point at the new file:
  - `docs/about.md` — link to rules now points to `selection-criteria.md`
  - `docs/sims/ml-workflow/index.md` — inclusion/exclusion criteria links merged into a single link to `selection-criteria.md`
  - `CLAUDE.md` — reference to treatment modality/acceptance criteria file updated to `docs/selection-criteria.md`

## Rationale

The three files had significant overlap (treatment modality list duplicated in all three) and had drifted apart in wording. A single source of truth reduces maintenance burden and avoids inconsistency between inclusion/exclusion phrasing.
