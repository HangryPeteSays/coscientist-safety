# REPOL: Risk Evaluation of Pharmacology by LLMs

A coscientist-safety testing platform that evaluates AI biosafety blind spots in drug discovery, with a focus on novel mechanistic risk that requires domain-expert calibration to detect.

**Author:** Peter-James H. Zushin, Ph.D. — Postdoctoral Fellow, Wu Lab, Stanford University
**Status:** Phase 2 (April 2026) — infrastructure built, capability-ladder pilot in progress
**Contact:** petezushin@gmail.com

---

## What this project is

Frontier AI labs are building drug-discovery coscientists at increasing capability. The standard biosafety evaluations for these systems are run on the model's own canonical safety axes — categories the model has been trained to flag. This works well for in-distribution risk and degrades on the regime that matters most: novel mechanistic risk that routes through specialized pathway-level reasoning, where the dangerous signal is not in any flag category but in domain-specific biology a subject matter expert would catch.

REPOL is built to surface exactly that gap. It runs a Generator → Critic → Ranker pipeline modeled on the Google DeepMind coscientist architecture, scaled down for tractable iteration, and applies it to cardiac antiarrhythmic drug discovery — a domain where the line between therapeutic and proarrhythmic mechanism is unusually thin. The pipeline's outputs are then scored by a domain expert (the author) against a pre-registered scoring rubric, with per-agent decomposition that identifies where in the pipeline safety reasoning succeeds or fails.

## The anchor finding

In Phase 1 (March 2026), running on Claude Sonnet 4.6, the pipeline proposed **SCN5A augmentation as an antiarrhythmic strategy** — a mechanism that would augment the late sodium current (INa-L), prolong cardiac action potential duration, and produce Torsades de Pointes (TdP), the precise lethal arrhythmia the compound was meant to treat. Neither the Critic nor the Ranker agent flagged this. The Phase 1 audit log capturing this run is preserved in [`breadcrumbs/audit_log_20260318_122119.txt`](breadcrumbs/audit_log_20260318_122119.txt).

This is the failure mode REPOL is designed to characterize: not a generic safety violation but a domain-specific mechanistic inversion that a cardiologist would catch immediately and a frontier model's own safety reasoning did not. The thesis is that detecting these failures requires domain-calibrated scoring, not additional flag categories.

## Methodology

**Pipeline architecture.** Three agents passing structured outputs:
- *Generator* proposes a drug-discovery hypothesis (target, mechanism, novelty, predicted effect, biosafety note)
- *Critic* evaluates scientific merit, safety concerns, feasibility, and fatal flaws
- *Ranker* synthesizes hypothesis and critique into a final score, biosafety flag, and memory summary that steers subsequent rounds

**Phase 2 design.** Same pipeline, run across the Claude capability ladder (Haiku 4.5 → Sonnet 4.6 → Opus 4.7) on a held-territory set of safety conditions in cardiac biology. The research question: does safety calibration scale with model capability? Results are scored against a pre-registered rubric ([`scoring_rubric.md`](scoring_rubric.md)) using:

- *Per-agent decomposition* — separate scoring of Generator, Critic, and Ranker contributions to identify where in the pipeline safety reasoning originates
- *Blinded cross-model scoring* — model identity is replaced with opaque codes during scoring to mitigate availability bias ([`generate_blinded_scoring_template.py`](generate_blinded_scoring_template.py))
- *Scorer confidence column* — every scored row carries a 1–5 confidence rating, enabling downstream weighting and identifying rows for second-rater review

**Phase 3 directions** (proposed, not yet implemented):
- Cross-lab comparative evaluation extending the protocol to frontier models from at least three labs
- Genetic-phenocopy module that routes safety inference through human-genetics phenotype data (ClinVar, ClinGen) rather than through compound-similarity training distributions

## Repo layout

| File | Purpose |
|------|---------|
| `coscientist.py`, `coscientist_v2.py` | Phase 1 pipeline (reference) |
| `coscientist_phase2.py` | Phase 2 pipeline with capability-ladder support and per-agent decomposition |
| `generate_blinded_scoring_template.py` | Generates unified blinded scoring CSV across all model runs |
| `scoring_rubric.md` | Pre-registered scoring rubric (v1) |
| `breadcrumbs/` | Curated Phase 1 audit logs (the SCN5A finding lives here) |
| `logs/` | Runtime outputs (gitignored; populated when pipeline runs) |

## Current status (April 2026)

- Phase 1: complete. SCN5A finding documented in breadcrumbs/.
- Phase 2: pipeline infrastructure built and tested. Pre-registered scoring rubric committed. Blinded cross-model scoring infrastructure in place. **Capability-ladder pilot runs (Haiku, Sonnet, Opus) in progress; data and scoring forthcoming over the next several weeks.**
- Phase 3: proposed in fellowship application materials; cross-lab and genetic-phenocopy work scoped but not yet implemented.

## Limitations (current pilot stage)

Documented honestly because methodological honesty matters more than methodological perfection at this stage:

- *Single-rater scoring.* Inter-rater reliability not yet established. Recruiting a second domain expert for blinded subsample scoring is planned post-application.
- *Soft blinding.* Source file paths reveal model identity; the scorer commits to ignoring this but cannot guarantee zero leakage.
- *Pilot scope.* Currently 2 conditions × 3 models × 10 rounds. Full 5-condition study is the Phase 2 completion target.
- *Domain restriction.* All work to date is in cardiac antiarrhythmic territory. Generalization to other therapeutic areas is asserted but not yet demonstrated.
- *Pre-registration scope.* The rubric was committed before pilot scoring begins, but cardiac-specific calibration examples in the rubric are under active refinement based on early observations.

## A note on workflow

This work is a collaboration between domain expertise and AI-assisted implementation. Scientific judgment — what to test, why it matters, how to score outputs, what counts as a structurally novel risk — is mine. The technical scaffolding (pipeline code, scoring infrastructure, log handling) was built with substantial assistance from Claude (Claude Code for implementation, Claude chat for design and architecture discussions). The intent is to be transparent about the workflow rather than pretend either contribution is doing the other's job: a non-expert with the same tools could not produce the SCN5A finding, and a domain expert without the tools could not have built this infrastructure in a quarter of part-time work.

## Project context

This work is part of Pete's transition into AI biosafety evaluation, alongside his postdoctoral work in cardiovascular biology and drug discovery at Stanford. It is the primary artifact referenced in his application to the Anthropic STEM Fellows program (June 2026 cohort).
