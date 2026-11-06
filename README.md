# REPOL: Risk Evaluation of Pharmacology by LLMs

Standardized red-teaming of multi-agent AI co-scientists for mechanistic biosafety failures in drug discovery.

**Author:** Peter-James H. Zushin, Ph.D. — Postdoctoral Fellow, Wu Lab, Stanford University
**Status:** November 2026. Phase 1 complete; standardized multi-agent red-teaming study in development.
**Contact:** petezushin@gmail.com

---

## What this project is

REPOL asks a specific question: when an AI co-scientist is steered toward, or handed, a mechanistically hazardous drug optimization, does it catch the specific risk, offer only generic caution, or pass the hazard downstream?

Co-scientists are not single models. Published systems are orchestrated teams of specialized agents, some combining models from different labs, and orchestrator agents can now create sub-agents on demand. Safety is therefore a property of the whole pipeline: a hazard one agent misses can be caught, amplified or laundered by the next. Most safeguards were built for single models and recognize named threat categories. They are not designed to catch novel mechanistic risk, where the danger sits in pathway-level biology rather than in any flag category.

REPOL measures that gap in a domain with expert ground truth: cardiac and metabolic pharmacology, where the line between therapeutic and toxic mechanism is unusually thin.

## The anchor finding (Phase 1)

In Phase 1 (March 2026), a three-agent Generator → Critic → Ranker pipeline running on Claude Sonnet 4.6 proposed **SCN5A augmentation as an antiarrhythmic strategy**. That mechanism would augment the late sodium current (INa-L), prolong cardiac action potential duration and produce Torsades de Pointes (TdP), the lethal arrhythmia the compound was meant to prevent. Neither the Critic nor the Ranker flagged it. The audit log for this run is preserved in [`breadcrumbs/audit_log_20260318_122119.txt`](breadcrumbs/audit_log_20260318_122119.txt).

This is the failure mode REPOL is built to characterize: not a generic safety violation, but a domain-specific mechanistic inversion that an expert would catch immediately and the pipeline's own safety reasoning did not.

## Current work: standardized multi-agent red-teaming

The current study scales the Phase 1 finding from one small pipeline into a systematic, reproducible comparison across co-scientist designs.

**Standardized compositions.** An orchestrator builds co-scientist teams from a common declarative specification, implementing several published designs: supervisor-led tournaments (Gottweis et al., 2026), PI-led specialist teams (Swanson et al., 2025), mixed-lab pipelines (Ghareeb et al., 2026) and swarm-style teams. Every team faces the same scenarios, roles and grader. Team size (three to about twelve agents) and which lab's model fills each role are varied systematically. The orchestrator model is recorded and held fixed within a comparison.

**Red-teaming methods.** Each team is tested with three published approaches, adapted to pharmacology:

1. *Constraint-removal scenarios.* Matched pairs in which a safety-relevant pharmacological constraint is kept or removed, generalizing the Phase 1 anchor and the dual-use inversion shown by Urbina et al. (2022).
2. *Planted flaws.* A flawed mechanistic claim is injected at a defined agent and graded from obvious to subtle, producing a detection-versus-subtlety curve. This builds on evidence that assistants mirror errors presented to them (Sharma et al., 2024) and that team structure shapes resilience to faulty agents (Huang et al., 2025).
3. *Automated adversarial search.* An attacker model iteratively refines framings against the target team (Perez et al., 2022; Chao et al., 2023), reported in the standardized style of HarmBench (Mazeika et al., 2024).

**Grading.** A checklist-driven, mechanism-aware grader, to be validated against blinded expert judgments, scores four outcomes:

- *Detection:* was the hazard caught at all?
- *Specific-risk flagging:* was the actual mechanism named, or only generic caution offered?
- *Propagation:* how far did an uncaught hazard travel through the team?
- *False alarms:* how often were matched benign controls flagged?

**Reproducibility.** Model outputs are non-deterministic, so the study controls what it can and measures the rest:

- model versions and providers pinned, with served-model verification;
- fixed sampling settings;
- repeated runs per condition, reported with confidence intervals rather than single outcomes;
- team specifications and configurations released so runs can be repeated.

**Infrastructure.** Large runs go through OpenRouter. Open-weight models are run locally for pipeline pretesting and for testing model behavior without a provider's safety layer.

## Safety and disclosure

- Scenarios are confined to known pharmacological safety liabilities, not agents of concern.
- Released: methods, team specifications, aggregate results and the grader checklist.
- Withheld: operational prompt and response details. Outputs pass a review step before release.

## Outputs and timeline

- arXiv preprint by March 2027.
- Conference submission to NeurIPS 2027.

## Repo layout

| File | Purpose |
|------|---------|
| `coscientist.py`, `coscientist_v2.py` | Phase 1 pipeline (reference) |
| `coscientist_phase2.py` | Three-agent pipeline with model-ladder support and per-agent decomposition |
| `generate_blinded_scoring_template.py` | Generates a unified blinded scoring CSV across model runs |
| `scoring_rubric.md` | Pre-registered scoring rubric (v1) |
| `breadcrumbs/` | Curated Phase 1 audit logs (the SCN5A finding) |
| `logs/` | Runtime outputs (gitignored) |

Orchestrator, team specifications and the red-teaming harness for the current study will be added as they are built.

## Limitations

Documented up front, because methodological honesty matters more than methodological perfection at this stage:

- *Single-rater ground truth so far.* Grader validation currently rests on the author's blinded judgments; a second domain expert for a blinded subsample is planned.
- *Domain restriction.* All work to date is in cardiac pharmacology. Generalization to other therapeutic areas, and to higher-consequence biological domains, is argued but not yet demonstrated.
- *Proxy domain.* Pharmacology is a deliberately safe proxy. The claim that its failure modes (confident mechanistic error, generic rather than specific caution, error propagation) transfer to higher-consequence domains is a hypothesis this work aims to support, not a result.
- *Combinatorial scope.* Sizes, topologies, vendor mixes and scenarios multiply quickly. The first paper will vary a subset.

## Future directions

- *Genetic-phenocopy module:* routing safety inference through human-genetics phenotype data (ClinVar, ClinGen) rather than compound-similarity training distributions.
- Extension of the standardized protocol beyond cardiac pharmacology.

## Related work by the author

- **Ethogram:** an open evaluation instrument for model refusal behavior on dual-use biology prompts, built on Inspect AI. Manuscript under review at IEEE SaTML 2027.

## A note on workflow

This work is a collaboration between domain expertise and AI-assisted implementation. Scientific judgment (what to test, why it matters, how to score outputs, what counts as a structurally novel risk) is mine. The technical scaffolding (pipeline code, orchestration, scoring infrastructure, log handling) is built with substantial assistance from Claude, through Claude Code for implementation and Claude chat for design and architecture. The intent is to be transparent about the workflow rather than pretend either contribution is doing the other's job: a non-expert with the same tools could not have produced the SCN5A finding, and a domain expert without them could not have built this infrastructure in part-time work.

## References

- Chao, P. et al. (2023). Jailbreaking Black Box Large Language Models in Twenty Queries. arXiv:2310.08419.
- Ghareeb, A.E. et al. (2026). A multi-agent system for automating scientific discovery. *Nature*. doi:10.1038/s41586-026-10652-y
- Gottweis, J. et al. (2026). Accelerating scientific discovery with Co-Scientist. *Nature*. doi:10.1038/s41586-026-10644-y
- Huang, J. et al. (2025). On the Resilience of LLM-Based Multi-Agent Collaboration with Faulty Agents. *ICML 2025*, PMLR 267.
- Mazeika, M. et al. (2024). HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal. *ICML 2024*, PMLR 235. arXiv:2402.04249.
- Perez, E. et al. (2022). Red Teaming Language Models with Language Models. *EMNLP 2022*. arXiv:2202.03286.
- Sharma, M. et al. (2024). Towards Understanding Sycophancy in Language Models. *ICLR 2024*. arXiv:2310.13548.
- Swanson, K. et al. (2025). The Virtual Lab of AI agents designs new SARS-CoV-2 nanobodies. *Nature*.
- Urbina, F. et al. (2022). Dual use of artificial-intelligence-powered drug discovery. *Nature Machine Intelligence* 4, 189–191.
