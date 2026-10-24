# TTS Project Entry Point

Read `PROJECT_BRIEF.md` in full before working on this project. It records the previously discussed objectives, authorization boundaries, experiment design, and delivery standards. Later explicit user instructions take precedence.

## Starting and Continuing Work

- When the user explicitly asks to start this project or its Goal, first read the code and perform necessary read-only checks. Ask any questions that require the user's decision together; make routine technical choices independently and do not reconfirm known information.
- After receiving necessary answers, create and start the Goal; if there are no required questions, start directly. Reading instructions, editing plans, or answering questions alone does not authorize starting the research.
- Once started, continue without waiting for a request to continue after every stage. System permissions, credentials, and irreplaceable information may still require the user's involvement and must not be bypassed.
- Communicate in Chinese by default. Record key findings, decisions, actual progress, and blockers.

## Project Scope

- Keep implementation, isolated environments, configuration, data, checkpoints, and results within this subdirectory. Protect the parent directory and other projects.
- Run all computation locally. Download necessary public resources as needed; do not use paid APIs or cloud GPUs. Do not automatically publish or upload material, or submit coursework, except within the explicit repository authorization below.
- Establish the Part C baseline first. Model and method choices remain open: evidence-based training changes, local architecture changes, and other local models are allowed. Do not add architecture complexity merely for novelty.
- Prioritize one complete continued-training epoch over all valid training data when feasible. Allocate the budget using measured throughput, preserving time for fair comparisons, final evaluation, and reporting.
- Use multiple agents as needed. The lead agent manages editing boundaries, resources, integration, and verification.

## Recovery and Evidence

- After implementation starts, maintain `STATUS.md` with confirmed decisions, completed work, running jobs and log paths, checkpoints, unresolved questions, next steps, and recovery commands. This handoff record does not replace the application's Goal state.
- Preserve important configurations, seeds, data sources, and raw results in files, not only in chat.
- Save recoverable checkpoints periodically. Use bounded retries for downloads and installation, and judge stalled jobs from actual progress.
- Do not fabricate experiments, human listening, metrics, speedups, or resume accomplishments. Negative results can support a complete research conclusion; experiments that have not run are not complete.

## Repository Authorization — 2026-10-23

The user authorized consistent English project files, an independent local Git repository rooted in `tts-reliability/` (not the parent workspace), and creation of and upload to a **private** repository on the user's GitHub account. Curate the tracked files for presentation and reproducibility, excluding private personal artifacts and bulk assets such as datasets, model weights, environments, and generated audio. This authorization does not start or resume research or training, authorize public sharing, or authorize coursework submission. Default chat communication remains Chinese.
