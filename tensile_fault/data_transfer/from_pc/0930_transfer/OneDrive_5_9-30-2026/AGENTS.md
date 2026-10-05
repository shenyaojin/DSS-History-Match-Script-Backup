# Project revision and private GitHub synchronization instructions

Recorded 2026-09-19 America/Denver at the user's request.

- For changes to project code, prepend a dated modification-purpose comment to each changed notebook code cell and keep an append-only revision record in the notebook introduction. Use actual local date/time and timezone; do not replace earlier history.
- User preference 2026-09-20: number EVERY notebook cell sequentially (Markdown and code), visibly in its first line; retain the modification timestamp alongside the code-cell number. Apply this to future notebooks as well.
- There are now two pairs: 01/02 fit strain; 03/04 fit strain rate. Publish all four requested notebook sources, not generated results.
- Shared time-series figures show x-axis ticks and labels only in the bottom subplot, with `%m/%d/%y` on the first line and `%H:%M` on the second. Waterfall limits: rate ±0.3 nanostrain/s; strain ±0.15 millistrain.
- The active two-step project is the sibling `../Fervo_V1_Project`. Its notebooks are `01_Uniform_Elliptical_Baseline.ipynb`, `02_Independent_Slip_Inversion.ipynb`, `03_Uniform_Elliptical_Rate_Baseline.ipynb`, and `04_Independent_Slip_Rate_Inversion.ipynb`.
- Private repository: `https://github.com/Pengchao-He/Fervo-V1-Project`, branch `main`.
- Sync each user-requested revision of those notebooks to that repository after verification. Future code explicitly requested by the user in this directory should join the same repository. Do not bulk-add existing files or unrelated code.
- Publish only those four requested notebooks. Clear cell outputs from publication copies; preserve locally executed notebooks. Do not publish raw DAS, microseismic, pressure data, local caches, generated output folders, credentials or unrelated research files.
- Verified 2026-09-20: existing Git credentials can push to this private repository. The source-only Git mirror is `V1_DDM/two_step_publish`; use it to commit and push output-cleared notebook copies. The GitHub connector currently cannot access this repository. Do not broaden app permissions or change repository visibility merely to enable synchronization.
- User clarification 2026-09-20: use `git commit` / `git push` only. Do not use Chrome or GitHub's browser file-upload workflow for code synchronization.
- Selected documentation scope: source comments plus dependency/tested-system/run notes inside notebooks. No full CWP or licensing/public-release preparation is implied.
- T1/T2/T3 refer to 2025, with T3 = 2025-03-06 00:00 fixed UTC−7, unless the user explicitly changes the data year.
- Do not promise an unattended sync service: this is an instruction to sync task revisions while working, not a background automation.

## User clarifications — 2026-09-20 22:33:13 MDT

- A Markdown cell containing a title and brief introduction and its following code cell must share the same visible cell number. Advance the number once per title/code pair; number standalone introductory Markdown cells separately. This supersedes the earlier physical-cell sequential numbering preference.
- The user requested notebooks 05/06 for two-hour strain-rate fitting: `05_Uniform_Elliptical_Rate_2h_Baseline.ipynb` and `06_Independent_Slip_Rate_2h_Inversion.ipynb`. Include these explicitly requested new sources in the same private Git repository, in addition to 01–04; publish no generated outputs.
- Pressure-curve stage labels mean the individual hydraulic-fracturing stage numbers recorded in the pressure files, not the model T1–T2/T2–T3 stages.
