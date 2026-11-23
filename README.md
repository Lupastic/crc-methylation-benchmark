# crc-methylation-benchmark

Comparative analysis of CpG feature selection and machine learning methods for colorectal cancer tissue classification.

Reproducible comparison of CpG feature-selection methods and machine-learning classifiers for colorectal tumor-versus-nontumor tissue classification. The experiment develops models on **GSE48684** and evaluates frozen models on the separate **GSE68060** cohort.

This is a research pilot comparing existing methods. It does not introduce a new algorithm or establish a clinical screening test.

## What is compared

- Selectors: absolute mean methylation difference (`DeltaBeta`), mutual information (`MI`), and standardized L2-logistic-regression coefficient magnitude (`L2coef`).
- Classifiers: logistic regression, linear SVM, and random forest.
- CpG panel sizes: 30, 100, and 500, giving 27 configurations.
- Evaluation: five-fold outer / three-fold inner cross-validation, external balanced accuracy, descriptive feature stability, rank transfer, and measured computational resources.

The main question is whether the ranking of methods within the development cohort persists on an independent cohort under the same feature budgets.

## Repository contents

| Path | Contents |
|---|---|
| [`Research_Code/README.md`](Research_Code/README.md) | Full protocol, interpretation limits, Russian quick-start instructions |
| [`Research_Code/run.py`](Research_Code/run.py) | Data download, preprocessing, training, evaluation and bootstrap |
| [`Research_Code/config.json`](Research_Code/config.json) | Frozen experiment configuration |
| [`Research_Code/analyze_results.py`](Research_Code/analyze_results.py) | Figures generated from recorded results |
| [`Research_Code/audit_results.py`](Research_Code/audit_results.py) | Integrity checks on recorded outputs |
| [`Research_Code/tests/test_pipeline.py`](Research_Code/tests/test_pipeline.py) | Small software checks; synthetic test arrays are not scientific data |
| [`results/summary.csv`](results/summary.csv) | Results for all 27 configurations |
| [`results/manifest.json`](results/manifest.json) | Execution versions, hashes, hardware and resource measurements |
| [`results/`](results/) | Fold results, predictions, CpG panels, bootstrap draws and public sample metadata |

## Reproduce

Run these commands from the repository directory on Linux or macOS:

```bash
cd Research_Code
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run.py --output ../reproduced_results
python analyze_results.py --results ../reproduced_results
python audit_results.py --results ../reproduced_results
```

Using `reproduced_results` preserves the committed reference outputs. The tested Python/package versions are recorded in the reference manifest; dependencies are pinned in `requirements.txt`. The first run downloads the deposited GEO matrices and creates a local cache. Allow several GB of RAM and disk space; runtime depends on hardware and connection speed. Large matrices, caches and virtual environments are excluded from Git.

Rendered figures are distributed with the accompanying paper package and can be regenerated with `analyze_results.py`; figure binaries are not committed here. The reference manifest describes the full original run/package, including figure artifacts and a packaging-only upload guide. Those files are intentionally absent from this text-only repository; the actual run manifest is preserved unchanged.

## Data and scope

Development uses 64 cancer and 17 normal-H specimens from [GSE48684](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE48684). Its 24 normal-C specimens and 42 adenomas are excluded from training and evaluated only exploratorily. External evaluation uses 82 tumor and 36 normal specimens from [GSE68060](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE68060), excluding two experimental controls. Sample inclusion and labels follow the deposited metadata.

The script fits all downstream filtering, imputation, variance preselection, supervised selection and scaling on training partitions only. However, GSE48684 was already normalized and batch-adjusted by its depositors. The original preprocessing is not repeated within these folds. External control recruitment differs from development normal-H, donor identifiers are incomplete, and external bootstrap grouping is inferred from sample titles. The development cohort also has different sex distributions between classes; sex-chromosome, SNP-overlap and cross-reactive probe filters are not applied. Read the full protocol before interpreting the scores or selected CpGs.

These are retrospective tissue classifications. They do not measure blood-based early detection, clinical cancer risk or causal biomarker effects.

## Dataset publications

1. Luo et al. (2014). *Differences in DNA methylation signatures reveal multiple pathways of progression from adenoma to colorectal cancer*. [doi:10.1053/j.gastro.2014.04.039](https://doi.org/10.1053/j.gastro.2014.04.039).
2. Conesa-Zamora et al. (2015). *Methylome profiling reveals functions and genes which are differentially methylated in serrated compared to conventional colorectal carcinoma*. [doi:10.1186/s13148-015-0128-7](https://doi.org/10.1186/s13148-015-0128-7).
3. García-Solano et al. (2018). *Two histologically colorectal carcinomas subsets from the serrated pathway show different methylome signatures and diagnostic biomarkers*. [doi:10.1186/s13148-018-0571-3](https://doi.org/10.1186/s13148-018-0571-3).

## Preparation assistance

OpenAI ChatGPT/Codex assisted with code, documentation and literature organization. Numerical reference outputs come from executing the published analysis code on the deposited datasets. Publication of this repository does not constitute submission or acceptance of a scientific article.
