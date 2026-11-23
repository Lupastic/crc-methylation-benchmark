# Colorectal methylation feature-selection transferability — real-data pilot

## Быстрый запуск

Это воспроизводимый учебный исследовательский пилот на реальных опубликованных матрицах метилирования ДНК. Все числовые результаты получены запуском кода, а не сгенерированы. Исследование не является клиническим диагностическим тестом и не проверяет раннее выявление рака по крови.

Проверено на Python 3.12.14, Linux, CPU, один вычислительный поток. В каталоге проекта:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run.py --output ../reproduced_results
python analyze_results.py --results ../reproduced_results
python audit_results.py --results ../reproduced_results
```

Первый запуск скачивает два общедоступных файла GEO (суммарно около 758 МБ) и сохраняет локальный кэш. Нужны интернет, свободное место для исходных матриц и кэша и несколько ГБ оперативной памяти. При повторном запуске используются локальные данные. Папки `data/raw` и `data/processed` не входят в Git. Числовые CSV и JSON находятся в папке, переданной через `--output`. Время воспроизведения зависит от CPU и интернета. Не прерывайте процесс во время записи кэша; после прерывания удалите неполный каталог соответствующего набора в `data/processed`.

`config.json` фиксирует все исследуемые варианты до внешней оценки. `protocol_frozen.json` записывается до обучения; `frozen_external_models.json` — после выбора гиперпараметров по обучающему набору и до внешних прогнозов. `manifest.json` содержит версии, SHA-256, оборудование и измеренные ресурсы. `run.log` — журнал выполненного эталонного запуска. Малые искусственные массивы в unit tests служат только проверке программы и никогда не используются как научные данные.

## English scientific protocol

### Research question and scope

A comparative benchmark asks whether CpG-selection approaches retain predictive performance and panel stability when colorectal tissue data move between public cohorts. It compares existing methods; it does not introduce a new machine-learning algorithm. The original broader project proposed TCGA-COAD/READ development. This completed pilot uses **GSE48684 → GSE68060**, with three classifiers, three selectors and three panel budgets. TCGA, gradient boosting, raw-IDAT reprocessing, biological validation and liquid-biopsy analysis remain future work.

### Cohorts and provenance

- Development: GSE48684, 64 colorectal cancer and 17 normal-H specimens (81 total). The 24 normal-C samples and 42 adenomas are excluded from all training/tuning and receive exploratory predictions only. Equal numeric suffixes of cancer and normal-C titles do not establish donor matches: some corresponding sexes differ. Using normal-H prevents these uncertain tumour/adjacent pairs from entering both sides of development CV.
- External: GSE68060, 82 Tumor and 36 Normal specimens (118 total), after excluding two IVD controls. Labels follow deposited metadata. External normal tissue has a different control context from development normal-H; external performance changes combine cohort and control-type shifts and cannot be attributed to only one.
- Raw deposited matrix dimensions are 485,577×147 and 472,139×120 probes×samples. Their intersection restricted to `cg` IDs is 469,160 features. Probe intersection uses identifiers only, never external beta values or labels. Raw files are checked for sample/probe identifier uniqueness, exact duplicate methylation profiles and beta range. Download hashes and missingness are in `qc_summary.json`.
- GSE48684 source processing already used SWAN and ComBat. Those source-level operations cannot be refit within our folds. The training-only guarantee therefore applies to **our downstream processing**, not to the entire raw-data pipeline. GSE68060 uses submitted GenomeStudio beta values. No additional joint normalization or batch correction is applied.
- GEO metadata: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE48684 and https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE68060
- Original studies: Luo et al. (2014), PMID 24793120, DOI 10.1053/j.gastro.2014.04.039; Conesa-Zamora et al. (2015), DOI 10.1186/s13148-015-0128-7. The expanded serrated-pathway analysis is García-Solano et al. (2018), Clinical Epigenetics 10:141, DOI 10.1186/s13148-018-0571-3. Current downloaded sample counts are derived from matrix metadata, rather than copied from an earlier abstract. Verify bibliographic formatting against the accompanying manuscript.

### Downstream pipeline

Each inner training partition independently learns: (1) probe missingness ≤5%; (2) median imputation; (3) nonconstant filtering and the 5,000 highest-variance probes; (4) a supervised ranking and top-k panel. All fitted states transform that partition's validation data without refitting. The same sequence is independently refit on each outer training set and finally on all development data.

Selectors:

1. `DeltaBeta`: absolute difference in mean beta between cancer and normal controls.
2. `MI`: scikit-learn continuous-feature mutual information, k-nearest-neighbour estimator with `n_neighbors=3`, a fixed seed, and the training labels.
3. `L2coef`: absolute coefficients of class-balanced L2 logistic regression, C=1, fitted on standardized training features. It is a dense regularized ranking, **not L1 or Elastic Net**. Ties use deterministic probe order inherited from the variance filter and shared CpG IDs.

Panel sizes are 30, 100 and 500. LR and linear SVM receive training-standardized panel values; RF uses imputed beta values without standardization. All classifiers use balanced class weights. LR uses liblinear; SVM uses LinearSVC. C is selected from [0.01, 0.1, 1, 10]. RF uses 100 trees, `max_features='sqrt'`, depth {None, 5}, and minimum leaf size {1, 3}: four candidates per classifier. Hyperparameters maximize mean inner balanced accuracy; first grid order breaks ties. Predictions use native fixed decision thresholds (LR/RF probability 0.5; SVM margin 0), with no external calibration or threshold tuning. Scores from SVM are margins, not probabilities.

### Validation and stability

Five outer folds and three inner folds use `StratifiedGroupKFold`, seed 42 and fixed derived seeds. Development group IDs are unique source specimen-title identifiers; verified donor UUIDs are unavailable. Consequently grouped splitting prevents repeated recorded IDs but does not prove donor independence. The same folds are used for every configuration. Every candidate is evaluated; no winner is selected using external labels.

Primary performance is balanced accuracy. ROC-AUC and average precision are secondary; sensitivity, specificity, F1 and confusion counts are retained. Internal means and descriptive SDs are computed across the five outer folds. Pooled OOF balanced accuracy is additional; pooled OOF SVM AUC is deliberately not used because independently fitted margins need not share scale. Fold SD is not a confidence interval.

Within each outer training set, ten 80% group subsamples rerun all downstream preprocessing and selection. Each selector/panel has 45 dependent pairwise Jaccards per fold. Their mean and between-fold descriptive SD describe perturbation stability; pairwise values are not independent biological replicates.

Final parameters and panels are fixed using only development data before external prediction. Rank transfer uses descriptive Spearman correlation of internal mean BA and external BA for nine configurations at each fixed k; the overall 27-configuration correlation is additional. Rank ties are determined after rounding BA to 12 decimal places solely to remove floating-point noise; reported metric values are unchanged. Configurations share samples and are dependent, so no independent-observation p-value is claimed.

Two contrasts are declared in the frozen config: LR k=100, MI−DeltaBeta and L2coef−DeltaBeta. External percentile 95% intervals use 2,000 paired bootstrap resamples of **84 inferred clusters**, seed 2026. Clusters conservatively remove the tissue letter T/N and optional M/P from country/histology/numeric title codes; no meaning is asserted for M/P. `ESSMN` lacks a numeric suffix and remains an unresolved separate cluster. These are unverified heuristic clusters, not confirmed patients. Intervals are conditional on fitted models and the observed cohort, exclude retraining/selection uncertainty and do not establish clinical validity. The two comparisons are exploratory and not adjusted for multiple testing.

### Timing and memory

Classifier-search time includes twelve inner classifier fits plus one outer/full-development classifier fit. Preprocessing and ranking are cached once per split and reused across classifiers and k. Their measured times are attributed in full to each applicable configuration to make components visible; **attributed totals must not be summed across configurations**. Per-panel scaler fitting/transformation overhead is not included in these attributed component totals; it is included in overall run wall time. Memory values named `_mb` in the CSV/JSON are measured in MiB (2^20 bytes), and host RAM `_gb` in GiB. Peak RSS samples the entire process every 20 ms and includes loaded matrices and caches. It is not isolated model memory. Total wall/CPU time covers loading, CV, stability, final fitting, evaluation and bootstrap. In the supplied run downloads were cached; plots and document production are excluded. Full definitions and measured values are in `manifest.json`.

### Interpretation limits

Sample sizes are small, particularly normal-H (17). Development sexes differ (CRC 41 female/23 male; normal-H 6 female/11 male). Only common cg-probe identifiers are filtered; sex-chromosome, SNP-overlap and cross-reactive probe filters are not applied. No gene/promoter annotation or causal biomarker conclusion is claimed. Previously processed source cohorts, unidentified donors, control-type differences and country/tumour-subtype differences may influence performance. Exploratory normal-C specificity may be optimistic if those tissues share donors with training tumours. Adenoma positive rate is a descriptive cancer-score response, not sensitivity or diagnostic accuracy for adenomas. These are tissue case-control results, not estimates of population screening performance or early-stage sensitivity.

## Output map

| File | Meaning |
|---|---|
| `summary.csv` | All 27 configurations: internal mean/SD, external scores, stability and resource components |
| `fold_metrics.csv` | 135 outer-fold evaluations and confusion counts |
| `predictions.csv` | Specimen-level internal, external and exploratory fixed-threshold predictions |
| `hyperparameter_search.csv` | Every candidate and all three inner-fold BA values |
| `selected_features.csv`, `external_selected_features.csv` | CpG panel identities from outer-training and final-training fits |
| `splits.csv` | Auditable development outer fold assignments |
| `stability_by_fold.csv`, `stability_summary.csv` | Train-only perturbation stability |
| `external_metrics.csv`, `exploratory_metrics.csv` | Final external and explicitly exploratory subgroup evaluations |
| `bootstrap_replicates.csv`, `primary_comparison.json` | Conditional paired-cluster bootstrap and predeclared contrasts |
| `rank_transfer.json` | Descriptive fixed-budget and overall rank correlation |
| `qc_summary.json`, `GSE*_metadata.csv` | Data counts, missingness, exclusions, group rules and source metadata |
| `integrity_audit.json` | Recomputed metrics, split/panel counts, frozen-code consistency checks |
| `protocol_frozen.json`, `frozen_external_models.json`, `manifest.json` | Frozen protocol, final settings, versions, hashes, hardware and resources |
| `figures/*.png`, `figures/*.pdf` | Figures reproduced by `analyze_results.py` from reported CSVs |

Public metadata contain accession codes, specimen titles and study-level attributes, not personal names. The repository contains reproducible code and compact result tables; large source matrices are downloaded from GEO.
