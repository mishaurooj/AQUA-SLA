# AQUA-SLA: A Leakage-Aware Hybrid Quantum-Classical Framework for SLA Violation Prediction and Quantum-Cloud Resource Orchestration with Qiskit and iQuantum


---
# Dataset Used in AQUA-SLA

## Google Cluster Workload Traces 2019

AQUA-SLA uses a sample derived from the **Google Cluster Workload Traces 2019** dataset as the empirical workload source for SLA-risk prediction, resource-aware scheduling, hybrid quantum-classical experimentation, and iQuantum-based quantum-cloud simulation.

This file documents the dataset provenance, scope, acquisition procedure, intended use, limitations, licensing, and reproducibility requirements for this repository.

---

## 1. Dataset identity

| Field | Description |
|---|---|
| Dataset name | Google Cluster Workload Traces 2019 |
| Trace version | Version 3, commonly called `ClusterData2019` |
| Original provider | Google Research |
| Workload manager | Google Borg |
| Observation period | May 2019 |
| Number of production clusters/cells | Eight Borg cells |
| Repository source used by AQUA-SLA | Kaggle sample/mirror |
| Full-trace access | Google BigQuery |
| Domain | Cloud computing, cluster scheduling, resource management, workload characterization |

### Primary sources

- Official Google trace documentation:  
  https://github.com/google/cluster-data/blob/master/ClusterData2019.md

- Official Google Research announcement:  
  https://research.google/blog/yet-more-google-compute-cluster-trace-data/

- Kaggle sample used for local experimentation:  
  https://www.kaggle.com/datasets/derrickmwiti/google-2019-cluster-sample

- Google Research paper describing and analyzing the 2019 Borg trace:  
  https://research.google/pubs/borg-the-next-generation/

- DOI for the accompanying EuroSys 2020 paper:  
  https://doi.org/10.1145/3342195.3387517

- Additional archive link referenced on the Kaggle dataset page:  
  https://drive.google.com/file/d/10r6cnJ5cJ89fPWCgj7j4LtLBqYN9RiI9/view

---

## 3. What the trace contains

The trace is intended for research on cloud infrastructure and scheduling. Depending on the selected table or sample, relevant information may include:

- jobs and tasks;
- submission and scheduling events;
- task states and transitions;
- resource requests;
- CPU, memory, and other resource-usage measurements;
- machine or cell-level placement information;
- task priority and scheduling class;
- collection and alloc-set relationships;
- job-parent relationships;
- time-dependent resource behavior.

The exact columns available in this repository depend on the Kaggle sample files downloaded by the researcher. The repository must not claim that every field from the full BigQuery trace is present in the local sample.

---

## 4. What the trace does not contain

The trace focuses on infrastructure-level requests, scheduling, and resource usage. According to the official documentation, it does not contain:

- end-user identities;
- user application data;
- storage contents;
- user access patterns to storage systems or services.

The dataset is nevertheless derived from a real production environment and should be handled according to its stated license and research-use conditions.

---

## 5. Role of the dataset in AQUA-SLA

The Google trace is the **classical workload dataset** used to construct and evaluate the AQUA-SLA pipeline.

It is not a quantum-computing dataset. Quantum methods are applied later in the workflow to derived scheduling or optimization representations.

The intended data flow is:

```text
Google 2019 Borg trace sample
        |
        v
Data cleaning and schema normalization
        |
        v
Feature engineering and temporal ordering
        |
        v
Project-defined SLA-violation target
        |
        v
Leakage-aware train/validation/test evaluation
        |
        v
Calibrated SLA-risk probabilities
        |
        v
Risk-aware resource scheduling
        |
        +---------------------------+
        |                           |
        v                           v
Qiskit/QAOA experiments      iQuantum simulation
        |                           |
        +-------------+-------------+
                      |
                      v
       Consolidated AQUA-SLA evaluation
```

### Important target-label clarification

The raw Google trace does **not** contain a native field called `sla_violation` defined specifically for AQUA-SLA.

Any SLA-violation label, risk class, deadline breach, resource-pressure indicator, or related prediction target used in this repository is a **project-defined derived target**. 

Researchers reproducing the project should not assume that the label was supplied directly by Google.

---

## 6. Relationship to the iQuantum integration

The Google trace provides the empirical workload characteristics. iQuantum provides a separate discrete-event simulation environment for quantum-cloud resource-management experiments.

In the AQUA-SLA workflow:

1. the Google trace sample is processed by the Python pipeline;
2. SLA-risk scores and scheduling assignments are generated;
3. the selected workload is exported to an iQuantum-compatible task representation;
4. the Java iQuantum simulation evaluates quantum-resource allocation, waiting time, execution time, completion behavior, makespan, and related platform metrics;
5. iQuantum outputs are imported back into the consolidated AQUA-SLA results package.

Therefore, iQuantum is not a replacement for the Google dataset. It is a downstream simulation platform that consumes workload information derived from the dataset.

iQuantum results must be described as **simulation results**, not measurements from physical quantum hardware.
____

## 4. Platform requirements

The current scripts are configured for Windows paths and use:

- Windows 10/11;
- Anaconda or Miniconda;
- Python 3.11 recommended for this repository;
- Java JDK 17 or newer for iQuantum;
- Apache Maven 3.9 or newer for iQuantum;
- sufficient RAM for quantum-kernel and QAOA experiments.

The official iQuantum project describes itself as a lightweight discrete-event toolkit based on CloudSim and lists Java 17+ and Maven 3.9+ as prerequisites.

---

## 4. Clone and create the Anaconda environment

### 4.1 Clone this repository

```bat
git clone https://github.com/YOUR-USERNAME/AQUA-SLA-Hybrid-Quantum-Orchestration.git D:\other\AQUA-SLA
cd /d D:\other\AQUA-SLA
```

The scripts currently contain:

```python
PROJECT_DIR = Path(r"D:\other\AQUA-SLA")
```

Cloning to this location allows the scripts to run without path changes. For a portable public release, refactor `PROJECT_DIR` to use an environment variable or repository-relative path.

### 4.2 Create the environment

```bat
conda create -n aqua-sla python=3.11 -y
conda activate aqua-sla
python -m pip install --upgrade pip
```

### 4.3 Install the core scientific stack

```bat
python -m pip install numpy pandas scipy scikit-learn xgboost joblib matplotlib pillow openpyxl tabulate
```

### 4.4 Install the Qiskit stack

```bat
python -m pip install qiskit qiskit-aer qiskit-machine-learning qiskit-algorithms qiskit-optimization
```

### 4.5 Optional hardware-provider packages

IBM Quantum:

```bat
python -m pip install qiskit-ibm-runtime
```

IQM:

```bat
python -m pip install qiskit-iqm
```

### 4.6 Verify Python dependencies

```bat
python -c "import numpy, pandas, scipy, sklearn, qiskit, qiskit_machine_learning; print('Python stack OK')"
```

### 4.7 Freeze the tested environment

After a successful final run:

```bat
conda env export --no-builds > environment.yml
python -m pip freeze > requirements-lock.txt
```

Commit these files so reviewers can recreate the exact environment.

---

## 5. Dataset placement

Place the input file at:

```text
D:\other\AQUA-SLA\Dataset\borg_traces_data.csv
```

The repository expects generated datasets under:

```text
D:\other\AQUA-SLA\Dataset\processed\
```

Do not commit the raw dataset unless its license permits redistribution. At minimum, publish:

- dataset name and source;
- acquisition date/version;
- preprocessing steps;
- checksum;
- column description;
- instructions for obtaining it independently.

---

## 6. Run order and expected output from every file

All commands below assume:

```bat
conda activate aqua-sla
cd /d D:\other\AQUA-SLA\Code
```

### Data inspection and preparation

| Script | Purpose | Expected result |
|---|---|---|
| `01_inspect_dataset.py` | Inspect raw columns, shapes, types, and sample records. | Console-only diagnostic confirming that `borg_traces_data.csv` can be read. |
| `02_dataset_profile.py` | Produce a compact raw-data profile. | `results/dataset_profile.csv`. |
| `03_prepare_dataset.py` | Parse structured fields, engineer resource/distribution features, derive labels, and write the prepared dataset. | `Dataset/processed/aqua_sla_prepared.csv` and `aqua_sla_preparation_metadata.json`. |
| `04_check_prepared_dataset.py` | Validate the prepared dataset, labels, missingness, and columns. | Console validation report. |
| `05_inspect_structured_fields.py` | Inspect dictionary/array-like fields before or after parser changes. | Console-only structured-field diagnostic. |
| `05_create_model_datasets.py` | Create leakage-aware early-prediction and runtime-prediction tables. | `aqua_sla_early_prediction.csv`, `aqua_sla_runtime_prediction.csv`, and `model_dataset_metadata.json`. |
| `06_event_distribution.py` | Summarize raw event types and violation-event counts. | `results/event_distribution.csv`. |
| `07_time_analysis.py` | Inspect and sanity-check time columns and units. | Console-only time analysis. |

Run the preparation sequence:

```bat
python 01_inspect_dataset.py
python 02_dataset_profile.py
python 03_prepare_dataset.py
python 04_check_prepared_dataset.py
python 05_inspect_structured_fields.py
python 05_create_model_datasets.py
python 06_event_distribution.py
python 07_time_analysis.py
```

### Baseline and early quantum models

| Script | Purpose | Expected result |
|---|---|---|
| `06_train_baselines.py` | Train classical baseline pipelines on the early-prediction dataset. | Models under `models/classical`; metrics, predictions, metadata, and split report under `results/metrics`. |
| `07_train_quantum_models.py` | Train a pure quantum-kernel SVM, static hybrid kernel, and AQUA-SLA Lite model. | Models under `models/quantum`; kernel samples, alpha search, model comparison, predictions, and metadata under `results/quantum`. |
| `08_train_aqua_sla_adaptive.py` | Train the adaptive hybrid-kernel architecture, optimize adaptive parameters, and choose an SLA threshold. | Adaptive model, optimization history, threshold search, test predictions, metrics, and metadata. |
| `09_repeated_aqua_sla_experiments.py` | Repeat the early hybrid experiment across seeds. | `results/quantum/repeated/repeated_quantum_results.csv`, summary, and metadata. |

Commands:

```bat
python 06_train_baselines.py
python 07_train_quantum_models.py
python 08_train_aqua_sla_adaptive.py
python 09_repeated_aqua_sla_experiments.py
```

### AQUA-SLA v2-v4 prediction architectures

| Script | Purpose | Expected result |
|---|---|---|
| `10_train_aqua_sla_v2.py` | Runtime-telemetry three-kernel multiple-kernel learning. | Feature ranking, optimization history, threshold search, predictions, metrics, model parameters, and metadata under `results/quantum/aqua_sla_v2`. |
| `11_train_aqua_sla_v3.py` | Scalable classical/latent-quantum hybrid predictor with validation-only blend and threshold selection. | `aqua_sla_v3.joblib`, feature ranking, blend search, threshold search, test comparisons, predictions, and metadata. |
| `12_aqua_sla_v4_quantum_strengthening.py` | Repeated quantum configuration search across feature maps, latent dimensions, and seeds. | Seed-specific searches/models, `repeated_model_results.csv`, summary, paired tests, and `iquantum_risk_export.csv`. |

Commands:

```bat
python 10_train_aqua_sla_v2.py
python 11_train_aqua_sla_v3.py
python 12_aqua_sla_v4_quantum_strengthening.py
```

### Quantum-aware scheduling and locked combined core

| Script | Purpose | Expected result |
|---|---|---|
| `13_aqua_sla_quantum_scheduler_FAST.py` | Fast QAOA scheduling smoke test with reduced repetitions and optimizer iterations. | Scheduling batches, assignments, summaries, improvements, iQuantum-ready scheduler metrics, and metadata. |
| `13_aqua_sla_quantum_scheduler.py` | Full repeated scheduler comparison using round-robin, risk-aware greedy, Hungarian, and QAOA methods. | Same output family as the FAST version, with a larger experiment. |
| `14_aqua_sla_combined_final.py` | Lock the v3 prediction evaluation and connect it to the scheduling experiment. | `results/combined/aqua_sla_final/` containing locked predictions, metrics, assignments, scheduling summaries, improvements, tests, exact-match summary, metadata, and `paper_ready_summary.txt`. |

Recommended order:

```bat
python 13_aqua_sla_quantum_scheduler_FAST.py
python 13_aqua_sla_quantum_scheduler.py
python 14_aqua_sla_combined_final.py
```

The generic scheduler can consume substantial memory. Treat the FAST script as a functional test and the controlled QAOA experiments in Script 21 as the more reproducible sensitivity study.

### Sample-size, scalability, and classical-branch ablation

| Script | Purpose | Expected result |
|---|---|---|
| `16_aqua_sla_sample_size_study.py` | Evaluate training sizes 2k, 5k, and 10k across five seeds. | Run-level metrics, summary with confidence intervals, paired tests, selected features, feature stability, and metadata. |
| `17_aqua_sla_sample_size_study_100k.py` | Extend sample sizes to 50k and 100k. | Updated versions of the same sample-size tables. |
| `18_aqua_sla_extended_ablation_save_models.py` | Extend the study to 248k and save every model/prediction artifact. | `saved_models/`, `test_predictions/`, run/summary/test/stability tables, and metadata. |
| `19_aqua_sla_complete_sample_ablation.py` | Produce the complete sample-size package, best model, diagnostic figures, report, and Excel workbook. | `results/prediction/sample_size_study/` with CSVs, models, best model, figures, Markdown report, metadata, and `aqua_sla_ablation_results.xlsx` when `openpyxl` is installed. |

For the final paper, Script 19 supersedes Scripts 16-18 as the complete sample-size artifact generator, while the earlier scripts document the experimental evolution.

```bat
python 19_aqua_sla_complete_sample_ablation.py
```

### Quantum-circuit figures and complete quantum ablation

| Script | Purpose | Expected result |
|---|---|---|
| `20_save_quantum_circuit_figures.py` | Draw publication-grade state-preparation, feature-map, variational, kernel, SWAP-test, and QAOA circuits. | JPEG circuit figures under `results/aqua_sla_final/sample_size_study/saved_models/quantum_circuit_figures_final`. |
| `21_aqua_sla_complete_quantum_ablation.py` | Run prediction-component ablation, controlled 2-task x 2-resource QAOA sensitivity, save circuits/QASM, produce figures/reports, and optionally run IBM/IQM smoke tests. | `results/aqua_sla_final/quantum_ablation_complete/` with `csv/`, `models/`, `circuits/`, `figures_jpeg/`, `hardware/`, `reports/`, and `logs/`. |

Commands:

```bat
python 20_save_quantum_circuit_figures.py
set AQUA_RUN_IBM_HARDWARE=0
set AQUA_RUN_IQM_HARDWARE=0
python 21_aqua_sla_complete_quantum_ablation.py
```

Optional hardware execution must be explicitly enabled and configured. Never commit tokens.

### Leakage-controlled honest hybrid evaluation

| Script | Purpose | Expected result |
|---|---|---|
| `26_aqua_sla_honest_hybrid_quantum_ablation.py` | Run nested repeated hybrid model selection, out-of-fold evaluation, transparent component ablation, and optional companion QAOA extension. | `results/aqua_sla_final/honest_hybrid_quantum_ablation/` with fold metrics, OOF predictions, model-selection records, summaries, models, figures, report, logs, metadata, and optional Excel workbook. |

Run with the QAOA companion:

```bat
set AQUA_SKIP_QAOA=0
python 26_aqua_sla_honest_hybrid_quantum_ablation.py
```

Run only the honest fusion study:

```bat
set AQUA_SKIP_QAOA=1
python 26_aqua_sla_honest_hybrid_quantum_ablation.py
```

### CLOUDS Lab iQuantum integration

| Script | Purpose | Expected result |
|---|---|---|
| `27_aqua_sla_iquantum_platform_integration.py` | Convert locked AQUA-SLA scheduling assignments into iQuantum QTasks, generate a Java bridge, execute iQuantum through Maven, import task-level simulation results, and create a platform ablation. | `results/aqua_sla_final/iquantum_platform_integration/` with workload/results CSVs, merged results, platform summary, ablation, logs, status JSON, and Markdown report. |

### Final experiment packaging

| Script | Purpose | Expected result |
|---|---|---|
| `28_aqua_sla_master_experiment_runner.py` | Run selected final scripts, collect separate result layers, preserve circuits, snapshot sources, save logs/manifests, and create a combined inventory/workbook. | A timestamped directory under `results/aqua_sla_experiment_runs/`. |

---

## 10. iQuantum installation and integration

### 10.1 What iQuantum contributes

AQUA-SLA produces risk-aware scheduling assignments in Python. Script 27 translates those assignments into a workload for the CLOUDS Lab iQuantum platform. The integration is:

```text
AQUA-SLA locked scheduling assignments
        -> Python workload transformation
        -> aqua_sla_iquantum_workload.csv
        -> generated AquaSlaIQuantumBridge.java
        -> Maven compile and exec:java
        -> iQuantum discrete-event simulation
        -> iquantum_task_results.csv
        -> Python merge, summary, platform ablation, report
```

The generated bridge performs the following operations:

1. initializes the iQuantum simulation engine;
2. creates two QNode profiles with 7 and 27 qubits;
3. creates chain qubit topologies and a native gate set (`CX`, `ID`, `RZ`, `SX`, `X`);
4. creates a quantum datacenter, broker, and cloud gateway;
5. reads AQUA-SLA tasks from CSV and creates iQuantum `QTask` objects;
6. assigns each task a preferred QNode from the AQUA-SLA scheduling output;
7. starts and stops the discrete-event simulation;
8. exports status, QNode, qubits, layers, shots, waiting time, QPU time, start/finish times, and cost;
9. merges those outputs with the original AQUA-SLA workload;
10. writes a platform-level summary and ablation table.

### 10.2 Install Java and Maven

Verify:

```bat
java -version
javac -version
mvn -version
where java
where mvn
```

Expected minimums:

```text
Java/JDK: 17+
Maven:    3.9+
```

Example environment variables:

```bat
set JAVA_HOME=C:\Program Files\Eclipse Adoptium\jdk-17.0.19.10-hotspot
set MAVEN_HOME=C:\Program Files\Apache\maven
set PATH=%JAVA_HOME%\bin;%MAVEN_HOME%\bin;%PATH%
```

Use the actual installed JDK directory on your machine.

### 10.3 Clone iQuantum

```bat
cd /d D:\other\AQUA-SLA
mkdir third_party
cd third_party
git clone https://github.com/Cloudslab/iQuantum.git
```

Set the integration variables:

```bat
set IQUANTUM_HOME=D:\other\AQUA-SLA\third_party\iQuantum
set AQUA_RUN_IQUANTUM=1
```

Build iQuantum from the directory containing its root `pom.xml`:

```bat
cd /d D:\other\AQUA-SLA\third_party\iQuantum
mvn clean install -DskipTests
```

Record the tested iQuantum commit:

```bat
git rev-parse HEAD
```

Save this commit hash in the paper and final experiment metadata.

### 10.4 Prepare the required input

Script 27 searches for:

```text
locked_scheduling_assignments.csv
```

Generate it by running Script 21, or ensure a compatible locked assignment file is present under the results tree.

### 10.5 Run the integration

```bat
conda activate aqua-sla
set IQUANTUM_HOME=D:\other\AQUA-SLA\third_party\iQuantum
set AQUA_RUN_IQUANTUM=1
cd /d D:\other\AQUA-SLA\Code
python 27_aqua_sla_iquantum_platform_integration.py
```

### 10.6 Verify successful integration

Open:

```text
results\aqua_sla_final\iquantum_platform_integration\reports\iquantum_execution_status.json
```

A successful platform run requires:

```text
requested = true
success = true
status = completed
returncode = 0
result_file_exists = true
```

Also verify:

```text
csv\iquantum_task_results.csv                         exists and is non-empty
csv\aqua_sla_iquantum_merged_results.csv              exists
csv\iquantum_platform_summary.csv                     executed = True
csv\aqua_sla_iquantum_platform_ablation.csv           contains the platform row
logs\iquantum_combined.log                            contains no unresolved exception
```

Quick verification command:

```bat
python -c "import json, pathlib; p=pathlib.Path(r'D:\other\AQUA-SLA\results\aqua_sla_final\iquantum_platform_integration\reports\iquantum_execution_status.json'); s=json.loads(p.read_text()); print(s); assert s['success'] and s['status']=='completed' and s['result_file_exists']"
```

If the assertion fails, report the integration as **implemented but not successfully executed** and inspect:

```text
logs\iquantum_stdout.txt
logs\iquantum_stderr.txt
logs\iquantum_combined.log
```

---

## 11. Running the final release pipeline

First apply the Script 28 consistency corrections documented above.

### Full final run

For a strict final experiment, omit `--continue-on-error` so any failed component stops the release:

```bat
conda activate aqua-sla
set IQUANTUM_HOME=D:\other\AQUA-SLA\third_party\iQuantum
set AQUA_RUN_IQUANTUM=1
cd /d D:\other\AQUA-SLA\Code
python 28_aqua_sla_master_experiment_runner.py --mode full --run-name release_v1_0
```

### Debug/collection run

Use this during development so remaining outputs are still packaged after one component fails:

```bat
python 28_aqua_sla_master_experiment_runner.py --mode full --run-name debug_run --continue-on-error
```

### Package existing results without retraining

```bat
python 28_aqua_sla_master_experiment_runner.py --mode collect-only --run-name existing_results
```

### Generate/package circuits only

```bat
python 28_aqua_sla_master_experiment_runner.py --mode circuits-only --run-name circuit_archive --continue-on-error
```

### Run/package iQuantum only

```bat
python 28_aqua_sla_master_experiment_runner.py --mode iquantum-only --run-name iquantum_validation
```

---

