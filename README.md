# AQUA-SLA


---
# Dataset Used in AQUA-SLA

## Google Cluster Workload Traces 2019

AQUA-SLA uses a sample derived from the **Google Cluster Workload Traces 2019** dataset as the empirical workload source for SLA-risk prediction, resource-aware scheduling, hybrid quantum-classical experimentation, and iQuantum-based quantum-cloud simulation.

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



## Relationship to the iQuantum integration

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

## iQuantum installation and integration

### What iQuantum contributes

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

### Install Java and Maven

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

### Clone iQuantum

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

### prepare the required input

Script 27 searches for:

```text
locked_scheduling_assignments.csv
```

Generate it by running Script 21, or ensure a compatible locked assignment file is present under the results tree.

###  Run the integration

```bat
conda activate aqua-sla
set IQUANTUM_HOME=D:\other\AQUA-SLA\third_party\iQuantum
set AQUA_RUN_IQUANTUM=1
cd /d D:\other\AQUA-SLA\Code
python 27_aqua_sla_iquantum_platform_integration.py
```

### Verify successful integration

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
