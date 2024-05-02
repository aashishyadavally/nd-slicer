# Artifact for "Predictive Program Slicing via Execution Knowledge-Guided Dynamic Dependence Learning"

ND-Slicer is a learning-based predictive slicing tool. The source code, data, and model artifacts are publicly available on [GitHub](https://github.com/aashishyadavally/nd-slicer) and [Zenodo]().

## Table of Contents

* [Getting Started](#getting-started)
  - [Setup](#setup)
    - [Hardware Requirements](#hardware-requirements)
    - [Project Environment](#project-environment)
  - [Directory Structure](#directory-structure)
  - [Usage Guide](#usage-guide)
* [Contributing Guidelines](#contributing-guidelines)
* [License](#license)

## Getting Started
This section describes the preqrequisites, and contains instructions, to get the project up and running.

### Setup 

#### Hardware Requirements
``ND-Slicer`` requires a GPU to run *fast* and produce the results. On machines without a GPU, note that it can be notoriously slow.

#### Project Environment
Currently, ``ND-Slicer`` works well on Ubuntu OS, and can be set up easily with all the prerequisite packages by following these instructions (if ``conda`` is already installed, update to the latest version with ``conda update conda``, and skip steps 1 - 3):
  1. Download the latest, appropriate version of [conda](https://repo.anaconda.com/miniconda/) for your machine (tested with ``conda 23.11.0``).
  2. Install  it by running the `conda_install.sh` file, with the command:
     ```bash
     $ bash conda_install.sh
     ```
  3. Add `conda` to bash profile:
     ```bash
     $ source ~/.bashrc
     ```
  4. Navigate to ``nd-slicer`` (top-level directory) and create a conda virtual environment with the included `environment.yml` file using the following command:
     
     ```bash
     $ conda env create -f environment.yml
     ```

     To test successful installation, make sure ``autoslicer`` appears in the list of conda environments returned with ``conda env list``.
  5. Activate the virtual environment with the following command:
     
     ```bash
     $ conda activate autoslicer
     ```

### Directory Structure

#### 1. Data Artifacts
Navigate to ``nd-slicer/data/`` to find:
* Raw dataset file (``codenetmut_test.json``) -- use these files to build train/validation/test splits from scratch.
* Processed dataset files (``{full|train|val|test}-dataset.jsonl``) -- use these files to benchmark predictive slicing approaches, or replicate intrinsic evaluation results in the paper (Section 5, Table 1).

#### 2. Model Artifacts
Navigate to ``nd-slicer/outputs/`` to find the trained model weights for CodeExecutor (B2), GraphCodeBERT+PointerTransformer (B3), GraphCodeBERT+Transformer (B4), CodeExecutor+PointerTransformer (B5), and CodeExecutor+Transformer (B6).

#### 3. Code
Navigate to ``nd-slicer/src/`` to find the source code for running all the experiments/using ND-Slicer to predict dynamic slices for a Python program.

### Usage Guide
See [link](https://github.com/aashishyadavally/nd-slicer/tree/main/src/README.md) for details about replicating results in the paper, as well as using ``ND-Slicer`` to predict slices for Python programs. Here's an executive summary of the same:

| Experiment                                        | Table # in Paper | Run Command(s)                                                        | Model Artifact(s) for Direct Inference |
| ---                                               | :----:           | :---:                                                                 | :---:                                  |
| **(RQ1)** Intrinsic Evaluation on *Executable Python Code* | 1       | [click here](src/README.md/#intrinsic-evaluation-on-executable-python-code)   | [CodeExecutor (B2)]()                 |
|                                                   |                  |                                                                       | [GraphCodeBERT + PointerTransformer (B3)]()          |
|                                                   |                  |                                                                       | [GraphCodeBERT + Transformer (B4)]()          |
|                                                   |                  |                                                                       | [CodeExecutor + PointerTransformer (B5)]()          |
|                                                   |                  |                                                                       | [CodeExecutor + Transformer (B6)]()          |
| **(RQ2)** Intrinsic Evaluation on *Non-Executable Python Code*  | -  |  [click here](src/README.md/#intrinsic-evaluation-on-partial-code)    | [CodeExecutor + Transformer (B6)]()  |
| **(RQ3)** Extrinic Evaluation (Crash Detection)   | 2                |  [click here](src/README.md/#extrinsic-evaluation)                    | [CodeExecutor + Transformer (B6)]()
| **(RQ4)** Qualitative Analysis (Statement Types)  | 3                |  [click here](src/README.md/#statement-types/)                        |  [CodeExecutor + Transformer (B6)]()
| **(RQ5)** Qualitative Analysis (Execution Iterations)  | 4           |  [click here](src/README.md/#execution-iterations/)                   |  [CodeExecutor + Transformer (B6)]()
| **(RQ6)** Inter-Procedural Analysis               | -                |  [click here](src/README.md/#inter-procedural-analysis/)              |  [CodeExecutor + Transformer (B6)]()

## Contributing Guidelines
There are no specific guidelines for contributing, apart from a few general guidelines we tried to follow, such as:
* Code should follow PEP8 standards as closely as possible
* Code should carry appropriate comments, wherever necessary, and follow the docstring convention in the repository.

If you see something that could be improved, send a pull request! 
We are always happy to look at improvements, to ensure that `nd-slicer`, as a project, is the best version of itself. 

If you think something should be done differently (or is just-plain-broken), please create an issue.

## License
See the [LICENSE](https://github.com/aashishyadavally/nd-slicer/tree/main/LICENSE) file for more details.
