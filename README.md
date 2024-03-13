# Code and Data for "Deep learning–assisted insights into molecular-level transport in heterogeneous electrolyte films on electrodes"


## Overview

This repository holds the code developed for the paper "Deep learning–assisted insights into molecular-level transport in heterogeneous electrolyte films on electrodes". The code provides guidance and resources for replicating the processes mentioned in the paper.

## Directory Structure

- `Part1.Molecular_dynamics_simulation/`: Contains code and raw data for the molecular dynamics section.
- `Part2.Machine_learning/`: Contains code and raw data for the machine learning section.

## Environment Setup

- The code in the path `Part1.Molecular_dynamics_simulation/` all runs in LAMMPS software package.
- The code in the path `Part2.Machine_learning/` all runs in Python. Before using the code, please configure the appropriate environment and install the required dependency packages:


    ```bash
    pip install -r requirements.txt
    ```

## Notes on Reproducibility

- Please note that due to the inherent randomness associated with certain model architectures and hardware configurations, the parameters, weight updates, or results obtained from these models may exhibit variability. As a result, when replicating experiments using this codebase, slight deviations from the results presented in the paper might occur. However, these variations are typically within an acceptable range and do not significantly affect the overall findings or conclusions.
- We have saved the parameters of the CNN model used in the paper to path `Part2.Machine_learning/Part1.Model_trainning/CNN_training_data/max_net` for your reference.

## Contribution
If you wish to contribute to this project or have found issues, please raise an Issue or submit a Pull Request.

## Contact
For any inquiries or suggestions, please contact:
- `Linhao Fan`: lhfan@tju.edu.cn
