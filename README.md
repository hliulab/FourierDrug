# Generalizing drug response prediction by Fourier asymmetric attention on domain generalization
![Screenshot](framework.png)
panCancerDR is a novel domain generalization framework designed to predict drug response in out-of-distribution samples, including individual cells and patient data, using only in vitro cancer cell line data for training. By leveraging adversarial domain generalization and innovative feature extraction techniques, panCancerDR addresses the limitations of traditional domain adaptation methods, which are unsuitable for unseen target domains.


## **Prerequisites**

Before running the code, ensure you have the following libraries installed:

- **Python Libraries**:
  - `pandas`: For data manipulation and preprocessing.
  - `torch`: PyTorch framework for building and training models.
  - `torchvision`: For additional PyTorch utilities (if required).
  - `sklearn`: For data splitting, metrics, and other utilities.
  - `tqdm`: For progress bar visualization.
  - `numpy`: For numerical operations.
  - `matplotlib`: For plotting and visualizing results.

## Installation Guide

### Clone the Repository
```bash
git clone https://anonymous.4open.science/r/panCancerDR-FC03.git
```

## Instructions for Use

### Directory Structure
- **data/**: Contains the datasets for different drugs, each compressed into a separate `.zip` file named after the corresponding drug.  
- **source/**: Includes the source code for the model. Each drug has a dedicated Python script for training, testing, and evaluation, named after the drug (e.g., `drugA.py`, `drugB.py`).  
- **trained_model/**: Stores pre-trained models, with each model saved in a subdirectory named after the corresponding drug.  

## Cell Line Experiment Instructions

To train the model on cell line experiments, follow these steps:  

1. **Download the code files**  
   - `cell_line.py`  
   - `model_test0.py`  

2. **Prepare the dataset**  
   Download the corresponding cell line dataset into the `datasets/hold-out` folder.  

3. **Configure the parameters**  
   - Set `source_path` to the path of the cell line dataset.  
   - Set `test_label` to the cancer type ID (usually a number from `0` to `22`).  

4. **Run the experiment**  
   Execute the script to perform Leave-One-Out Cross Validation (LOOCV).
## Single-Cell Experiment Instructions

To train the model on single-cell experiments, follow these steps:  

1. **Download the code files**  
   - `Single-cell.py`  
   - `model_test0.py`  

2. **Prepare the dataset**  
   Download both the cell line and single-cell datasets from the `datasets/single-cell` folder.  

3. **Configure the parameters**  
   - In `Single-cell.py`, set `drug_name` to the drug you want to validate.  
   - Set `source_dir` to the path of the corresponding **cell line dataset**.  
   - Set `target_dir` to the path of the corresponding **single-cell dataset**.  

4. **Run the experiment**  
   Execute the script to perform single-cell validation.
## Patient Experiment Instructions

To train the model on patient data experiments, follow these steps:  

1. **Download the code files**  
   - `TCGA.py`  
   - `model_test0.py`  

2. **Prepare the dataset**  
   Download both the cell line and patient datasets from the `datasets/patient` folder.  

3. **Configure the parameters**  
   - In `TCGA.py`, set `drug_name` to the drug you want to validate.  
   - Set `source_dir` to the path of the corresponding **cell line dataset**.  
   - Set `target_dir` to the path of the corresponding **patient dataset**.  

4. **Run the experiment**  
   Execute the script to perform patient data validation.
## Time-Series Experiment Instructions

To run time-series experiment validation, follow these steps:  

1. **Download the code files**  
   - `timeseq.py`  
   - `model_test0.py`  

2. **Prepare the dataset**  
   Download the required data from the `datasets/dynamic` folder.  

3. **Configure the parameters**  
   - In `timeseq.py`, set `drug_name` to `'Bortezomib'`.  
   - Set `source_dir` to the file path of **`Bortezomib_source1.csv`**.  
   - Set `target_dir` to the file path of **`Bortezomib_target.csv`**.  

4. **Run the experiment**  
   Execute the script to perform time-series experiment validation.



Feel free to let me know if you need additional sections or customizations!
