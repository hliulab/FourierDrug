![image](https://github.com/user-attachments/assets/9e5f99f0-2eeb-4e12-bcb9-cb36c3801908)

panCancerDR: A Domain Generalization Framework for Drug Response Prediction
panCancerDR is a novel domain generalization framework designed to predict drug response in out-of-distribution samples, including individual cells and patient data, using only in vitro cancer cell line data for training. By leveraging adversarial domain generalization and innovative feature extraction techniques, panCancerDR addresses the limitations of traditional domain adaptation methods, which are unsuitable for unseen target domains.
Key Features
Latent Independent Projection (LIP): A plug-and-play module that extracts expressive and decorrelated features to mitigate overfitting and improve generalization across diverse datasets.
Asymmetric Adaptive Clustering Constraint: Ensures sensitive samples are tightly clustered, while resistant samples form diverse clusters in the latent space, reflecting real-world biological diversity in drug responses.
Comprehensive Validation: The model has been rigorously evaluated on bulk RNA-seq, single-cell RNA-seq (scRNA-seq), and patient-level datasets, demonstrating superior or comparable performance to current state-of-the-art (SOTA) methods.
Contributions
Novel LIP Module: Encourages the encoder to extract informative, non-redundant features, enabling robust predictions across unseen domains.
Asymmetric Clustering for Generalization: Inspired by the distinct characteristics of sensitive and resistant cells, this approach ensures effective latent space representation for drug response prediction.
Extensive Evaluation: Achieved high performance on 13 cancer types, single-cell data, and patient-level drug response tasks, highlighting the model's versatility and predictive power.
Here’s a comprehensive README file for your project:

---

# **Drug-Specific Data Analysis and Modeling**

This repository contains code and resources for analyzing drug-specific data and building machine learning models using PyTorch. The project leverages advanced GPU computation (NVIDIA 4090 GPU) to efficiently process large datasets and train complex neural network architectures.

## **Project Overview**

The primary goal of this project is to investigate drug-specific effects and build predictive models based on corresponding datasets. Each dataset corresponds to a specific drug and is stored in a separate file.

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
  
- **Hardware Requirements**:
  - **GPU**: NVIDIA 4090 or equivalent is required for training large-scale models.
Here’s the translated README with the requested information:


Here’s a README draft modeled after the provided example:

---

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

## Model Training  
To train the model on your dataset, use the appropriate script based on the data type:  

- For single-cell data:  
  ```bash  
  python single_cell_drugX.py  
  ```  

- For patient data (TCGA):  
  ```bash  
  python TCGA_drugX.py  
  ```  

Replace `drugX` with the name of the specific drug you want to train the model on (e.g., `drugA`, `drugB`).


---



Feel free to let me know if you need additional sections or customizations!
