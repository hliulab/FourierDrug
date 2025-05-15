# Cross-domain pan-cancer drug response prediction by latent independent projection for asymmetric constrained domain generalization
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
