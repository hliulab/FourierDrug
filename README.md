# Generalizing drug response prediction by Fourier asymmetric attention on domain generalization
![Screenshot](framework.png)

FourierDrug is a novel domain generalization framework designed to predict drug response in out-of-distribution samples, including individual cells and patient data, using only in vitro cancer cell line data for training. By leveraging adversarial domain generalization and innovative feature extraction techniques, FourierDrug addresses the limitations of traditional domain adaptation methods, which are unsuitable for unseen target domains.


## 1.  Clone Repository

~~~bash
git clone https://github.com/hliulab/FourierDrug.git
~~~

## 2. Create Python environment

We provide an environment.yml file, which can be directly used to create the environment with Conda:

```bash
cd FourierDrug
conda env create -f environment.yml
```

After the environment is created, activate it:

```bash
conda activate FourierDrug
```


## 3. Unzip datasets
Enter the datasets directory, and you will find five compressed files. Each corresponds to the four experiments conducted in this study: the LOOV GDSC dataset (hold-out), single-cell drug sensitivity (single-cell), GCGA patient drug response (patient), drug response dynamics transition (dynamic). You may extract the dataset of interest (e.g. hold-out) to proceed with the subsequent steps using the command:
~~~bash
cd datasets
unzip hold-out
~~~

## 4. Reprouduce Experiments
### 4.1 Validation on hold-out cancer types 

This experiment performs validation on the GDSC cell line drug response dataset, where all cell lines from one cancer type are held out as the test set in each fold, and the remaining data are used for training. Run the following command to train the FourierDrug model for a specific drug of interest and then perform testing:

```bash
python cell_line.py --source_path ../datasets/hold-out/cell_Afatinib.csv
```

Parameter description:

- `--source_path`: Specifies the path of input data file. This example specify the path `../datasets/hold-out/cell_Afatinib.csv`, by which the model is built for drug Afatinib. You may replace it with other datasets in the `hold-out` folder as needed.



## 4.2 Predicting single-cell drug response 

To train the FourierDrug model for a specific drug based on GDSC dataset and then apply the trained model to predict single-cell drug response, you can run the following command:

```bash
python Single-cell.py --drug_name Afatinib --source_dir ../datasets/single-cell/Afatinib.csv --target_dir ../datasets/single-cell/Target_expr_resp_z.Afatinib_tp4k.csv
```

Parameter description:

* `--drug_name`: Specifies the drug name (here **Afatinib**).
* `--source_dir`: Path to the input data file (here `../datasets/single-cell/Afatinib.csv`).
* `--target_dir`: Path to the expression profile and response data of testing file (here `../datasets/single-cell/Target_expr_resp_z.Afatinib_tp4k.csv`).


## 4.3 Predicting patient drug repsonse 
To train the FourierDrug model for a specific drug based on GDSC dataset and then apply the trained model to predict clinical patient drug response, you can run the following command:

```bash
python TCGA.py --drug_name 5-Fluorouracil --source_dir ../datasets/patient/5-Fluorouracil_cell_lines.csv --target_dir ../datasets/patient/5-Fluorouracil_patients.csv
```

Parameter description:

* `--drug_name`: Specifies the drug name (here **5-Fluorouracil**).
* `--source_dir`: Path to the input data file (here `../datasets/patient/5-Fluorouracil_cell_lines.csv`).
- `--target_dir`: Path to the expression profile and response data of testing file (here `../datasets/patient/5-Fluorouracil_patients.csv`).


## 4.4 Predicting single-cell dynamic transition toward drug resistance

Run the following command to train the model for the drug Bortezomib based on GDSC dataset, and apply it to predict the transition of single cells toward a drug-resistant state under continuous drug exposure.

```bash
python timeseq.py --source_path ../datasets/dynamic/Bortezomib_source1.csv --target_dir ../datasets/dynamic/Bortezomib_target.csv
```

## Support
For further assistance, bug reports, or to request new features, please contact us at hliu@njtech.edu.cn or open an issue on the [GitHub repository page](https://github.com/hliulab/FourierDrug).

---
