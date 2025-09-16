# Generalizing drug response prediction by Fourier asymmetric attention on domain generalization
![Screenshot](framework.png)
panCancerDR is a novel domain generalization framework designed to predict drug response in out-of-distribution samples, including individual cells and patient data, using only in vitro cancer cell line data for training. By leveraging adversarial domain generalization and innovative feature extraction techniques, panCancerDR addresses the limitations of traditional domain adaptation methods, which are unsuitable for unseen target domains.


## **1. 下载项目代码**

### Clone the Repository

~~~bash
git clone https://github.com/hliulab/FourierDrug.git
cd FourierDrug
~~~

## **2. 解压数据集**

~~~bash
unzip dataset.zip
~~~

## 3. 创建运行环境

项目提供了 `environment.yml` 文件，可以直接用 Conda 创建环境：

```bash
cd code
conda env create -f environment.yml
```

创建完成后，激活环境：

```bash
conda activate FourierDrug
```

------

## 4.1 Cell Line Experiment Instructions

运行以下命令来训练 **Cell Line Experiment**：

```bash
python cell_line.py --source_path ../datasets/hold-out/cell_Afatinib.csv
```

参数说明：

- `--source_path`：指定输入数据文件路径。示例中使用的是 `../datasets/hold-out/cell_Afatinib.csv`，你也可以根据需要替换为 `hold-out` 文件夹中的其他数据集。

------



------

## 4.2 Single-Cell Experiment Instructions

运行以下命令来训练 **Single-Cell Experiment**：

```bash
python Single-cell.py --drug_name Afatinib --source_dir ../datasets/single_cell/Afatinib.csv --target_dir ../datasets/single_cell/Target_expr_resp_z.Afatinib_tp4k.csv
```

参数说明：

- `--drug_name`：指定药物名称（此处为 **Afatinib**）。
- `--source_dir`：输入数据文件路径（此处为 `../datasets/single_cell/Afatinib.csv`）。
- `--target_dir`：目标表达及反应数据文件路径（此处为 `../datasets/single_cell/Target_expr_resp_z.Afatinib_tp4k.csv`）。

通过指定single_cell中的文件改变训练数据集。

## 4.3 Patient Experiment Instructions

运行以下命令来训练 **Patient Experiment**：

```bash
python TCGA.py --drug_name Afatinib --source_dir ../datasets/single_cell/Afatinib.csv --target_dir ../datasets/single_cell/Target_expr_resp_z.Afatinib_tp4k.csv
```

参数说明：

- `--drug_name`：指定药物名称（此处为 **Afatinib**）。
- `--source_dir`：输入数据文件路径（此处为 `../datasets/single_cell/Afatinib.csv`）。
- `--target_dir`：目标表达及反应数据文件路径（此处为 `../datasets/single_cell/Target_expr_resp_z.Afatinib_tp4k.csv`）。

------

通过指定patient中的文件改变训练数据集。

## 4.4 Time-Series Experiment Instructions

运行以下命令来训练 **Time-Series Experiment**：

```bash
python timeseq.py
```
