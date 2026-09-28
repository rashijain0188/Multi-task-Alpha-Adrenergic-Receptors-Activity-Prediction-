# A Multi-Task D-MPNN Framework for Predicting Alpha-1 Adrenergic Receptor Antagonists 



This repository contains a multi-task directed message-passing neural network (D-MPNN)-based prediction pipeline for identifying potential **α1-adrenergic receptor antagonists** from molecular SMILES strings. The model predicts whether a compound is active or inactive for the three subtypes: **ADRA1A, ADRA1B, and ADRA1D** and generates molecular interpretation results using **Integrated Gradients**.

\---

## Features

* Predicts antagonists from SMILES
* Multi-task D-MPNN (directed message-passing neural network) Model
* Molecular graph interpretation using Integrated Gradients 
* Highlights features driving to positive class in green and others in red
* Batch prediction from CSV input
* Saves prediction results and molecular images automatically

\---

## Requirements

Install the following Python packages before running the prediction script:

```
pip install numpy pandas rdkit torch torch-geometric
```



Versions:

numpy==1.26.4

pandas==2.3.3

rdkit==2025.09.6

torch==2.11.0

torch-geometric==2.7.0



With versions: 

```
pip install numpy==1.26.4 pandas==2.3.3 rdkit==2025.09.6 torch==2.11.0 torch-geometric==2.7.0
```





## Input File

Create a "Molecules\_to\_check.csv" file containing a column named `SMILES`.

Example:

```csv
SMILES
CCO
CCN(CC)CC
```

Save the "Molecules\_to\_check.csv" file in the same folder as scripts.

\---

## Running the Prediction

1. Download or clone the repository.
2. dmpnn\_core contains the featurisation and architecture.
3. Place "Molecules\_to\_check.csv" in the project directory.
4. Run the prediction script:

```

python dmpnn\_predict\_and\_explain.py \\

   --csv my\_molecules.csv \\

   --smiles\_col SMILES \\

   --model\_dir model \\

   --hp\_config hpo/best\_hyperparameters.json \\

   --out\_dir results

```



or in Sypder with

run\_predict\_explain\_spyder.py

## 

## Output

After execution, a folder named `predict\_out` will be generated containing:

### `predictions.csv`

### `explanations/`





## Notes

* The input CSV must contain a valid `SMILES` column.
* Invalid SMILES entries will be skipped automatically.
* All model files and scripts must be present in the same project directory before running predictions.

\---



## Publication

If you use this repository or the associated model in your research, please cite the related publication:

```text
Rashi Jain, Prabha Garg. A Multi-Task D-MPNN Framework for Predicting Alpha-1 Adrenergic Receptor Antagonists with a Drug Repurposing Study Targeting ADRA1D. 
```

\---

