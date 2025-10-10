<div align="center">
  
# Loan Approval Prediction

</div>

This project predicts whether a loan application will be approved using machine learning. The workflow was developed in **Google Colab** and focuses on handling imbalanced data, feature encoding, and model optimization for maximum precision, recall, and F1-score.

---

## Folder Structure
```BASH
loan-approval-prediction/
│
├── data/
│ ├── loan_approval_dataset.csv # Main dataset CSV
│ └── download_dataset.py # Script to download dataset from KaggleHub
│
├── notebook/
│ └── Loan_Approval_Prediction_Description.ipynb # Colab notebook with analysis
│
├── requirements.txt # Python dependencies
└── README.md # Project overview
```

---

## Files Description

- **data/loan_approval_dataset.csv**: Raw dataset used for training and evaluation.  
- **notebook/Loan_Approval_Prediction_Description.ipynb**: Colab notebook containing preprocessing, SMOTE oversampling, Random Forest training, and hyperparameter tuning.  
- **requirements.txt**: Lists Python packages required to run the notebook.  
- **README.md**: Project overview and results.

---

## Results

- **Random Forest Accuracy:** ~97.8%  
- **Precision / Recall / F1-score:** ~0.98 for both classes  
- SMOTE is used to handle class imbalance and improve model performance.

---

## Author
**Gazanfar Ali**  
BSc Artificial Intelligence, UMT Lahore  
 
