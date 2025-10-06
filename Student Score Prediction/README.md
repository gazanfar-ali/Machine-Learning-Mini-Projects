<div align="center">
  
# 🎓 Student Score Prediction

</div>


This project focuses on predicting **student exam scores (GPA)** based on multiple academic and lifestyle factors.  
The work was carried out as part of my **Machine Learning Internship at ELEVVO**.  

---

## 📌 Project Overview
- **Objective**: Build models to predict student performance and identify key factors that affect GPA.  
- **Dataset**: [Student Performance Dataset (Kaggle)](https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset).  
- **Approach**: Applied regression models, performed feature engineering, and evaluated model accuracy.  

---

## 📂 Repository Structure
```bash
student-performance-prediction/
│
├── data/
│   └── students-performance.csv      # Dataset (if allowed, otherwise add Kaggle link in README)
│
├── notebooks/
│   └── student_performance.ipynb     # Your Colab notebook (main project)
│
├── results/
│   ├── correlation_heatmap.png       # Visualization outputs
│   ├── actual_vs_predicted.png       # Scatter plot
│   └── model_comparison_table.md     # Performance summary
│
├── README.md                         # Project overview
└── requirements.txt                  # Python dependencies
```
---

## ⚙️ Methodology
1. **Data Collection**  
   - Downloaded dataset from Kaggle using `kagglehub`.
   - Prepared data for analysis and model training.  

2. **Data Cleaning & Visualization**  
   - Checked missing values and ensured data consistency.  
   - Used **matplotlib** and **seaborn** for exploratory analysis.  

3. **Model Training**  
   - Linear Regression (baseline).  
   - Polynomial Regression (degree=2).  
   - Full Feature Linear Regression (all features included).  

4. **Evaluation Metrics**  
   - R² Score (explained variance).  
   - RMSE (root mean squared error).  

---

## 📊 Results Summary
| Model                      | Features Used                          | R² Score | RMSE   |
|-----------------------------|----------------------------------------|----------|--------|
| Linear Regression           | StudyTimeWeekly, Absences, Tutoring, ParentalSupport | 0.9283 | 0.2434 |
| Polynomial Regression (d=2) | Same 4 features                        | 0.9278   | 0.2444 |
| Full Feature Model          | All features except StudentID & target | **0.9532** | **0.1966** |

---

## 🔑 Key Insights
- **Linear Regression** already explained ~93% of the variance.  
- **Polynomial Regression** added complexity but did not improve performance.  
- **Full Feature Model** performed best (R² ≈ 0.95, RMSE ≈ 0.19).  
- GPA is influenced not only by study time and absences but also by **lifestyle and family factors**.  

---

## 🔗 Open in Google Colab

You can open and run this project notebook directly using Colab:
```bash
https://colab.research.google.com/drive/1_FaMEowvqhR0gHj3k6f_ImaOL56ulC6l?usp=sharing
```

---

## 🛠️ Tech Stack
- **Language**: Python  
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn  
- **Platform**: Google Colab, Jupyter Notebook  

---

## 📌 Author
**Gazanfar Ali**  
BSc Artificial Intelligence, UMT Lahore  
Machine Learning Intern @ ELEVVO  
