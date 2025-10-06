# 📊 Model Comparison – Student Score Prediction

This document summarizes the performance of different models applied to the **Student Performance Dataset**.

---

## 🔹 Models Evaluated

| Model                      | Features Used                          | R² Score | RMSE   |
|-----------------------------|----------------------------------------|----------|--------|
| **Linear Regression**       | Study hours, Sleep hours, Classes missed, Participation | 0.9283   | 0.2434 |
| **Polynomial Regression** (degree=2) | Same as Linear Regression            | 0.9278   | 0.2444 |
| **Full Feature Model**      | All available features (study + lifestyle factors) | **0.9532** | **0.1966** |

---

## 🔹 Observations

- **Linear Regression** already performs very well with high R² (~92.8%).  
- **Polynomial Regression** (degree=2) did not improve results significantly and slightly increased error.  
- **Full Feature Model** achieved the **best performance** with R² = 95.3% and the lowest RMSE = 0.1966, showing that additional features like **sleep** and **participation** positively impact predictions.  

---

✅ **Conclusion**: Using all features in a Linear Regression model is the most effective approach for predicting student exam scores in this dataset.
