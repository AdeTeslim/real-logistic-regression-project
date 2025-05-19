# Logistic Regression Project

This project applies logistic regression using scikit-learn to classify a binary outcome based on input features. It includes data preprocessing, model training, evaluation, and prediction on custom inputs.

## 📌 Project Overview

The goal of this project is to:

* Load and preprocess a dataset
* Train a logistic regression model
* Evaluate the model using classification metrics
* Allow manual testing of the model with new input values

## 🧰 Technologies Used

* Python
* Jupyter Notebook
* pandas
* scikit-learn
* NumPy
* Matplotlib / Seaborn (for visualization)

## 📁 Project Structure

```bash
.
├── Logistic Regression.ipynb           # Jupyter notebook with full workflow
├── README.md                           # Project documentation
```

## 🧪 Model Evaluation

The model was evaluated using the `classification_report` from scikit-learn. Metrics include:

* Accuracy
* Precision
* Recall
* F1-score

Output:

```
              precision    recall  f1-score   support

           0       0.96      0.98      0.97       162
           1       0.98      0.96      0.97       168

    accuracy                           0.97       330
   macro avg       0.97      0.97      0.97       330
weighted avg       0.97      0.97      0.97       330
```

## 🔍 Manual Testing

After training, the model can be tested with new inputs:

```python
import numpy as np

# Example input (replace with your real features)
new_data = np.array([[45, 1, 120.0, 85.0, 1]])  # Sample data

# Apply same scaling if StandardScaler was used
new_data_scaled = scaler.transform(new_data)

# Predict
prediction = model.predict(new_data_scaled)
print("Prediction:", prediction[0])

# Optional: view probabilities
proba = model.predict_proba(new_data_scaled)
print("Probability of class 0:", proba[0][0])
print("Probability of class 1:", proba[0][1])
```

## 🚀 How to Run

1. Clone the repo:

```bash
git clone https://github.com/AdeTeslim/your-repo-name.git
cd your-repo-name
```

2. Make sure to install all dependencies



3. Launch the Jupyter Notebook:

```bash
jupyter notebook
```

4. Run through the notebook step by step to train and evaluate the model.

## 📌 Notes

* Ensure that any test inputs you provide to the model match the structure and scale of the training features.
* You can tune hyperparameters such as `max_iter` and `solver` if the model doesn’t converge.

## 📬 Contact

For questions or collaboration, feel free to reach out at \[[seunmomodu123@gmail.com]] or open an issue.
