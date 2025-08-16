# machine_learning_models_simulation



# Machine Learning Models Simulation

An interactive web platform that allows users to upload CSV datasets and test various machine learning models, including **KNN** and **Logistic Regression**, while providing detailed performance metrics.

---

## Features

- Upload CSV datasets and select the desired machine learning model
- Evaluate models using:
  - Accuracy
  - Confusion Matrix
  - Precision
  - Recall
  - F1-score
- Web-based interface built with **Flask**
- Backend implementation using **Python, scikit-learn, and Pandas**
- Designed to provide an interactive environment for experimenting with different models on custom datasets
- Example CSV datasets included for quick testing (`examples/` folder)

---

## Installation

1. Clone the repository:
bash

git clone https://github.com/airemc/machine_learning_models_simulation.git

2.Navigate to the project directory:

cd machine_learning_models_simulation

3.Install required dependencies:

pip install -r requirements.txt

4.Run the Flask application:

python app.py


Usage
1. Open your web browser and go to http://127.0.0.1:8000
2. Upload your CSV dataset( or use heart_disease_uci.csv )
3. Select a machine learning model (e.g., KNN, Logistic Regression)
4. Click "Train" to evaluate the model
5. View performance metrics and results on the web interface
