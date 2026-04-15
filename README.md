# 🧠 ANN Customer Churn Classification

A deep learning project that predicts whether a bank customer will churn (exit) using an Artificial Neural Network (ANN) built with TensorFlow/Keras. The model is deployed as an interactive web app using Streamlit.

---

## 📌 Problem Statement

Customer churn is one of the biggest challenges for banks. This project builds a binary classification model to predict if a customer will leave the bank based on their profile data, enabling proactive retention strategies.

---

## 🗂️ Project Structure

```
ANN-Classification/
│
├── app.py                      # Streamlit web application
├── experiments.ipynb           # Model training notebook
├── model.h5                    # Trained ANN model
├── label_encoder_gender.pkl    # Label encoder for Gender
├── onehot_encoder_geo.pkl      # One-hot encoder for Geography
├── scaler.pkl                  # Standard scaler for features
├── Churn_Modelling.csv         # Dataset
├── requirements.txt            # Python dependencies
└── logs/                       # TensorBoard training logs
```

---

## 🧬 Dataset

The dataset contains 10,000 bank customer records with the following features:

| Feature | Description |
|---|---|
| CreditScore | Customer credit score |
| Geography | Country (France, Germany, Spain) |
| Gender | Male / Female |
| Age | Customer age |
| Tenure | Years with the bank |
| Balance | Account balance |
| NumOfProducts | Number of bank products used |
| HasCrCard | Has credit card (1/0) |
| IsActiveMember | Active member (1/0) |
| EstimatedSalary | Estimated annual salary |
| **Exited** | **Target: 1 = Churned, 0 = Stayed** |

---

## 🏗️ Model Architecture

```
Input Layer  →  64 neurons (ReLU)
                    ↓
Hidden Layer →  32 neurons (ReLU)
                    ↓
Output Layer →   1 neuron (Sigmoid)
```

- **Optimizer:** Adam (learning rate = 0.01)
- **Loss:** Binary Crossentropy
- **Metrics:** Accuracy
- **Callbacks:** EarlyStopping, TensorBoard
- **Epochs:** 100

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/amnsingh05/ANN-Classification-Churn.git
cd ann-classification-churn
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the App

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`

---

## 📊 TensorBoard (Training Visualization)

To view training metrics:

```bash
tensorboard --logdir logs/fit
```

Then open `http://localhost:6006`

---

## 📦 Requirements

```
tensorflow==2.15.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
tensorboard==2.15.2
matplotlib==3.9.0
streamlit==1.35.0
protobuf==4.25.3
```

---

🌐 Live Demo

👉 https://ann-classification-churn-bxf9tjypx5gptkngugaa3y.streamlit.app/

---

## 🛠️ Tech Stack

- **Python 3.11**
- **TensorFlow / Keras** — ANN model
- **Scikit-learn** — Preprocessing
- **Pandas / NumPy** — Data manipulation
- **Streamlit** — Web app
- **TensorBoard** — Training visualization

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).