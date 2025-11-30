# 📧 Spam Email Detection using Machine Learning

This project demonstrates a complete machine learning workflow to classify emails as **Spam** or **Ham (Not Spam)**.  
It covers preprocessing, feature engineering, model training, evaluation, and comparison of multiple machine learning algorithms to identify the most accurate spam classifier.

---

## 🚀 Features

- Text Preprocessing:
  - Remove punctuations
  - Tokenization
  - Stop words filtering
  - Stemming using NLTK
- TF-IDF Vectorization
- Multiple ML Models Tested
- Confusion Matrix & Classification Metrics
- Model comparison visualization
- Export & reuse trained model

---

## 🧠 Model Workflow

📥 Load dataset
⬇️
🧹 Text pre-processing (stemming + stopwords removal)
⬇️
🧮 Convert text to TF-IDF numerical features
⬇️
🤖 Train and compare ML models
⬇️
📈 Evaluate metrics & accuracy
⬇️
🏆 Save and deploy best-performing model

yaml
Copy code

---

## 📊 Results

After evaluating all models, the best-performing classifier achieved:

> 🎯 **Final Accuracy: 97.64%**

---

## 🗂 Dataset

Dataset used: **Spam Mail Dataset** from Kaggle.

🔗 *(Insert dataset link if required)*

---

## 🛠 Technologies & Libraries Used

| Category | Tools / Libraries |
|---------|------------------|
| Language | Python |
| Data Handling | pandas, numpy |
| Visualization | matplotlib, seaborn |
| NLP Processing | nltk |
| Machine Learning | scikit-learn, xgboost |

---

## ▶️ How to Run This Project Locally

Follow the steps below to run the notebook and model on your system:

### 1️⃣ Clone the Repository
```bash
git clone <repository-url>
cd spam-email-detection
```

2️⃣ Install Dependencies
Make sure Python is installed, then run:
```bash
pip install -r requirements.txt
```

3️⃣ Run Jupyter Notebook
```bash
jupyter notebook Spam_Email_Detection.ipynb
```

🧪 Run Predictions on Your Own Text
If you exported the trained model (model.pkl) and TF-IDF vectorizer (vectorizer.pkl), use this script:

```bash
import pickle
```

# Load saved model and vectorizer
```bash
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
```

# Example email text
sample = ["Congratulations! You have won a FREE iPhone! Click now to claim."]

# Transform and predict
```bash
vectorized = vectorizer.transform(sample)
prediction = model.predict(vectorized)
print("Prediction:", "SPAM" if prediction == 1 else "HAM")
```

📄 License
This project is licensed under the MIT License.

🙌 Acknowledgments
Dataset source: Kaggle
Libraries: NLTK, Scikit-Learn, Matplotlib

---
