# From Bag-of-Words to BERT: Sentiment Analysis of Movie Reviews

## 📌 Project Overview
This project focuses on building and comparing multiple Natural Language Processing (NLP)
models to classify movie reviews into **positive** or **negative** sentiments.
The project demonstrates the evolution of sentiment analysis techniques,
starting from traditional machine learning approaches to transformer-based deep learning models.

The IMDb Movie Reviews dataset is used as a benchmark to evaluate model performance.

---

## 🎯 Objectives
- Build a complete NLP pipeline for sentiment analysis
- Apply text preprocessing and feature extraction techniques
- Train and evaluate multiple machine learning models
- Compare traditional ML models with a transformer-based model (BERT)
- Analyze performance trade-offs between efficiency and accuracy

---

## 📂 Dataset
- **IMDb Dataset of 50K Movie Reviews**
- Binary classification: Positive / Negative
- Balanced dataset with 50,000 reviews

---

## 🧠 Methodology

### 1️⃣ Exploratory Data Analysis (EDA)
- Sentiment distribution visualization
- Review length analysis
- Sample inspection from each sentiment class

### 2️⃣ Text Preprocessing (Machine Learning Pipeline)
- Lowercasing
- Removal of punctuation and non-alphabetic characters
- Tokenization
- Stopword removal
- Lemmatization

### 3️⃣ Feature Extraction
- TF-IDF (Term Frequency–Inverse Document Frequency)
- Unigrams and bigrams
- Feature dimensionality control

---

## 🤖 Models Used

### 🔹 Traditional Machine Learning Models
- Logistic Regression
- Naive Bayes
- Support Vector Machine (SVM)
- SGD Classifier
- Passive Aggressive Classifier
- Decision Tree
- Random Forest

### 🔹 Deep Learning Model
- **BERT (bert-base-uncased)**
- Fine-tuned for binary sentiment classification

---

## 📊 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix (for selected models)

---

## 📈 Results Summary
- Traditional ML models achieved strong baseline performance using TF-IDF features.
- Logistic Regression and SVM provided the best results among classical models.
- Fine-tuning BERT improved contextual understanding and achieved competitive performance,
  demonstrating the advantages of transformer-based models for sentiment analysis.

---

## ⚖️ Model Comparison
The project highlights the trade-offs between:
- **Traditional ML models**: Faster training and lower computational cost
- **BERT-based model**: Better contextual understanding with higher computational requirements

---

## 🛠️ Technologies & Libraries
- Python
- Pandas, NumPy
- NLTK
- Scikit-learn
- Hugging Face Transformers
- PyTorch
- Matplotlib, Seaborn

---

## 🚀 Future Improvements
- Hyperparameter tuning for BERT
- Using larger transformer models
- Applying the pipeline to Arabic sentiment analysis
- Deploying the model as a web application

