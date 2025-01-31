# 🏛️ German Parliamentary Speech Classifier

This project provides a complete pipeline for **classifying parliamentary speeches** based on **topics** and **sentiment analysis**.  
It includes **data preprocessing, model training, evaluation, hyperparameter optimization, and interpretability**.

🔗 **Data Gathering Pipeline:** Need the raw data? No problem! Check out the dedicated pipeline here: [parliamentary-speech-pipeline](https://github.com/ptzlukas/parliamentary-speech-pipeline.git) 🚀  

---

## 🚀 Features

👉 **Data Preprocessing (`preprocess.py`)**  
   - Converts a single training dataset into two specific datasets:  
     - **Topic Classification**  
     - **Sentiment Analysis**  
   - The original training dataset can be downloaded from:  
     📂 [Speicherwolke Link](https://speicherwolke.uni-leipzig.de/index.php/s/Mp5XdqbHZ3J2t7g?path=%2FAbgabe)

👉 **Model Training (`train_topic.py` & `train_sentiment.py`)**  
   - Trains models for **topic classification** and **sentiment analysis** via CLI  
   - Requires a `--run-name` parameter for MLflow logging  
   - Supports multiple model types, cross-validation & hyperparameter tuning  
   - Loads configuration from YAML files  
   - Saves trained models in the `models/` directory  
   - Generates a **confusion matrix** and an **ROC curve** in the `plots/` directory  

👉 **MLflow Integration**  
   - Automatically logs all training experiments  
   - Saves training parameters, metrics & model artifacts  
   - Start MLflow UI with:  
     ```bash
     mlflow ui
     ```  
   - View all past experiments and results in the MLflow UI  

👉 **Custom Word2Vec Model (`word2vec_creation.py`)**  
   - A dedicated **Word2Vec model** for the topic classifier  
   - Can be retrained using new data from the [Speicherwolke](https://speicherwolke.uni-leipzig.de/index.php/s/Mp5XdqbHZ3J2t7g?path=%2FAbgabe)

👉 **Model Interpretability with LIME (`topic_lime.py`)**  
   - Uses **LIME (Local Interpretable Model-agnostic Explanations)**  
   - Helps understand model decisions better  

---

## ⚙️ Installation & Setup

### 1️⃣ Create & activate Conda environment  
```bash
conda env create -f environment.yaml
conda activate parliamentary-speech-classifier
```

### 2️⃣ Prepare the data  
```bash
python src/preprocess.py
```

### 3️⃣ Start MLflow Dashboard  
```bash
mlflow ui
```
Then open in browser: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

### 4️⃣ Train a model  

#### 📝 Topic Classification  
```bash
python src/train_topic.py --run-name "my_experiment"
```

#### 📝 Sentiment Analysis  
```bash
python src/train_sentiment.py --run-name "my_experiment"
```

**Note:** The `--run-name` parameter is required for MLflow logging.

---

## 📊 Parameter Configuration

The `params_sentiment.yaml` and `params_topic.yaml` files allow you to:

- ✅ Select different model types (**Logistic Regression, SVM, XGBoost, Bagging Classifier**)
- ✅ Enable **Cross-Validation & Hyperparameter Optimization**
- ✅ Adjust the **Vectorization Method** (TF-IDF or spaCy or customm Word2Vec)
- ✅ Track experiments in **MLflow**

---

## 🏆 Results & Visualizations

After training, the following files will be generated:

- 📁 **Models:** Saved in the `models/` directory  
- 📊 **ROC Curve & Confusion Matrix:** Saved in the `plots/` directory  
- 📈 **MLflow Logging:** All metrics and results are visible in the MLflow UI  

---
## 📂 Project Structure

```
PARLIAMENTARY-SPEECH-CLASSIFIER
│── data/                     # Preprocessed datasets
│   ├── processed_sentiment_data.csv
│   ├── processed_topic_data.csv
│   ├── trainings_dataset.csv
│
│── models/                   # Trained models
│   ├── random_forest_model_sentiment.pkl
│   ├── trained_models_bagging.pkl
│   ├── word2vec.model
│   ├── word2vec.model.wv.vectors.npy
│
│── plots/                    # Stored confusion matrices & ROC curves
│
│── src/                      # Main project scripts
│   ├── hyperparameter_optimization/  # Hyperparameter tuning scripts
│   ├── interpretability/            # LIME-based model interpretability
│   ├── preprocess.py                 # Data preprocessing
│   ├── train_sentiment.py             # Sentiment model training
│   ├── train_topic.py                 # Topic classification training
│   ├── word2vec_creation.py           # Word2Vec training
│
│── .gitignore
│── environment.yaml          # Conda environment with required dependencies
│── params_sentiment.yaml      # Sentiment model parameters
│── params_topic.yaml          # Topic classification model parameters
```

---
