# Fake Profile Detection Using Machine Learning

This project focuses on detecting fake or cloned social media profiles using machine learning techniques. The objective is to classify user accounts as either **fake** or **genuine** based on profile-level features extracted from the dataset.

The initial implementation was developed in a Google Colab notebook. This repository restructures the work into a cleaner and more reproducible research-code pipeline with separate modules for data loading, preprocessing, training, evaluation, cross-validation, model saving, and inference.

---

## Paper

This work is related to the paper:

**Fake Profile Detection on Online Social Networks Using Machine Learning**

Paper link:
https://www.researchgate.net/profile/Saptak-Sil/publication/395449350_Fake_Profile_Detection_on_Online_Social_Networks_Using_Machine_Learning/links/68c44937508ac7086f595929/Fake-Profile-Detection-on-Online-Social-Networks-Using-Machine-Learning.pdf

---

## Project Overview

Fake and cloned profiles are a major problem on social media platforms. They can be used for spam, impersonation, misinformation, phishing, and other malicious activities.

This project uses supervised machine learning models to classify profiles as fake or genuine. The models are trained using structured user-profile features such as follower count, friend count, status count, favourites count, account age, profile description availability, and other profile metadata.

---

## Dataset

The project uses two CSV files:

* `fusers.csv` — fake user profiles
* `users.csv` — genuine user profiles

The two datasets are combined into a single binary classification dataset.

Label convention:

| Label | Class        |
| ----: | ------------ |
|     0 | Genuine user |
|     1 | Fake user    |

The dataset files should be placed inside the `data/` directory:

```text
data/
├── fusers.csv
└── users.csv
```

---

## Methodology

The complete pipeline follows these steps:

1. Load fake and genuine user datasets.
2. Assign binary labels to both datasets.
3. Combine both datasets into a single dataframe.
4. Remove unnecessary columns.
5. Convert textual and boolean profile attributes into machine-learning-friendly features.
6. Create account-age features from profile creation date.
7. Split the dataset into training and testing sets.
8. Scale selected numerical features using `MinMaxScaler`.
9. Train multiple machine learning models.
10. Evaluate models using standard classification metrics.
11. Run stratified k-fold cross-validation.
12. Save the trained model, scaler, and feature-column order for inference.
13. Predict whether a new profile is fake or genuine using the saved model.

---

## Models Used

The following models are implemented:

* Artificial Neural Network
* Support Vector Machine
* Logistic Regression
* Naive Bayes

The ANN is used as the primary model for final inference, while the other models are used for comparison.

---

## Evaluation Metrics

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* Classification report
* Execution time

The project supports both:

* Train-test evaluation
* Stratified k-fold cross-validation

---

## Project Structure

```text
Fake_Users/
│
├── data/
│   ├── fusers.csv
│   └── users.csv
│
├── results/
│   ├── metrics.csv
│   ├── model_comparison.png
│   ├── cv_results.csv
│   ├── cv_summary.csv
│   ├── best_ann_model.keras
│   ├── scaler.pkl
│   └── feature_columns.pkl
│
├── src/
│   ├── dataloader.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── cross_validation.py
│   ├── predict.py
│   └── utils.py
│
├── main.py
├── inference.py
├── requirements.txt
├── README.md
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd Fake_Users
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training and Evaluation

To train and evaluate all models, run:

```bash
python main.py
```

This script performs:

* Dataset loading
* Data preprocessing
* Train-test split
* Feature scaling
* ANN training
* Classical ML model training
* Train-test evaluation
* Model comparison plot generation
* Cross-validation
* Model saving

---

## Inference

After training, run inference using:

```bash
python inference.py
```

The inference script loads:

* `results/best_ann_model.keras`
* `results/scaler.pkl`
* `results/feature_columns.pkl`

It then preprocesses the new profile using the same training pipeline and predicts whether the profile is fake or genuine.

Example output:

```text
Prediction: Fake user | Fake probability: 1.000000
```

or

```text
Prediction: Genuine user | Fake probability: 0.000000
```

---

## Output Files

After running `main.py`, the following files are generated inside the `results/` folder:

```text
metrics.csv
model_comparison.png
cv_results.csv
cv_summary.csv
best_ann_model.keras
scaler.pkl
feature_columns.pkl
```

File descriptions:

| File                   | Description                                             |
| ---------------------- | ------------------------------------------------------- |
| `metrics.csv`          | Train-test evaluation results                           |
| `model_comparison.png` | Accuracy comparison plot                                |
| `cv_results.csv`       | Fold-wise cross-validation results                      |
| `cv_summary.csv`       | Mean and standard deviation of cross-validation metrics |
| `best_ann_model.keras` | Saved ANN model                                         |
| `scaler.pkl`           | Saved scaler used during preprocessing                  |
| `feature_columns.pkl`  | Saved feature column order for inference                |

---

## Results Summary

The ANN achieved the best overall classification performance, while Naive Bayes provided the fastest execution time.

In the train-test evaluation, the ANN reached approximately **98–99% accuracy**, while Naive Bayes achieved approximately **97% accuracy** with significantly lower computation time.

Cross-validation results also showed that ANN consistently performed best across folds, while Naive Bayes remained a strong lightweight baseline.

---


## Future Work

Possible future improvements include:

* Hyperparameter tuning
* API or web interface for fake profile detection
* Testing on additional social media datasets
* Model calibration for more reliable probability scores

---

## Author

**Saptak Sil and Munmun Bhattacharya**
