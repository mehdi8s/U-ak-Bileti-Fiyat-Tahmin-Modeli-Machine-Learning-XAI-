# ✈️ Airline Flight Price Prediction

> A comprehensive machine learning project to predict airline ticket prices using real-world flight data from India.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Pipeline](#project-pipeline)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation & Usage](#installation--usage)
- [Results](#results)
- [Project Structure](#project-structure)

---

## 📌 Overview

This project builds and evaluates multiple machine learning regression models to predict airline ticket prices. The goal is to find the most accurate model that can estimate flight prices based on various features such as airline, route, departure time, duration, and number of stops.

**Problem Type:** Regression  
**Objective:** Minimize prediction error (MAE, RMSE) and maximize R² score

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Total Records** | 300,153 entries |
| **Features** | 12 columns |
| **After Outlier Removal** | 300,030 entries |
| **Target Variable** | `price` (in Indian Rupees) |

### Price Statistics
| Metric | Value |
|---|---|
| Min | ₹1,105 |
| Mean | ₹20,890 |
| Max | ₹1,23,071 |

### Dataset Columns

| Column | Type | Description |
|---|---|---|
| `airline` | Categorical | Airline carrier name (6 unique) |
| `flight` | Categorical | Flight code (1,561 unique) |
| `source_city` | Categorical | City of departure (6 unique) |
| `departure_time` | Categorical | Time of departure (6 slots: Early Morning, Morning, Afternoon, Evening, Night, Late Night) |
| `stops` | Categorical | Number of stops (zero, one, two\_or\_more) |
| `arrival_time` | Categorical | Time of arrival (6 slots) |
| `destination_city` | Categorical | City of arrival (6 unique) |
| `class` | Categorical | Seat class (Economy / Business) |
| `duration` | Numerical | Flight duration in hours |
| `days_left` | Numerical | Days left before departure |
| `price` | Numerical | 🎯 Target – Ticket price (₹) |

### Airlines Distribution

| Airline | Count |
|---|---|
| Vistara | 127,859 (most frequent) |
| AirAsia | — |
| Indigo | — |
| GO\_FIRST | — |
| Air India | — |
| SpiceJet | — |

---

## 🔬 Project Pipeline

The project follows a structured, step-by-step machine learning pipeline:

```
1. Data Loading          →  Load raw CSV dataset
2. Statistical Analysis  →  Descriptive stats, categorical summaries
3. Outlier Removal       →  Remove extreme price outliers (IQR method)
4. Visualization         →  Histograms, scatter plots, correlation heatmap
5. Preprocessing         →  Label encoding, feature engineering
6. Model Training        →  Multiple regression algorithms
7. Hyperparameter Tuning →  Grid Search / Randomized Search CV
8. Evaluation            →  MAE, RMSE, R² comparison
9. Prediction            →  Final model inference
```

### Step Details

#### 1️⃣ Veri Yükleme (Data Loading)
- Reads the raw flight dataset
- Displays first 5 rows, data types, and null value checks
- Initial shape: **300,153 rows × 12 columns**

#### 2️⃣ İstatistiksel Analiz (Statistical Analysis)
- Descriptive statistics for numerical features (`duration`, `days_left`, `price`)
- Categorical variable summaries (unique counts, most frequent values)
- **No missing values** detected in the dataset

#### 3️⃣ Veri Ön İşleme (Data Preprocessing)
- **Outlier removal** using IQR: reduced to 300,030 rows
- **Label Encoding** for all categorical features
- Feature selection: dropped `Unnamed: 0` index column and `flight` code
- **Train/Test Split**: 80% training / 20% testing

#### 4️⃣ Veri Görselleştirme (Data Visualization)
- Price distribution histogram
- Price vs. Duration scatter plot
- Price vs. Days Left scatter plot
- Price by Airline box plot
- Price by Class comparison
- Correlation heatmap of numerical features

#### 5️⃣ Model Eğitimi & Değerlendirme (Model Training & Evaluation)
Multiple regression models are trained and compared:

- **Linear Regression**
- **Ridge Regression**
- **Lasso Regression**
- **Decision Tree Regressor**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **XGBoost Regressor**

#### 6️⃣ Hiperparametre Optimizasyonu (Hyperparameter Tuning)
- Best model further tuned using cross-validated grid/random search
- Optimal parameters selected based on validation score

---

## 🛠️ Features

- ✅ End-to-end ML pipeline from raw data to final model
- ✅ Comprehensive EDA with multiple visualizations
- ✅ Comparison of 7 different regression algorithms
- ✅ Outlier detection and removal
- ✅ Categorical encoding for all string features
- ✅ Hyperparameter tuning for best model
- ✅ Clear console-based step progress output

---

## 💻 Technologies Used

| Technology | Purpose |
|---|---|
| ![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python) | Core programming language |
| **Pandas** | Data loading, manipulation, and analysis |
| **NumPy** | Numerical computations |
| **Matplotlib** | Data visualization (histograms, scatter plots) |
| **Seaborn** | Statistical data visualization (heatmaps, box plots) |
| **Scikit-learn** | ML models, preprocessing, cross-validation |
| **XGBoost** | Gradient boosting implementation |
| **Jupyter Notebook** | Interactive development environment |

---

## 🚀 Installation & Usage

### Prerequisites

Make sure you have Python 3.8+ installed.

### 1. Clone the repository

```bash
git clone https://github.com/your-username/airline-price-prediction.git
cd airline-price-prediction
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

### 3. Run the notebook

```bash
jupyter notebook main.ipynb
```

Or run all cells at once directly:

```bash
jupyter nbconvert --to notebook --execute main.ipynb
```

---

## 📈 Results

Models were evaluated using the following metrics:

| Metric | Description |
|---|---|
| **MAE** | Mean Absolute Error – average absolute difference |
| **RMSE** | Root Mean Squared Error – penalizes large errors |
| **R²** | Coefficient of Determination – explains variance |

The **ensemble/boosting models** (Random Forest, Gradient Boosting, XGBoost) significantly outperform linear models due to the non-linear relationships in flight pricing data.

> 📄 See `Rapor.pdf` for the full detailed analysis report, methodology write-up, and final results table.

---

## 📁 Project Structure

```
MakineÖğrenmesiProje/
│
├── main.ipynb      # Main Jupyter Notebook (complete pipeline)
├── Rapor.pdf       # Project report (analysis, methodology, results)
└── README.md       # This file
```

---

## 📝 Key Insights

- **Business class** tickets are significantly more expensive than Economy class
- **Flight duration** is one of the strongest predictors of price
- **Days left** before departure heavily influences pricing (last-minute flights are much pricier)
- **Number of stops** affects price — direct flights vary widely by airline pricing strategy
- **Vistara** dominates the dataset (~42% of all records)

---


