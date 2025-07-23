# ⚖️ Obesity Level Classification: Comparative Machine Learning Benchmark

This repository presents a robust and transparent benchmarking of classical and modern machine learning algorithms for **predicting obesity levels** using real-world biometric and lifestyle data. Developed for recruiters, data scientists, and technical leads, it aims to showcase advanced pipeline engineering, hands-on feature craftsmanship, and honest discussion of challenges in health-oriented ML.

---

## 🚀 Project Highlights

- **Purpose:** Thoroughly compare supervised models for multi-class classification of obesity levels based on structured health, behavior, and demographic attributes.
- **Key Strengths:**
    - Automated, reproducible pipelines with advanced feature engineering and class balancing.
    - Clear code modularity and extreme transparency in reporting, metrics, and preprocessing decisions.
    - Insightful visualizations and in-depth model interpretation.
    - Honest commentary on limitations and next steps.

---

## 📂 Repository Structure

| File/Notebook             | Purpose                                                                                         |
|--------------------------|-------------------------------------------------------------------------------------------------|
| `preprocessing.py`        | Loads and cleans raw data, encodes categorical variables, performs advanced class balancing, and engineers new features (e.g., BMI). Outputs the final training dataset. |
| `random-forest.py`        | Implements a full Random Forest workflow: modeling, hyperparameter optimization, feature importance, and visualizes confusion matrix.      |
| `gradientBoost.py`        | Builds and tunes a Gradient Boosting Classifier, executes comprehensive grid search, and reports best scores.                             |
| `supportVectorMachineDefault.py` | Constructs a pipeline using SVMs with full cross-validation, preprocessing (scaling, encoding), and test set evaluation.                       |
| `after_preprocessing.csv` | Processed dataset ready for model training and evaluation.                                       |

Each script is stand-alone and well-commented for clear understanding and rapid extension.

---

## 📦 Data Source

- **Dataset:** [Estimation of obesity levels based on eating habits and physical condition](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition) (UCI Machine Learning Repository)
- **Features:** Demographics, anthropometrics, and behavioral variables (age, gender, height, weight, eating habits, activity level, etc.)
- **Target:** Multi-class obesity status (7 categories)

---

## 🔬 Pipeline Overview

1. **Data Preparation (`preprocessing.py`):**
    - Categorical variable encoding using `OrdinalEncoder`.
    - Aggressive class balancing: Up-sampling with SMOTE for minority gender/class segments, down-sampling of majority classes.
    - Outlier and duplicate handling.
    - Engineering of the Body Mass Index (BMI) and correction of derived features.
    - Outputs: Fully processed and balanced CSV for all downstream tasks.

2. **Random Forest Analysis (`random-forest.py`):**
    - Comprehensive hyperparameter tuning via grid search (tree depth, estimator count, node requirements).
    - Feature importance visualization to highlight key predictors.
    - Generation of confusion matrix heatmaps for interpretability.
    - Test and validation reporting with best estimator selection.

3. **Gradient Boosting Modeling (`gradientBoost.py`):**
    - Grid search across critical parameters (learning rate, estimators, tree depth).
    - Training and accuracy reporting with classification summary.
    - Outputs optimal hyperparameters and detailed validation report.

4. **Support Vector Machine Benchmark (`supportVectorMachineDefault.py`):**
    - Pipeline-based scaling and encoding.
    - Nested cross-validation (train/validation/test splits with folds).
    - Evaluation: confusion matrices, accuracy, and mean validation/test metrics.
    - Fully reproducible results evaluating SVM’s generalization on multiclass problems.

---

## 📊 Results Snapshot

| Model           | Typical Accuracy | Notable Insights                                     |
|-----------------|------------------|------------------------------------------------------|
| Random Forest   | ~66%             | Best at capturing feature interactions, robust to noise. Feature importances provide actionable insight. |
| Gradient Boost  | ~67%             | Top single-model performer with careful tuning, warns of risk of overfitting.            |
| SVM             | ~60%             | Requires strong preprocessing; high-dimensional categorical data limits effectiveness.    |
| (Others: see scripts for details) |                  |                                                      |

> Refer to in-notebook visualizations (feature importance plots, confusion matrices) for full interpretability and model diagnostics.

---

## 💡 Why This Repo Stands Out

- **Recruiter-Ready:** Modern, readable, and modular code with extensive commentary.
- **All Steps Automated:** No manual data downloads; scripts execute end-to-end pipelines for each modeling paradigm.
- **Critical Reflection:** Discusses model suitability, practical trade-offs, and future improvement areas in health data.
- **Visuals for Insight:** Delivers essential exploratory and diagnostic plots for hiring managers or ML practitioners.

---

## 🛠️ Technologies Used

- **Python 3.12+**
- **pandas**, **numpy**
- **scikit-learn**
- **imblearn** (for class balancing)
- **matplotlib**, **seaborn** (visualizations)

---

## 📌 How to Run

1. Install dependencies:
    ```
    pip install pandas numpy scikit-learn imblearn matplotlib seaborn
    ```
2. Run data preprocessing to generate a clean CSV:
    ```
    python preprocessing.py
    ```
3. Run any modeling script (e.g., Random Forest or Gradient Boosting) as:
    ```
    python random-forest.py
    python gradientBoost.py
    python supportVectorMachineDefault.py
    ```
4. Check generated outputs for reports, confusion matrices, and feature visualizations.

---

## 📈 Extending the Project

- Add new models (e.g., XGBoost, CatBoost, deep neural nets) with minimal adaptation—plug into the preprocessed CSV.
- Expand feature engineering or add explainability modules (SHAP, LIME).
- Port workflow to notebooks for interactive EDA.

---

## 👨‍💻 Author

**Thiago Aragão and colleagues**  
Data Scientists | Machine Learning | Data Analysis

- GitHub: [@DrAragorn](https://github.com/DrAragorn)
- Email: thiago.alpha.06@gmail.com
- LinkedIn: [linkedin.com/in/thiago-r-aragao](https://linkedin.com/in/thiago-r-aragao)

---

*This repository embodies transparent, production-style machine learning for complex classification. Designed for real hiring conversations, technical due diligence, and inspired collaboration.*
