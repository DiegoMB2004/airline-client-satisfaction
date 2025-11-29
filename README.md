# Airline Client Satisfaction

Analysis of the public airline passenger satisfaction dataset using three classic supervised learning algorithms inside `airline_satisfaction.ipynb`. The notebook walks through data preparation, feature engineering, hyper-parameter search, evaluation, and custom helper functions aimed at understanding which service touchpoints drive client sentiment.

## Contents
- `airline_satisfaction.ipynb`: end-to-end exploration done in Google Colab (logistic regression, KNN, Gaussian Naive Bayes, PCA visualizations, custom predictors).
- `Proyecto Final Aerolínea.csv` (not tracked): raw data used by the notebook. The file name matches the reference inside the notebook; place it in the project root or update the file path in the loading cells.

## Dataset
- ~129k records and 24 engineered predictors describing passenger demographics, class, trip characteristics, service quality scores (Wi-Fi, booking, food, cleanliness, boarding, etc.), and delay information.
- Binary target `satisfaction` (`0 = dissatisfied`, `1 = satisfied`).
- Source: mirrors the [Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction) dataset; download it manually (Kaggle terms forbid redistribution) and rename it to `Proyecto Final Aerolínea.csv`.

## Local setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```
Then either launch Jupyter locally (`jupyter notebook airline_satisfaction.ipynb`) or open the notebook directly in Google Colab via the badge embedded in the first cell.

## Notebook structure
1. **Part A – Logistic Regression**
   - Cleans and explores the dataset, splits train/test, applies one-hot encoding to categorical variables, scales numeric features, and concatenates the processed matrices.
   - Performs a manual grid search over `C`, `max_iter`, and penalty (`l1` or `l2`) using `liblinear`.
   - Visualizes coefficient magnitudes to highlight the most influential services and includes a helper sigmoid demo plus a `predict_customer()` convenience wrapper.
2. **Part B – K-Nearest Neighbors**
   - Scales all predictors (essential for distance-based models) and scans a range of `k` values using the error-rate curve to select the optimal neighbor count.
   - Trains the best model, reports metrics, plots ROC, and projects the data into 2D via PCA for exploratory decision-boundary and neighbor visualizations.
3. **Part C – Gaussian Naive Bayes**
   - Repeats the preprocessing flow (train/test split + scaling) and demonstrates how a generative baseline performs on the same feature set.
   - Adds a reusable `predict_satisfaction` function that accepts raw feature dictionaries for quick what-if testing.

## Key results
| Model | Notes | Accuracy | Recall | F1 / ROC |
| --- | --- | --- | --- | --- |
| Logistic Regression | `C=0.01`, `penalty='l1'`, 100 iterations | 0.876 | 0.837 | F1 = 0.854 |
| KNN | Optimal `k` chosen via elbow curve, scaled features | 0.929 | 0.880 (class 1) | ROC AUC = 0.980 |
| Gaussian Naive Bayes | Fast baseline with standardized inputs | 0.849 | 0.814 (class 1) | F1 = 0.823 |

Confusion matrices, precision/recall reports, ROC plots, and PCA decision regions are rendered inside the notebook for deeper inspection.

## Reproducing the analysis
1. Download the CSV from Kaggle and place it next to the notebook (or adjust the `pd.read_csv` path).
2. Install the Python dependencies and launch Jupyter/Colab as outlined above.
3. Run all cells sequentially. The notebook is organized so each part can also be executed independently after the initial data load.
4. Swap in your own customer scenarios inside the helper functions to probe how service-level changes affect the predicted satisfaction class.
