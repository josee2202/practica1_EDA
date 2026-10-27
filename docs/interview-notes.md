# Interview notes

Short prep notes for discussing this project. "Original work" means the November 2024 - January 2025 assignment; "recovery" means the October 2026 work that made it reproducible.

## 60-90 second pitch

"I took a public dataset of about 300,000 consumer-loan applications, where only 8% of applicants later had payment difficulties, and built the whole tabular-ML workflow in five notebooks. First an EDA that decided which of the 122 columns to keep, based on how much was missing and whether missingness itself told us something about default. Then imputation fitted only on the training split, encoding (one-hot, label and target encoding), scaling and a comparison of feature-selection methods, which kept 26 features. Because of the imbalance I rebalanced the training data with SMOTE plus undersampling, compared seven classifiers and chose a regularised logistic regression for its recall/precision balance and interpretability. On a held-out test set it reaches ROC AUC 0.732 and PR AUC 0.205, against 0.5 and 0.081 for a no-skill baseline; with a threshold tuned on validation it catches 65% of defaults at 16% precision. SHAP shows which variables drive the score. Later I came back to it and made it reproducible: pinned environment, a runner that executes the notebooks from scratch, a synthetic-data test in CI, and I fixed two evaluation leaks where the test set had been used for decisions."

## Three technical decisions and their alternatives

1. **Rebalancing with SMOTE (to 0.4) followed by random undersampling (to 0.6), applied only to the training part.**
   Alternatives: class weights, only oversampling, only undersampling. On the raw 8% data most vanilla models almost never predicted a default (the logistic regression caught 0.9% of validation defaults; after rebalancing, 44%). The combination kept most of the real data while giving the minority class enough weight. Trade-off: cross-validation scores computed on resampled data are optimistic, which is why the final numbers come from an untouched test set.
2. **Logistic regression as the final model.**
   Alternatives tried in notebook 04: Naive Bayes, decision tree, XGBoost and LightGBM (tuned), and random forest (default settings only). The logistic regression gave the best recall/precision balance on validation in the original work, and its coefficients and SHAP values are easy to explain, which matters in credit decisions. In the current run it also had the best F1 on validation at threshold 0.5 (0.27, against 0.24 for LightGBM, 0.22 for the decision tree and 0.19 for XGBoost). Caveat: the models were compared at a single threshold, not on ROC or PR AUC, so a boosting model with a tuned threshold might do better.
3. **Choosing the decision threshold on validation, optimising F2.**
   Alternatives: keep 0.5, optimise G-mean on the ROC curve, or pick the threshold on the test set (what the original notebook did). F2 weights recall twice as much as precision, matching the cost asymmetry. Choosing it on validation (recovery fix) keeps the test estimate honest.

## Main difficulties and how they were solved

- **Imbalance** (original): accuracy was misleading, so the evaluation moved to recall, F2 and PR AUC, plus rebalancing.
- **Cost of wrapper methods** (original): on 246,008 rows x 148 features, RFE took about an hour and Boruta about 36 minutes, with no clear gain over the regularised-logistic selector (best F2 for the L2 set; Boruta's 57 features gave a PR AUC of 0.227 against 0.220 with the logistic regression).
- **KNN imputation at scale** (original): fitting the imputer on the full training set was too slow, so it was fitted on a 10% sample of train and applied to train and test.
- **Notebooks that no longer ran** (recovery): moved data paths, an absolute path from another machine, a module alias overwritten by a file handle, and helper functions that used `scipy` without importing it. Found by running every notebook in a fresh kernel and by a synthetic end-to-end run.
- **Hidden state between notebooks** (recovery): a cell in notebook 03 only worked because cells had been run out of order; in a linear run it turned an encoded column back into text. Fixed so the notebooks run top to bottom.
- **Evaluation leaks** (recovery): the feature-selector comparison and the tuned threshold had used the test set. Both moved to validation splits; the test set is now touched once.
- **A test that passed for the wrong reason** (recovery): the synthetic end-to-end run passed, but the real run failed in notebook 04, because a hand-written list of columns assumed the original feature selection, and the synthetic target happened to select those columns. Lesson: synthetic data proves the code runs, not that hidden assumptions about the real data still hold. The real run on the full data was necessary.

## Verifiable results and limitations

- Test set (61,503 rows, default rate 8.1%): ROC AUC 0.732, PR AUC 0.205. Threshold 0.5: recall 0.447, precision 0.201. Validation-chosen threshold 0.38: recall 0.655, precision 0.157. Source: `reports/metrics.json`, produced by `scripts/run_pipeline.py`.
- The synthetic-data CI run proves the code executes end to end; it says nothing about model quality.
- Limitations: one train/test split (no confidence intervals); target encoding fitted in-sample on train; EDA decisions (column drops) taken on the full dataset before splitting; probabilities not calibrated (the model was trained on resampled data); the `saga` solver stops at `max_iter` without converging; `DAYS_EMPLOYED` keeps a placeholder value for pensioners, so its weight is not a tenure effect; no inference pipeline for raw new applications; no fairness analysis although gender and family variables are in the data.

## What I would change for real use at scale

- Fix the known data and fitting issues first: turn the `DAYS_EMPLOYED` placeholder into a flag plus a missing value, and let the logistic regression converge (more iterations or `liblinear`).
- One scikit-learn / imblearn `Pipeline` from raw columns to prediction, so the same fitted transformations are applied at inference time, saved as a single artifact.
- Cross-validated target encoding and resampling inside each CV fold, with repeated or temporal validation and confidence intervals.
- Probability calibration on untouched data and a threshold set from actual costs (loss given default, margin on good loans).
- Gradient boosting with native missing-value handling as a challenger, plus features from the other tables of the original data (bureau, previous applications).
- Fairness and stability checks (performance by group, drift monitoring) and a model card.

## Five likely questions

1. **Why not accuracy?** With 8% positives, predicting "no default" for everyone gives about 92% accuracy and zero recall. The project reports recall, F2, PR AUC and ROC AUC, and compares with a no-skill baseline (notebook 05, `reports/metrics.json`).
2. **How did you avoid leakage?** Imputers, encoders, the scaler and the feature selector are fitted on the training split only (notebooks 02-03). Resampling is applied only to the train part of notebook 04, never to validation or test. The threshold is chosen on validation. Remaining weak points: target encoding is not cross-fitted, and the column-drop rule in notebook 01 looked at the full dataset.
3. **Why does SMOTE come before undersampling?** SMOTE first raises the minority class to 40% of the majority by interpolating between neighbouring defaults; undersampling then removes majority rows until the minority is 60% of the majority. Doing only undersampling would discard a large share of real data; only SMOTE would create many more synthetic rows.
4. **How do you read the SHAP values of a logistic regression?** For a linear model, SHAP equals coefficient times the feature's deviation from the background mean, in log-odds. `shap.LinearExplainer` is used with the test set as background; the bar chart ranks features by mean absolute contribution. Correlated features (for example `AMT_CREDIT` and `AMT_GOODS_PRICE`) can get large opposite contributions, so importances should be read per group.
5. **Why is the precision so low at the tuned threshold?** Because the threshold is set to favour recall (F2). At a 65% recall the model flags many good payers, so precision drops to 16%. The right operating point depends on the cost of a missed default versus a rejected good customer; the PR curve in `reports/figures/pr_curve.png` shows the whole trade-off.
