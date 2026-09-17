# Land Acquisition Delay Predictor

A Streamlit frontend for the ML project in `original_notebook.ipynb`. The
notebook is the untouched backend/model source of truth; this app adds a UI
on top of it. **The model is already trained** — `model/` contains real
artifacts trained on your actual dataset, ready to use.

## Project structure

```
land_acquisition_project/
├── app.py                     # Streamlit frontend
├── train_and_export.py        # Reproduces the notebook's pipeline (+ one fix, see below), exports artifacts
├── ml_backend.py              # Uses real scikit-learn if installed; otherwise a NumPy fallback (see below)
├── original_notebook.ipynb    # Your original ML notebook (unchanged)
├── data/
│   └── land_acquisition_delay_synthetic.csv   # Your real dataset (already included)
├── model/                     # Already generated — real trained artifacts
│   ├── final_model.pkl
│   ├── scaler.pkl
│   ├── feature_columns.json
│   ├── numeric_features.json
│   ├── categorical_options.json
│   ├── numeric_stats.json
│   └── metrics.json
├── requirements.txt
└── README.md
```

## One deliberate fix over the notebook's own logic

While validating the deployed prediction path end-to-end, a real bug in the
notebook surfaced: cell 15 scales `numeric_columns` on the **full dataframe**
before the train/test split; cell 36 then fits a **second** scaler
(`new_scaler`) on `X_train`'s numeric columns — but those were already scaled
by cell 15, so `new_scaler` ends up fit on already-standardized data
(mean≈0, scale≈1), making it nearly a no-op. Since `predict_with_user_input`
applies only `new_scaler` to **raw** user input, raw values passed through
almost unchanged and dominated the logistic regression's logit — the deployed
model predicted "Delayed" at ~100% confidence for nearly any input,
regardless of actual risk profile. This was confirmed by testing a clearly
low-risk and a clearly high-risk profile — both came back "Delayed, ~100%".

**The fix** (applied in `train_and_export.py`, after explicit approval): the
redundant first full-dataframe scaling pass was removed. `new_scaler` is now
the *only* scaling step, fit on `X_train`'s raw numeric columns, and every
model (`delay_model`, decision tree, random forest, `final_model`) is trained
on data that went through that same single scaling pass — matching what
`predict_with_user_input` assumed was happening. Nothing else changed: same
algorithms, same hyperparameters, same features, same train/test split, same
`random_state=42`. After the fix, a low-risk test profile predicts "Not
Delayed" at 86% confidence and a high-risk profile predicts "Delayed" at 99%
confidence — the model now actually discriminates.

## Why `ml_backend.py` exists

The machine this project was built on blocks scikit-learn/scipy's compiled
extensions via an Application Control (WDAC) policy. `ml_backend.py` tries to
import real scikit-learn first; if that fails, it falls back to pure-NumPy
reimplementations (Newton-Raphson logistic regression matching scikit-learn's
default L2/C=1.0 objective, and CART decision tree / random forest) with the
identical `fit`/`predict`/`predict_proba` interface. **The artifacts in
`model/` right now were trained with the NumPy fallback**, since scikit-learn
isn't available in this environment. `metrics.json`'s `_ml_backend` field
records which backend produced them.

If you install `scikit-learn` on a machine where it's not blocked and re-run
`train_and_export.py`, it will automatically use real scikit-learn instead —
no code changes needed — and should produce very similar results (both solve
the exact same logistic regression objective to convergence).

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

If `scikit-learn` fails to install/import on your machine for any reason,
everything still works via the NumPy fallback in `ml_backend.py` — no action
needed.

## 2. Run the Streamlit app

The model is already trained, so you can run the app immediately:

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

## 3. Re-training (only needed if the dataset changes)

```bash
python train_and_export.py --csv data/land_acquisition_delay_synthetic.csv
```

This overwrites the `model/` folder. You should see output like:

```
ML backend used: sklearn   (or: numpy_fallback)
Artifacts written to .../model
Deployed model (final_model) test accuracy: 0.6xx
```

## Testing the app

1. **Home** — confirm the project description loads.
2. **Prediction** — try a "safe" profile (few owners/parcels, no legal
   cases/disputes, few missing documents, generous planned duration) and a
   "risky" profile (many owners/parcels, legal cases, disputes, missing
   documents, short planned duration) and confirm the prediction and
   confidence actually differ between them (verified working — see above).
3. **Model Performance** — confirm the accuracy/confusion matrix/classification
   report values match what `train_and_export.py` printed to your terminal.
4. **Error handling** — try deleting the `model/` folder and reloading the
   app; you should see a friendly message telling you to run
   `train_and_export.py`, not a raw traceback.

## Deployment

- **Streamlit Community Cloud**: push this folder to a GitHub repo (including
  the `model/` folder — it's small — but NOT the raw CSV if it's sensitive),
  then deploy at [https://share.streamlit.io](https://land-acquisition-delay-predicto.streamlit.app/) pointing at `app.py`.
- **Any server/VM**: `pip install -r requirements.txt` then
  `streamlit run app.py --server.port 8501 --server.address 0.0.0.0`.
- Regenerate `model/` with `train_and_export.py` any time the dataset changes
  — the app itself never retrains.

## What was NOT changed

- The dataset, feature engineering, one-hot encoding, model algorithm
  (Logistic Regression), hyperparameters, and train/test split are identical
  to `original_notebook.ipynb`. The only change from the notebook's own logic
  is the double-scaling fix described above, which was diagnosed, explained,
  and explicitly approved before being applied — it does not change the
  algorithm, only removes a redundant/inconsistent extra scaling pass so
  training and inference use the same preprocessing.
