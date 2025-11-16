# 🧵 About the Project

**Project by Manel Comiche, Oriol Vila, Ana Jiménez, and Alicia Martí**,
developed for the **FME Datathon 2025** at the **Universitat Politècnica de Catalunya (UPC)**.

---

## 🎯 Inspiration

This project was born from the challenge of predicting product-level production for the **Mango Datathon**. With more than 90,000 training samples and 2,250 test items, the dataset combined rich categorical information (family, silhouette, season, fabric, color…) with continuous features and image embeddings.

We were inspired by a simple question:

> **Can we combine different modeling philosophies to build a forecasting system that is accurate, stable, and interpretable across product types?**

This idea evolved into two complementary components:

1. a **Stacking model (LightGBM + CatBoost)**, and
2. a **Super-Ensemble** merging v3, Stacking, and Ridge Regression.

---

# ⚙️ How We Built It

## 1. 🔦 Stacking Model: LightGBM + CatBoost

*(Document: “LGBM + CatBoost”)*

The stacking pipeline leverages the strengths of two boosting engines:

* **LightGBM** → excellent for numerical features
* **CatBoost** → state-of-the-art for categorical features
* **Meta-model LightGBM** → learns how to optimally merge both predictions

After internal validation, the model achieved:

* **MAE:** 5088.93
* **RMSE:** 8781.88
* **R²:** 0.9343

Mathematically, the stacking layer learns:

[
\hat{y} = f_{\text{meta}}\big(f_{\text{LGBM}}(X),\ f_{\text{CatBoost}}(X)\big)
]

We also built an interactive **product explorer** to inspect individual predictions, errors, and feature compositions.

---

## 2. 🧬 Ensemble Supermodel: v3 + Stacking + Ridge

*(Document: “Mango Ensemble Supermodel”)*

The Ensemble Supermodel was designed to achieve **low variance**, **smooth predictions**, and **better control** over production ranges.

It combines:

* **v3 model (0.55 weight):** strong baseline with a large feature set
* **Stacking model (0.35):** powerful but higher variance
* **Ridge model (0.10):** brings stability through PCA and regularization
* **Global boost factor:** (1.08\times) to calibrate scale

Final ensemble formula:

[
\hat{y}*{\text{final}} = 1.08 \cdot \big(0.55\hat{y}*{v3} + 0.35\hat{y}*{stack} + 0.10\hat{y}*{ridge}\big)
]

This approach helped:

* Prevent extreme outliers
* Produce smoother predictions
* Maintain accuracy while increasing robustness
* Provide **explainability**, showing each model’s contribution per product
* Generate a clean and ready-to-submit CSV

Visual validation included distribution histograms, scatter plots (v3 vs ensemble), and per-product model breakdown dashboards.

---

# 📚 What We Learned

* How **stacking** can leverage heterogeneous feature types better than any standalone model.
* The importance of **variance control** when constructing forecasting models.
* Why thoughtful **ensemble design** often surpasses heavy single-model tuning.
* How **per-product explainability** helps detect anomalies and justify forecasts.
* The value of iterative validation and error-distribution analysis.

---

# ⚠️ Challenges We Faced

* Handling **high-cardinality categorical variables** and image embeddings.
* Preventing overfitting in strong models like CatBoost and v3.
* Choosing the right **ensemble weights** and **boost factor** without distorting overall scale.
* Designing a **fair validation scheme** for stacking and baseline models.
* Managing computation times while experimenting with multiple architectures.

---

# 🚀 Final Result

The final system is a **robust hybrid forecasting framework** that blends:

* the predictive power of gradient boosting,
* the stability of linear models,
* and the reliability of ensemble learning.

It delivers accurate, smooth, and interpretable production predictions—ready for real-world retail workflows and Kaggle competition submission.
