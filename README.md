
## 🛒 Grocery Sales Forecasting

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Favorita-blue)](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting)
[![Conference Paper](https://img.shields.io/badge/CIKE--2022-Paper-green)](https://www.atlantis-press.com/proceedings/cike-22/125972906)


## 📚 Project Documents

- 📄 Conference Paper – CIKE 2022
- 📝 Technical Details Report
  

## 📌 Project Overview

This project tackles the real-world challenge of **forecasting daily unit sales** for over 200,000 unique products across hundreds of Ecuadorian grocery stores owned by **Corporación Favorita**. It was originally built for a Kaggle competition and later published in the 2022 International Conference on Creative Industry and Knowledge Economy (CIKE 2022).

Grocery sales forecasting is critical to:
- Reduce spoilage from overstocking.
- Prevent lost revenue from stock-outs.
- Optimize logistics and improve supply chain management.

Our solution integrates **LightGBM**, time-series feature engineering, and economic indicators to build a scalable machine learning forecasting model.

---

## 📂 Dataset

The dataset comes from [Kaggle](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting), provided by Corporación Favorita. It includes:

- `train.csv`: Historical unit sales with promotion info.
- `test.csv`: Features to predict unit sales.
- `items.csv`: Metadata for each item (family, class, perishable flag).
- `stores.csv`: Store-level metadata (city, cluster).
- `transactions.csv`: Daily customer transactions.
- `oil.csv`: Daily oil prices — reflecting macroeconomic trends.
- `holidays_events.csv`: National/local holidays with labels.

We focused on **2017 data** to minimize noise from outdated trends and seasonal shifts.

---

## 🧠 Methodology

### 🛠 Feature Engineering

- Log-transform of sales: `log1p(unit_sales)`
- Time-based rolling averages: 1, 3, 7, 14, 30, 60, 140-day windows
- Weekday seasonality: `mean_X_dowY` features
- Promotion history and flags
- Weighted training for **perishable items** (1.25× more)

### 🔁 Modeling Strategy

- Model: **LightGBM** (leaf-wise tree growth, histogram optimization)
- Training objective: `regression`, metric: `l2` (MSE)
- 16 separate models trained, one per forecast day
- Early stopping with validation sets
- Final evaluation: **Validation MSE = 0.3507**

### 🧪 Hyperparameters
```python
params = {
    'num_leaves': 31,
    'objective': 'regression',
    'min_data_in_leaf': 300,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'metric': 'l2',
    'num_threads': 4
}
```

---

## 🧾 Results & Impact

- 📊 **Model Accuracy**: Validation MSE = **0.3507**
- 💡 Approximates RMSLE using log-transformed targets.
- 🚚 Reduces overstocking and shortages.
- 📦 Enables smarter purchasing and inventory decisions.
- 🌍 Scalable to any retail supply chain system.

---

## 🌍 Societal Value

📌 After the 2016 Ecuador earthquake, sales patterns changed drastically. This model helps forecast **demand during crises**, aiding both supply chain stability and humanitarian efforts.

🛍️ In general, this model:
- Reduces waste
- Improves customer experience
- Provides a **data-driven forecasting framework** for developing economies

---

## 📰 Publication

This project was published in the **CIKE 2022 Conference Proceedings**:

**🔗 [Read the full paper](https://www.atlantis-press.com/proceedings/cike-22/125972906)**  
*“Grocery Sales Forecasting” by Yixu Liu*  
Atlantis Press | Advances in Economics, Business and Management Research, Volume 214

---


## ✍️ Author

**Yixu Liu**  
Email: yixuliu558@gmail.com  

---

## 📜 License

This work is published under the **CC BY-NC 4.0 License**  
(Non-commercial use permitted with attribution)

---

## 📌 References

- [Kaggle Competition Page](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting)
- [CIKE 2022 Published Paper](https://www.atlantis-press.com/proceedings/cike-22/125972906)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
