# DIP Diet Advisor — ML Web Application

A personalised dietary recommendation web app built with **Python**, **scikit-learn**, and **Streamlit** that classifies users into one of three **DIP Diet profiles** and delivers actionable meal plans, a 7-day schedule, and lifestyle tips.

> **DIP Diet** (Disciplined & Intelligent Plan) is a whole-food, plant-forward dietary philosophy that emphasises fruits, vegetables, legumes, millets, and proper hydration while eliminating processed foods and refined sugar.

---

## 📁 Project Structure

```
project/
├── app.py                    # Streamlit web application
├── train_model.py            # Model training script
├── requirements.txt          # Python dependencies
├── README.md
├── agent_instructions.md
├── .env.example
├── data/
│   └── dip_diet.csv          # DIP Diet dataset (200 rows, 3 profiles)
└── models/
    └── model.pkl             # Saved best-performing pipeline
```

---

## 🥗 Diet Profiles

| Profile | Emoji | Description |
|---|---|---|
| **Green Warrior** | 🟢 | Excellent habits — high natural food intake, active, low stress |
| **Balanced Builder** | 🔵 | Moderate habits — mix of natural and processed, room to improve |
| **Detox Starter** | 🟠 | High processed food / low activity — needs a structured reset |

---

## 🚀 Local Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Create and activate a virtual environment *(recommended)*
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model *(must run once before launching the app)*
```bash
python train_model.py
```
This will:
- Load `data/dip_diet.csv`
- Preprocess numeric + categorical features via `ColumnTransformer`
- Compare 6 ML classifiers via 5-fold cross-validation
- Save the best pipeline to `models/model.pkl`

### 5. Launch the Streamlit app
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ Deploy to Streamlit Community Cloud

1. Push the entire project (including `data/` and `models/`) to a **public GitHub repository**.
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**.
3. Select your repository, branch (`main`), and set the main file to `app.py`.
4. Click **Deploy**.

> **Note:** The `models/model.pkl` file must be committed to GitHub so the app can load it on startup.

---

## 🧠 ML Details

| Item | Value |
|---|---|
| Dataset | `data/dip_diet.csv` (~200 samples) |
| Problem type | Multi-class Classification (3 classes) |
| Numeric features | age, weight_kg, height_cm, bmi, activity_level, sleep_hours, water_intake_liters, fruits_per_day, vegetables_per_day, processed_food_score, sugar_intake_score, stress_level |
| Categorical features | health_goal (weight_loss / muscle_gain / maintenance) |
| Target | diet_profile (Green_Warrior / Balanced_Builder / Detox_Starter) |
| Preprocessing | StandardScaler (numeric) + OneHotEncoder (categorical) via ColumnTransformer |
| Models compared | Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting, SVM |
| Selection criterion | 5-fold cross-validation accuracy |

---

## 🔑 Environment Variables

No API keys are required for this project.  
See `.env.example` for the template if you extend the app with external services (e.g., LLM-powered recommendations).

---

## 📦 Dependencies

See `requirements.txt`:
- `streamlit` — web interface
- `pandas` — data loading & manipulation
- `numpy` — numerical operations
- `scikit-learn` — ML models, pipelines, and preprocessing
- `openpyxl` — Excel file support
