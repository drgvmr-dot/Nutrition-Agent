# Agent Instructions

## Project Overview
This is the **DIP Diet Advisor** — a Machine Learning Web Application that classifies
users into one of three personalised DIP Diet (Disciplined & Intelligent Plan) profiles
based on their body metrics, lifestyle habits, and dietary patterns.
The app is built with Python, scikit-learn, and Streamlit.

## Diet Profiles
| Profile | Key Characteristics |
|---|---|
| `Green_Warrior` | High fruit/vegetable intake, active lifestyle, low processed food, low stress |
| `Balanced_Builder` | Moderate diet, mix of natural and processed foods, moderate activity |
| `Detox_Starter` | High processed food/sugar, low activity, high BMI — needs a reset |

## Dataset
- File: `data/dip_diet.csv`
- Rows: ~200
- **Numeric features**: age, weight_kg, height_cm, bmi, activity_level, sleep_hours,
  water_intake_liters, fruits_per_day, vegetables_per_day, processed_food_score,
  sugar_intake_score, stress_level
- **Categorical features**: health_goal (weight_loss, muscle_gain, maintenance)
- **Target**: diet_profile (Green_Warrior, Balanced_Builder, Detox_Starter)
- Problem type: Multi-class Classification

## How to Extend

### Adding a new diet profile
1. Add training rows with the new `diet_profile` label to `data/dip_diet.csv`.
2. Re-run `python train_model.py`.
3. In `app.py`, add an entry to the `PROFILES` dict with `emoji`, `title`,
   `css_class`, `tagline`, `description`, `key_principles`, `meal_plan`,
   `weekly_schedule`, and `lifestyle_tips`.

### Adding a new feature
1. Add the column to `data/dip_diet.csv`.
2. In `train_model.py`, add it to `NUMERIC_COLS` or `CATEGORICAL_COLS`.
3. In `app.py`, add the corresponding input widget inside `st.form("diet_form")`
   and include it in the `user_data` dict.
4. Re-run `python train_model.py` and restart the app.

### Adding a new model
In `train_model.py`, add your classifier to the `candidates` dict:
```python
"My Model": MyClassifier(...)
```
The training script will automatically include it in the comparison and pick the best one.

### Switching to an Excel dataset
If the dataset is `.xlsx`, change the load line in `train_model.py`:
```python
df = pd.read_excel(DATA_PATH, engine="openpyxl")
```

## Environment Variables
No external API keys are needed for the base project.
If you add an LLM for richer recommendations, define keys in `.env` (never commit it)
and load them with `python-dotenv` or Streamlit secrets.

## Deployment Notes
- Commit `models/model.pkl` to Git so Streamlit Cloud can load the model.
- All paths are relative — no hard-coded absolute paths anywhere.
- Python 3.9+ is required.
