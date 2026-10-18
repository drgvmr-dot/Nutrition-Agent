"""
app.py
------
DIP Diet Advisor — Streamlit Web Application.

Classifies users into one of three DIP Diet profiles using a trained ML
pipeline, then delivers a personalised meal plan, lifestyle tips, and a
7-day diet schedule based on the Disciplined & Intelligent Plan (DIP Diet)
philosophy: whole natural foods, no processed items, proper hydration, and
daily discipline.
"""

import pathlib
import pickle

import numpy as np
import pandas as pd
import streamlit as st

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = pathlib.Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DIP Diet Advisor",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .profile-card {
        background: #f0fdf4;
        border-left: 5px solid #16a34a;
        border-radius: 8px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
    }
    .profile-card.balanced { background:#eff6ff; border-color:#2563eb; }
    .profile-card.detox    { background:#fff7ed; border-color:#ea580c; }
    .tip-box {
        background:#fafafa;
        border:1px solid #e5e7eb;
        border-radius:6px;
        padding:0.8rem 1rem;
        margin-bottom:0.5rem;
    }
    .metric-label { font-size:0.78rem; color:#6b7280; }
    .section-header {
        font-size:1.1rem;
        font-weight:700;
        margin-top:1.4rem;
        margin-bottom:0.4rem;
        color:#1f2937;
    }
</style>
""", unsafe_allow_html=True)

# ── Recommendation knowledge-base ─────────────────────────────────────────────
PROFILES = {
    "Green_Warrior": {
        "emoji": "🟢",
        "title": "Green Warrior",
        "css_class": "",
        "tagline": "You are already thriving on nature's best — keep it up!",
        "description": (
            "You have excellent dietary habits aligned with the DIP Diet philosophy. "
            "Your high fruit and vegetable intake, low processed-food score, and active "
            "lifestyle place you in the top tier of health-conscious eaters. "
            "Focus on variety, seasonal eating, and micro-nutrient optimisation."
        ),
        "key_principles": [
            "Eat at least 5 colours of fruits/vegetables daily",
            "Hydrate with 3+ litres of water or coconut water",
            "Include raw salads in every main meal",
            "Prefer whole grains: millets, brown rice, quinoa",
            "Avoid all refined sugar and white flour products",
        ],
        "meal_plan": {
            "Early Morning (6–7 AM)": "2 glasses warm water + 5 soaked almonds + 2 walnuts",
            "Breakfast (8–9 AM)":     "Fruit bowl (banana, papaya, apple) + a glass of vegetable juice",
            "Mid-Morning (11 AM)":    "Coconut water or 1 seasonal fruit",
            "Lunch (1–2 PM)":         "Large raw salad + 2 multigrain rotis + sabzi (cooked greens) + dal",
            "Evening Snack (4–5 PM)": "Handful of seeds (sunflower, pumpkin) + herbal tea",
            "Dinner (7–8 PM)":        "Vegetable soup + brown rice / millets + stir-fried vegetables",
            "Post-Dinner (9 PM)":     "Warm turmeric milk (dairy-free option: almond milk)",
        },
        "weekly_schedule": [
            ("Monday",    "Detox day — only fruits, raw veggies, and water"),
            ("Tuesday",   "Protein boost — sprouts, legumes, paneer/tofu"),
            ("Wednesday", "Grain day — millets, oats, quinoa"),
            ("Thursday",  "Green day — leafy greens in every meal"),
            ("Friday",    "Seed & nut day — add flaxseeds, chia, almonds"),
            ("Saturday",  "Rainbow day — eat all 7 colours of produce"),
            ("Sunday",    "Mindful eating — slow, grateful, no screens at meals"),
        ],
        "lifestyle_tips": [
            "🌅 Wake up by 6 AM and start with sunlight exposure for 15 min",
            "🚶 Walk 10,000 steps or do 45 min yoga / brisk walk",
            "🧘 Practise 10-min mindful breathing before meals",
            "📵 No screens 1 hour before bedtime — aim for 7–8 hrs sleep",
            "📓 Maintain a food journal to track mood and energy levels",
        ],
    },
    "Balanced_Builder": {
        "emoji": "🔵",
        "title": "Balanced Builder",
        "css_class": "balanced",
        "tagline": "Solid foundation — a few targeted upgrades will unlock your potential.",
        "description": (
            "You have a moderate diet with a mix of natural and processed foods. "
            "The DIP Diet approach for you focuses on gradually crowding out unhealthy "
            "choices by adding more whole-food options and improving consistency. "
            "Small, sustainable changes compounded daily yield powerful results."
        ),
        "key_principles": [
            "Replace one processed meal per week with whole foods",
            "Add one extra serving of fresh fruit every day",
            "Drink 8 glasses (2 L) of water before 6 PM",
            "Switch to unrefined cooking oils: cold-pressed coconut or mustard oil",
            "Reduce sugar in beverages — swap tea/coffee sugar for jaggery or none",
        ],
        "meal_plan": {
            "Early Morning (6–7 AM)": "1 glass warm lemon water + 4 soaked almonds",
            "Breakfast (8–9 AM)":     "Oats porridge with fruits OR 2 idlis with sambar and chutney",
            "Mid-Morning (11 AM)":    "1 seasonal fruit (avoid mango / banana if weight loss is goal)",
            "Lunch (1–2 PM)":         "2 rotis + dal + one cooked vegetable + a small salad",
            "Evening Snack (4–5 PM)": "Roasted chana / makhana + green tea",
            "Dinner (7–8 PM)":        "Khichdi with vegetables OR rice + rajma/chole + salad",
            "Post-Dinner (9 PM)":     "Warm water with a pinch of cinnamon",
        },
        "weekly_schedule": [
            ("Monday",    "Start with a fruit-only breakfast"),
            ("Tuesday",   "Swap white rice for brown rice or millets at lunch"),
            ("Wednesday", "Cook without oil for one meal — steam or boil"),
            ("Thursday",  "Add a large raw salad before lunch and dinner"),
            ("Friday",    "Sugar-free day — no sweets, no packaged drinks"),
            ("Saturday",  "Meal prep day — cook legumes and grains in bulk"),
            ("Sunday",    "Light meals only — soup, salad, fruits"),
        ],
        "lifestyle_tips": [
            "🌿 Add a 20-min walk after lunch and dinner",
            "🥤 Replace sodas and juices with infused water or nimbu pani",
            "🛒 Shop the perimeter of the grocery store — whole foods first",
            "📱 Use a nutrition app to log meals for 2 weeks — awareness is key",
            "😴 Prioritise 7 hrs sleep — poor sleep increases junk-food cravings",
        ],
    },
    "Detox_Starter": {
        "emoji": "🟠",
        "title": "Detox Starter",
        "css_class": "detox",
        "tagline": "Your reset journey begins now — every small step counts!",
        "description": (
            "Your current dietary patterns include high processed-food and sugar intake "
            "with relatively low natural food consumption. The DIP Diet reset protocol "
            "is designed for exactly your situation: a gentle, structured detox that "
            "removes toxins, resets metabolism, and builds sustainable habits — "
            "without crash diets or deprivation."
        ),
        "key_principles": [
            "Eliminate all packaged/ultra-processed foods for 21 days",
            "Start each morning with 2 glasses of warm water",
            "Introduce one fruit daily — start with banana or papaya",
            "Replace one meal per day with a vegetable soup or salad",
            "No fried food, no refined sugar, no white bread for 21 days",
        ],
        "meal_plan": {
            "Early Morning (6–7 AM)": "2 glasses warm water + 1 tsp honey + pinch of turmeric",
            "Breakfast (8–9 AM)":     "1 banana + 1 glass of vegetable juice (carrot, beetroot, ginger)",
            "Mid-Morning (11 AM)":    "1 fruit (apple / pear / guava)",
            "Lunch (1–2 PM)":         "1 roti + moong dal + cucumber-tomato salad + buttermilk",
            "Evening Snack (4–5 PM)": "Handful of roasted seeds + herbal tea (no sugar)",
            "Dinner (7–8 PM)":        "Clear vegetable soup + 1 small bowl of khichdi",
            "Post-Dinner (9 PM)":     "Warm water with cumin (jeera) — aids digestion",
        },
        "weekly_schedule": [
            ("Monday",    "Detox day 1 — only fruits until noon, light meals after"),
            ("Tuesday",   "Replace breakfast with fruits + 1 glass vegetable juice"),
            ("Wednesday", "Add a 15-min post-meal walk — start the movement habit"),
            ("Thursday",  "No sugar day — remove all added sugar from all meals"),
            ("Friday",    "Whole food lunch — dal, roti, salad, no fried sides"),
            ("Saturday",  "Kitchen audit — remove packaged foods from home"),
            ("Sunday",    "Reflect, plan next week's meals, prepare a shopping list"),
        ],
        "lifestyle_tips": [
            "🌊 Hydration first — keep a 1L water bottle visible at all times",
            "🚫 Remove junk food from the house — environment shapes behaviour",
            "🍽️ Eat slowly and chew 30 times per bite — improves digestion",
            "🕖 Fix meal times — eat within a 10-hour window (e.g., 8 AM–6 PM)",
            "💤 Prioritise 8 hrs sleep — the body detoxes during deep sleep",
        ],
    },
}

# ── Load model artifact ───────────────────────────────────────────────────────
@st.cache_resource
def load_artifact():
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


artifact = load_artifact()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🥗 DIP Diet Advisor")
st.markdown(
    "**Disciplined & Intelligent Plan (DIP Diet)** — "
    "Enter your health profile below and receive a personalised diet plan, "
    "7-day schedule, and actionable lifestyle recommendations."
)

if artifact is None:
    st.error(
        "⚠️ No trained model found at `models/model.pkl`.  \n"
        "Run `python train_model.py` first to train and save the model."
    )
    st.stop()

pipeline      = artifact["pipeline"]
label_encoder = artifact["label_encoder"]
feature_cols  = artifact["feature_cols"]
classes       = artifact["classes"]
model_name    = artifact["model_name"]
test_acc      = artifact["test_accuracy"]
cv_acc        = artifact["cv_accuracy"]

# ── Sidebar: model info ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 Model Info")
    st.markdown(f"**Algorithm:** {model_name}")
    st.markdown(f"**CV Accuracy:** `{cv_acc:.2%}`")
    st.markdown(f"**Test Accuracy:** `{test_acc:.2%}`")
    st.markdown("---")
    st.markdown("**Diet Profiles**")
    for profile_key, profile in PROFILES.items():
        st.markdown(f"{profile['emoji']} **{profile['title']}**")
        st.caption(profile["tagline"])
    st.markdown("---")
    st.markdown("**About DIP Diet**")
    st.caption(
        "The DIP (Disciplined & Intelligent Plan) Diet is rooted in natural, "
        "whole-food eating. It emphasises fruits, vegetables, legumes, millets, "
        "and adequate hydration while eliminating processed foods, refined sugar, "
        "and unhealthy fats."
    )

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("📝 Your Health Profile")

with st.form("diet_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🧍 Body Metrics**")
        age        = st.number_input("Age (years)",       min_value=10,  max_value=90,  value=30, step=1)
        weight_kg  = st.number_input("Weight (kg)",       min_value=30.0, max_value=200.0, value=70.0, step=0.5, format="%.1f")
        height_cm  = st.number_input("Height (cm)",       min_value=100, max_value=220, value=170, step=1)
        bmi        = round(weight_kg / ((height_cm / 100) ** 2), 1)
        st.info(f"🔢 Calculated BMI: **{bmi}**")

    with col2:
        st.markdown("**🏃 Lifestyle Habits**")
        activity_level = st.selectbox(
            "Physical Activity Level",
            options=[1, 2, 3, 4],
            format_func=lambda x: {1: "1 — Sedentary", 2: "2 — Light", 3: "3 — Moderate", 4: "4 — Very Active"}[x],
            index=1,
        )
        sleep_hours          = st.slider("Sleep Hours / Night",    min_value=3,   max_value=12,  value=7)
        water_intake_liters  = st.slider("Water Intake (litres/day)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
        stress_level         = st.selectbox(
            "Stress Level",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: {1: "1 — Very Low", 2: "2 — Low", 3: "3 — Moderate", 4: "4 — High", 5: "5 — Very High"}[x],
            index=2,
        )

    with col3:
        st.markdown("**🥦 Dietary Habits**")
        fruits_per_day     = st.slider("Fruit servings / day",      min_value=0, max_value=10, value=2)
        vegetables_per_day = st.slider("Vegetable servings / day",  min_value=0, max_value=10, value=3)
        processed_food_score = st.selectbox(
            "Processed Food Intake",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: {1: "1 — Never", 2: "2 — Rarely", 3: "3 — Sometimes", 4: "4 — Often", 5: "5 — Daily"}[x],
            index=2,
        )
        sugar_intake_score = st.selectbox(
            "Added Sugar / Sweets Intake",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: {1: "1 — None", 2: "2 — Minimal", 3: "3 — Moderate", 4: "4 — High", 5: "5 — Very High"}[x],
            index=2,
        )
        health_goal = st.selectbox(
            "Primary Health Goal",
            options=["weight_loss", "muscle_gain", "maintenance"],
            format_func=lambda x: {"weight_loss": "⚖️ Weight Loss", "muscle_gain": "💪 Muscle Gain", "maintenance": "🔄 Maintenance"}[x],
        )

    submitted = st.form_submit_button("🔍 Get My DIP Diet Plan", use_container_width=True, type="primary")

# ── Prediction & recommendations ──────────────────────────────────────────────
if submitted:
    user_data = {
        "age":                  age,
        "weight_kg":            weight_kg,
        "height_cm":            height_cm,
        "bmi":                  bmi,
        "activity_level":       activity_level,
        "sleep_hours":          sleep_hours,
        "water_intake_liters":  water_intake_liters,
        "fruits_per_day":       fruits_per_day,
        "vegetables_per_day":   vegetables_per_day,
        "processed_food_score": processed_food_score,
        "sugar_intake_score":   sugar_intake_score,
        "stress_level":         stress_level,
        "health_goal":          health_goal,
    }

    input_df = pd.DataFrame([user_data])[feature_cols]

    pred_enc        = pipeline.predict(input_df)[0]
    predicted_class = label_encoder.inverse_transform([pred_enc])[0]

    try:
        proba    = pipeline.predict_proba(input_df)[0]
        has_proba = True
    except AttributeError:
        has_proba = False

    profile = PROFILES[predicted_class]
    css_cls = profile["css_class"]

    st.markdown("---")
    st.subheader("🎯 Your DIP Diet Profile")

    # Profile card
    st.markdown(
        f'<div class="profile-card {css_cls}">'
        f'<h3>{profile["emoji"]} {profile["title"]}</h3>'
        f'<p><em>{profile["tagline"]}</em></p>'
        f'<p>{profile["description"]}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Confidence bar
    if has_proba:
        conf_df = pd.DataFrame(
            {"Profile": classes, "Confidence": proba}
        ).sort_values("Confidence", ascending=False)
        st.markdown('<p class="section-header">📊 Classification Confidence</p>', unsafe_allow_html=True)
        for _, row in conf_df.iterrows():
            emoji = PROFILES[row["Profile"]]["emoji"]
            label = PROFILES[row["Profile"]]["title"]
            st.progress(float(row["Confidence"]), text=f"{emoji} {label}: {row['Confidence']:.1%}")

    # ── Tabs for detailed recommendations ────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🥣 Daily Meal Plan", "📅 7-Day Schedule", "💡 Lifestyle Tips", "📋 Key Principles"]
    )

    with tab1:
        st.markdown("### 🥣 Your Personalised Daily Meal Plan")
        st.caption("Based on the DIP Diet philosophy — whole foods, no processing, natural nourishment.")
        for meal_time, meal_desc in profile["meal_plan"].items():
            st.markdown(
                f'<div class="tip-box"><strong>{meal_time}</strong><br>{meal_desc}</div>',
                unsafe_allow_html=True,
            )

    with tab2:
        st.markdown("### 📅 7-Day DIP Diet Schedule")
        st.caption("Follow this weekly rhythm to build sustainable habits.")
        schedule_df = pd.DataFrame(profile["weekly_schedule"], columns=["Day", "Focus & Action"])
        st.dataframe(schedule_df, use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("### 💡 Personalised Lifestyle Tips")
        for tip in profile["lifestyle_tips"]:
            st.markdown(
                f'<div class="tip-box">{tip}</div>',
                unsafe_allow_html=True,
            )

    with tab4:
        st.markdown("### 📋 Core DIP Diet Principles for You")
        for i, principle in enumerate(profile["key_principles"], 1):
            st.markdown(f"**{i}.** {principle}")

    # ── BMI gauge & quick stats ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📈 Your Health Snapshot")

    bmi_cat = (
        "Underweight" if bmi < 18.5 else
        "Normal weight" if bmi < 25 else
        "Overweight" if bmi < 30 else
        "Obese"
    )
    bmi_color = {"Underweight": "🟡", "Normal weight": "🟢", "Overweight": "🟠", "Obese": "🔴"}[bmi_cat]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("BMI", f"{bmi}", bmi_cat)
    m2.metric("Activity", f"Level {activity_level}/4")
    m3.metric("Sleep", f"{sleep_hours} hrs")
    m4.metric("Water", f"{water_intake_liters} L/day")
    m5.metric("Stress", f"Level {stress_level}/5")

    st.caption(f"{bmi_color} BMI Category: **{bmi_cat}**  |  Goal: **{health_goal.replace('_', ' ').title()}**")

    # ── Expand: input summary ─────────────────────────────────────────────────
    with st.expander("🔎 View full input summary"):
        st.dataframe(input_df, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "🌿 DIP Diet Advisor — Built with Python · scikit-learn · Streamlit  |  "
    "Based on the Disciplined & Intelligent Plan (DIP Diet) philosophy"
)
