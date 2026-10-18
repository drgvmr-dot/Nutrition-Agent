# 🥗 AI-Powered Nutrition Agent

> **IBM Granite + IBM Watsonx AI + RAG-powered Multi-Agent Nutrition System**

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the application
python run.py
```

Open your browser at: **http://localhost:8000**  
API Docs: **http://localhost:8000/docs**

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AI Nutrition Agent                        │
│                  IBM Granite-3.3-8B-Instruct                 │
├─────────────┬──────────────┬──────────────┬─────────────────┤
│  Nutrition  │     Diet     │    Health    │   Food Log &    │
│  Knowledge  │ Recommend.   │  Advisory    │   Feedback      │
│   Agent     │   Agent      │   Agent      │    Agent        │
├─────────────┴──────────────┴──────────────┴─────────────────┤
│              RAG Vector Store (ChromaDB)                     │
│         USDA FoodData + NIH + AHA + ADA Knowledge Base      │
├─────────────────────────────────────────────────────────────┤
│                 FastAPI Backend + REST API                   │
├─────────────────────────────────────────────────────────────┤
│          Visualization Dashboard (HTML + Chart.js)           │
└─────────────────────────────────────────────────────────────┘
```

---

## 🤖 Multi-Agent System

| Agent | Role | Technology |
|-------|------|-----------|
| 🔬 **Nutrition Knowledge Agent** | Fetches/summarizes nutritional data via RAG | IBM Granite + ChromaDB |
| 🥗 **Diet Recommendation Agent** | Personalized 7-day meal plans (Mifflin-St Jeor BMR) | IBM Granite |
| ⚕️ **Health Advisory Agent** | Disease-specific dietary guidance | IBM Granite |
| 📝 **Food Log & Feedback Agent** | Meal logging + nutritional analysis + weekly reports | IBM Granite |

---

## 📡 API Endpoints

### Nutrition Knowledge Agent
```
POST /api/nutrition/query       # Query nutritional data via RAG
```

### Diet Recommendation Agent
```
POST /api/diet/plan             # Generate personalized 7-day meal plan
```

### Health Advisory Agent
```
POST /api/health/advisory       # Get health-focused dietary advice
POST /api/health/disease-plan   # Disease-specific diet plan
```

### Food Log & Feedback Agent
```
POST /api/food-log/analyze              # Log and analyze a meal
GET  /api/food-log/daily-summary/{uid}  # Daily nutrition summary
GET  /api/food-log/weekly-report/{uid}  # Weekly nutrition report
GET  /api/food-log/logs/{uid}           # All logs
```

### Chat & Dashboard
```
POST /api/chat                  # Master chat (auto-routes to agents)
GET  /api/dashboard/{uid}       # Dashboard data
GET  /api/health                # System health check
```

---

## 🔧 Tech Stack

| Layer | Technology |
|-------|-----------|
| **AI Model** | IBM Granite 3.3-8B-Instruct |
| **AI Platform** | IBM Watsonx AI |
| **Backend** | FastAPI (Python) |
| **RAG** | ChromaDB + Sentence Transformers |
| **Frontend** | Vanilla HTML/CSS/JS + Chart.js |
| **Auth** | IBM Cloud API Key |

---

## 🌟 Features

- ✅ **Personalized Diet Plans** — Mifflin-St Jeor BMR calculation, 7-day meal plans
- ✅ **RAG Nutrition Database** — USDA, NIH, AHA, ADA sourced knowledge base
- ✅ **Meal Logging & Analysis** — Text-based meal logging with full nutritional breakdown
- ✅ **Disease Management** — Diabetes, hypertension, heart disease, kidney disease, PCOS, hypothyroidism
- ✅ **Preventive Health Focus** — Evidence-based dietary interventions
- ✅ **Multi-Agent Routing** — Smart chat interface auto-routes to best specialist
- ✅ **Visualization Dashboard** — Nutrient tracking, activity charts, macro distribution
- ✅ **Daily & Weekly Reports** — AI-generated summaries and recommendations
- ✅ **Cultural Preferences** — Supports diverse dietary patterns (halal, vegetarian, regional)

---

## 🏥 Supported Disease Plans

| Disease | Key Dietary Focus |
|---------|------------------|
| Type 2 Diabetes | Low GI foods, carb counting, fiber |
| Hypertension | DASH diet, sodium restriction |
| Heart Disease | Mediterranean diet, omega-3, LDL reduction |
| Kidney Disease | Potassium/phosphorus/protein restriction |
| PCOS | Insulin resistance management, anti-inflammatory |
| Hypothyroidism | Iodine, selenium, zinc optimization |

---

## 📁 Project Structure

```
PROJECT3/
├── app/
│   ├── main.py                        # FastAPI application
│   ├── config.py                      # IBM Watsonx AI config
│   ├── agents/
│   │   ├── nutrition_knowledge_agent.py
│   │   ├── diet_recommendation_agent.py
│   │   ├── health_advisory_agent.py
│   │   └── food_log_agent.py
│   └── rag/
│       └── vector_store.py            # ChromaDB + nutrition KB
├── static/
│   └── index.html                     # Dashboard frontend
├── requirements.txt
├── run.py
└── README.md
```

---

*Built with IBM Granite-3.3-8B-Instruct | IBM Watsonx AI | IBM Cloud*
