# 🚂 SETU Rail — Smart Emergency & Travel Understanding for Indian Railways

> **Bharat Bricks Hacks 2026 · IIT Madras · Rail-Drishti Track**

A Lakehouse-native AI copilot built on **Databricks Free Edition** that fuses cascade-aware delay prediction, a multilingual rulebook chatbot, and a passenger action panel — serving 12+ Indian languages.

---

## ✨ What It Does

| Module | Name | Role |
|--------|------|------|
| 🛤️ **Dhara** | *The Flow* | Medallion data engineering: Bronze→Silver→Gold from real data.gov.in CSVs + AQ data |
| 🤖 **Drishti** | *The Vision* | GBT delay predictor + cascade simulator for downstream trains |
| 💬 **Vani** | *The Voice* | RAG chatbot over Railway Act 1989 + General Rules 1976 in 12+ Indian languages |

---

## 🏗️ Architecture

![Architecture](images/architecture.png)

---

## 🧰 Databricks Tech Stack

| Feature | Usage |
|---------|-------|
| **Delta Lake (Bronze/Silver/Gold)** | Medallion architecture, Z-ORDER, Time Travel, CDF |
| **Apache Spark / PySpark** | Distributed ingestion, window functions, feature engineering |
| **Unity Catalog + Volumes** | Governed storage of CSVs and PDFs |
| **Spark MLlib — GBTRegressor** | Distributed delay prediction model |
| **MLflow** | Experiment tracking, model registry, RMSE/R² logging |
| **Databricks Vector Search** | Delta-synced semantic index over rulebook chunks |
| **Foundation Model Serving** | `databricks-claude-sonnet-4` / `databricks-param-1-2-9b-instruct` |
| **Databricks Genie** | Natural language SQL over Gold Delta tables |
| **AI/BI Dashboard** | Live PM2.5 × delay analytics |
| **Databricks Apps (Streamlit)** | Production user-facing frontend |

### 🇮🇳 Indian Models
- **Param-1 (2.9B Instruct)** — BharatGen / IIT Bombay (primary LLM for Vani)
- **IndicTrans2** — AI4Bharat (multilingual translation across 22 Indic languages)

---

## 📋 Prerequisites

| Requirement | Version / Notes |
|-------------|----------------|
| Databricks Free Edition workspace | [Sign up free](https://www.databricks.com/try-databricks) |
| Serverless compute | Enabled by default on Free Edition |
| Unity Catalog | Enabled in your workspace |
| Internet access | To clone repo and download data |

> ✅ **No local Python/pip setup needed.** All notebooks run on Databricks Serverless. No cluster configuration required.

---

## 🚀 Setup & Run (End-to-End, Fresh Workspace)

### Step 1 — Clone the repo into Databricks

```
Databricks Workspace → Repos → Add Repo
→ Paste this GitHub URL → Click "Create"
```

### Step 2 — Create the catalog, schema, and volume

```
Navigate to:  01_dhara_data_engineering/00_setup_catalog_schema_volume
Click: Run All
```

This creates:
- Catalog: `setu_rail`
- Schemas: `bronze`, `silver`, `gold`
- Volume: `/Volumes/setu_rail/bronze/raw_files/`

### Step 3 — Upload source data files

Upload these files to `/Volumes/setu_rail/bronze/raw_files/`:

| File | Source | Notes |
|------|--------|-------|
| `train_*.csv` | [data.gov.in Indian Railways](https://data.gov.in) | One or more train timing CSVs |
| `air_quality_*.csv` | [data.gov.in CPCB](https://data.gov.in) | Any CPCB pollutant CSV |
| `1976_general_rules_railways_pdf.pdf` | [Indian Railways official PDFs](https://indianrailways.gov.in) | Rulebook |
| `the_railways_act__1989.pdf` | [Indian Railways official PDFs](https://indianrailways.gov.in) | Act PDF |
| `railways_running_history.csv` *(optional)* | [Kaggle](https://www.kaggle.com) | Real delay data; system synthesizes if absent |

```
Databricks Workspace → Catalog → setu_rail → bronze → raw_files → Upload
```

### Step 4 — Run notebooks in order

Run each notebook by opening it and clicking **Run All**:

#### 📦 Phase 1: Data Engineering (Dhara)

```
Step 1:  01_dhara_data_engineering/00_setup_catalog_schema_volume
Step 2:  01_dhara_data_engineering/01_ingest_train_timings_demo
Step 3:  01_dhara_data_engineering/02_ingest_air_quality
Step 4:  01_dhara_data_engineering/03_ingest_rulebook_pdfs
Step 5:  01_dhara_data_engineering/04_build_silver_enriched
Step 6:  01_dhara_data_engineering/05_build_gold_features
```

#### 🤖 Phase 2: ML Pipeline (Drishti)

```
Step 7:  02_drishti_ml_pipeline/01_synthesize_delays     ← skip if Kaggle CSV present
Step 8:  02_drishti_ml_pipeline/02_train_gbt_delay_model
Step 9:  02_drishti_ml_pipeline/03_shap_explainability
Step 10: 02_drishti_ml_pipeline/04_cascade_simulator
```

#### 💬 Phase 3: RAG Pipeline (Vani)

```
Step 11: 03_vani_rag_pipeline/01_chunk_and_embed_rules
Step 12: 03_vani_rag_pipeline/02_build_vector_index      ← takes ~5 min for endpoint
Step 13: 03_vani_rag_pipeline/03_vani_rag_agent
```

#### 📊 Phase 4: Analytics (UI — follow in-notebook instructions)

```
Step 14: 04_genie_space/01_create_genie_space            ← follow UI steps in notebook
Step 15: 05_dashboard/01_create_dashboard                ← follow UI steps in notebook
Step 16: 06_application/00_deploy_app                    ← follow UI steps in notebook
```

### Step 5 — Open the deployed app

```
Databricks Workspace → Compute → Apps → setu-rail-app → Copy URL
```

---

## 🎯 Demo Steps (What Judges Should Click)

### Tab 1: 🚂 Dhara (Delay Prediction)
1. Enter **Train No**: `12615`
2. Enter **Station**: `MAS`
3. Enter **Date**: `2024-01-15`
4. Click **Predict** → See predicted delay + SHAP waterfall chart

### Tab 2: 💬 Vani (Multilingual Rulebook Chat)
1. Select **Language**: `Tamil`
2. Type: `என் ரயில் 4 மணி நேரம் தாமதம், எனக்கு என்ன உரிமை உள்ளது?`
3. See Vani reply **in Tamil** with citation from Railways Act 1989, Section 124A

### Tab 3: 🎫 Drishti (Passenger Rights)
1. Enter any **PNR** (mock)
2. See: delay forecast + applicable rights + auto-generated TDR action card

### Tab 4: 📊 Analytics (Live Genie)
1. Type: `Top 5 cities by PM2.5`
2. Type: `Average delay by month`
3. See natural language → SQL → chart

---

## 📂 Catalog Structure

```
setu_rail (catalog)
├── bronze (schema)
│   ├── sr_timings                 ← raw train CSVs
│   ├── air_quality_raw            ← raw CPCB pollutant data
│   ├── air_quality                ← pivoted wide format
│   ├── railways_running_history   ← Kaggle real delays (if uploaded)
│   ├── rules_raw                  ← PDF chunks with page metadata
│   ├── trains                     ← trains.json reference
│   └── stations                   ← stations.json reference
├── silver (schema)
│   ├── sr_enriched                ← normalized + joined timings
│   ├── schedule_features          ← ML features (stop_seq, dwell, AQ)
│   ├── zone_state_map             ← zone → state → PM2.5 lookup
│   └── rules_chunks               ← chunked PDFs (CDF enabled)
└── gold (schema)
    ├── features_delay_ml          ← labeled GBT training data (Z-ORDERed)
    ├── station_train_graph        ← cascade simulator edges
    ├── predictions_daily          ← batch inference output
    ├── rules_vs_index             ← Vector Search index
    └── vw_* (analytics views)
```

---

## 📈 MLflow Metrics Logged

| Metric | Value (typical) |
|--------|----------------|
| `rmse` | ~8–12 min |
| `r2`   | ~0.72–0.85 |
| `mae`  | ~6–9 min |
| `feature_count` | 11 |
| Model registry | `setu_rail_delay_gbt` |

Navigate to: **Experiments → setu_rail_delay_model** to see all runs.

---

## ✍️ 500-Character Submission Write-Up

> SETU Rail is a Databricks Lakehouse AI copilot for Indian Railways that fuses real-time delay prediction, cascade propagation modeling, and multilingual passenger rights assistance. Built on Delta Lake (Bronze→Silver→Gold), GBTRegressor, MLflow, Databricks Vector Search, and Param-1 (🇮🇳), it answers passenger queries in 12+ Indian languages with legally-cited responses from the Railways Act 1989 — powered entirely on Databricks Free Edition.

---

## ✅ Bonus Evaluation

| Criterion | Status | Notes |
|-----------|--------|-------|
| MLflow logs | ✅ YES | RMSE, R², MAE, feature importance logged |
| Metrics included | ✅ YES | Distribution validation vs MoR published stats |
| Indian model (Param-1) | ✅ YES | `databricks-param-1-2-9b-instruct` as primary LLM |
| IndicTrans2 | ✅ YES | 22 Indic language translation |
| BhashaBench | ⚠️ Partial | Not formally benchmarked; add via: run Vani on BhashaBench QA pairs, log accuracy to MLflow |

### Fastest Way to Add BhashaBench
```python
# In 03_vani_rag_agent notebook, add this cell:
import mlflow
bhasha_qs = [...]  # Load BhashaBench evaluation set
with mlflow.start_run(run_name="vani_bhasha_eval"):
    for q, expected_lang in bhasha_qs:
        result = vani_answer(q)
        mlflow.log_metric("answer_accuracy", score(result, expected_lang))
```

---

## 🏆 Judging Checklist

- [x] Runs on fresh Databricks Free Edition workspace
- [x] All commands are copy-paste ready
- [x] Medallion architecture (Bronze/Silver/Gold) with Delta Lake
- [x] Spark MLlib distributed training (GBTRegressor)
- [x] MLflow experiment tracking + model registry
- [x] Databricks Vector Search (Delta-synced, BGE-Large-EN)
- [x] Databricks Genie (NL → SQL over Gold tables)
- [x] AI/BI Dashboard (PM2.5 × delay correlation)
- [x] Databricks Apps (Streamlit frontend)
- [x] Indian LLM: Param-1 (primary) + IndicTrans2 (translation)
- [x] 12+ Indian languages supported
- [x] Cascade delay propagation (innovation differentiator)
- [x] Legal citations from Railways Act 1989

---

## 👥 Team

Built for **Bharat Bricks Hacks 2026** · IIT Madras campus · April 24–25, 2026 · Rail-Drishti Track

## 📄 License

MIT
