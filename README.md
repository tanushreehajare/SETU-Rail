# 🚂 SETU-Rail

**Smart Emergency & Travel Understanding for Indian Railways**

A Lakehouse-native AI copilot built on Databricks Free Edition for the Bharat Bricks Hacks 2026 (IIT Madras) — Rail-Drishti track.

## What it does (1-line)
SETU-Rail fuses **Dhara** (cascade-aware delay prediction), **Vani** (Param-1 powered multilingual rulebook chatbot) and **Drishti** (passenger action panel) into a single Databricks app serving 12+ Indian languages.

## Architecture
![Architecture](docs/architecture.png)

See `ARCHITECTURE.md` for a detailed walkthrough.

## Databricks Tech Used
- **Delta Lake** (Medallion: Bronze/Silver/Gold) — partitioning, Z-ORDER, Time Travel
- **Apache Spark / PySpark** — millions-of-rows distributed ingestion & feature engineering
- **Unity Catalog + Volumes** — governed storage of raw files
- **Spark MLlib (GBTRegressor)** — distributed delay prediction
- **MLflow** — experiment tracking + model registry
- **Databricks Vector Search** — semantic retrieval over Railway Rules PDFs
- **Foundation Model Serving** (databricks-claude-sonnet-4) — generation
- **Databricks Genie** — natural language SQL over Delta tables
- **AI/BI Dashboard** — live analytics
- **Databricks Apps (Streamlit)** — user-facing frontend

## Indian Models Used
- **Param-1 (2.9B Instruct)** — via prompt-adapted RAG (BharatGen/IIT Bombay) 🇮🇳
- **IndicTrans2 (AI4Bharat)** — multilingual output across 22 Indic languages 🇮🇳

## How to run (end-to-end, fresh workspace)

### 1. Load this repo into Databricks
- Workspace → Repos → Add Repo → paste this GitHub URL
- All notebooks run on **Serverless** compute (no cluster setup needed)

### 2. Run notebooks in order
```
01_dhara_data_engineering/00_setup_catalog_schema_volume
01_dhara_data_engineering/01_ingest_train_timings
01_dhara_data_engineering/02_ingest_air_quality
01_dhara_data_engineering/03_ingest_rulebook_pdfs
01_dhara_data_engineering/04_build_silver_enriched
01_dhara_data_engineering/05_build_gold_features
02_drishti_ml_pipeline/01_synthesize_delays    (if no Kaggle CSV)
02_drishti_ml_pipeline/02_train_gbt_delay_model
02_drishti_ml_pipeline/03_shap_explainability
02_drishti_ml_pipeline/04_cascade_simulator
03_vani_rag_pipeline/01_chunk_and_embed_rules
03_vani_rag_pipeline/02_build_vector_index
03_vani_rag_pipeline/03_vani_rag_agent
04_genie_space/01_create_genie_space           (follow UI instructions)
05_dashboard/01_create_dashboard               (follow UI instructions)
06_application/00_deploy_app                   (follow UI instructions)
```

### 3. Open the deployed app
Copy the deployed app URL from `Compute → Apps → setu-rail-app`.

## Demo steps (what judges should click)
1. **Tab "🚂 Dhara"**: Enter Train `12615`, Station `MAS`, Date `2024-01-15` → see predicted delay + SHAP waterfall
2. **Tab "💬 Vani"**: Select language `Tamil`, ask *"என் ரயில் 4 மணி நேரம் தாமதம், எனக்கு என்ன உரிமை உள்ளது?"* → Vani replies in Tamil with citation from 1989 Act Section 124A
3. **Tab "🎫 Drishti"**: Enter PNR (mock) → shows delay forecast + applicable rights + auto-generated TDR action card
4. **Tab "📊 Analytics"**: Live Genie chat over Delta tables

## Catalog Structure
```
setu_rail (catalog)
├── bronze (schema)
│   ├── train_runs
│   ├── air_quality
│   └── rules_raw
├── silver (schema)
│   ├── runs_enriched
│   └── rules_chunks
└── gold (schema)
    ├── features_delay_ml
    ├── station_graph
    ├── predictions_daily
    └── vw_* (analytics views)
```

## Team
Built for Bharat Bricks Hacks 2026, IIT Madras campus — April 24–25, 2026.

## License
MIT
