# LLM Müştəri Dəstək Agent-i — Tam Data-Driven Layihə

LLM Agent + Tool Calling konsepti (`check_order`, `create_ticket`, `search_faq`)
üçün **2 real dataset** üzərində qurulmuş tam data-driven backend: EDA,
5 ML modeli və 2-modullu interaktiv Streamlit dashboard.

## Dataset-lər

| Dataset | Ölçü | Agent Aləti | Mənbə |
|---|---|---|---|
| **Olist Brazilian E-Commerce** | 95,621 real sifariş (2016-2018) | `check_order` | [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) |
| **Bitext Customer Support** | 26,872 real-tərz mesaj, 27 intent | `search_faq` / `create_ticket` | [GitHub](https://github.com/bitext/customer-support-llm-chatbot-training-dataset) |

**Niyə 2 dataset birləşdirilib:** Real LLM Agent-lər struktur backend data
(sifariş sistemi) ilə NLP qatını (istifadəçi niyyətinin başa düşülməsi)
birlikdə istifadə edir. Hər iki dataset **tam real** məlumatdır (əvvəlki
sınaqda istifadə edilən tam sintetik/təsadüfi CSV-dən fərqli olaraq) və
təsdiqlənmiş, mənalı statistik siqnal ehtiva edir:
- Olist: gecikmə ↔ rəy balı korrelyasiyası **r ≈ -0.27**
- Bitext: intent təsnifatı **99.5%** dəqiqlik (baseline 3.7%)

## Qovluq strukturu

```
project/
├── notebook/
│   └── llm_agent_combined_final_project.ipynb   ← Google Colab notebook (13 bölmə)
├── streamlit_app/
│   ├── app.py                                   ← 2-modullu dashboard
│   └── requirements.txt
├── models/                                      ← 6 öyrədilmiş model (.pkl)
│   ├── tfidf_vectorizer.pkl
│   ├── intent_classifier.pkl
│   ├── category_classifier.pkl
│   ├── tool_router_classifier.pkl
│   ├── olist_late_delivery_classifier.pkl
│   ├── olist_late_encoders.pkl / olist_late_feature_order.pkl
│   ├── olist_bad_review_classifier.pkl
│   ├── olist_review_encoders.pkl / olist_review_feature_order.pkl
│   └── metrics_report.json
├── data/
│   ├── olist/                     ← Xam Olist CSV-ləri (8 fayl)
│   ├── bitext/                    ← Xam Bitext CSV-si
│   ├── olist_clean.parquet        ← Təmizlənmiş Olist (dashboard üçün)
│   └── bitext_clean.parquet       ← Təmizlənmiş Bitext (dashboard üçün)
└── train_models.py                ← Bütün 5 modeli yenidən öyrətmək üçün
```

## 🔧 Modellərin Xülasəsi

| Model | Dataset | Tapşırıq | Nəticə | Baseline |
|---|---|---|---|---|
| A. İntent Təsnifatı | Bitext | 27-sinif NLP | 99.5% acc. | 3.7% |
| B. Kateqoriya Təsnifatı | Bitext | 11-sinif NLP | 99.8% acc. | 9.1% |
| C. Tool-Routing | Bitext | 3-sinif NLP | 99.6% acc. | 33.3% |
| D. Gecikmə Riski | Olist | Binar (imbalanced) | 0.75 ROC-AUC | 0.50 |
| E. Narazılıq Riski | Olist | Binar (imbalanced) | 0.76 ROC-AUC | 0.50 |

**Metodoloji qeyd:** Olist modelləri üçün accuracy əvəzinə ROC-AUC/F1
istifadə edilib, çünki hədəf dəyişənlər balanssızdır (gecikmə ~6.8%,
pis rəy ~13.1%) — accuracy tək başına aldadıcı ola bilər. Bax notebook
Bölmə 4 və dashboard-un "Model Performansı" tab-ı.

## 1. Google Colab-da işə salmaq

1. `notebook/llm_agent_combined_final_project.ipynb` faylını Colab-a yükləyin.
2. **Olist hissəsi üçün:** `data/olist/` qovluğundakı 8 CSV faylını Colab-a
   yükləyin (və ya notebook-dakı GitHub alternativ linkini istifadə edin).
3. **Bitext hissəsi üçün:** heç bir upload lazım deyil — dataset birbaşa
   GitHub-dan oxunur.
4. Bütün hüceyrələri ardıcıl işə salın (`Runtime → Run all`).
5. Notebook sonunda bütün 6 model `models/` qovluğuna saxlanılır.

## 2. Streamlit Dashboard-u lokal işə salmaq

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

Dashboard 3 rejimə bölünüb (sol paneldəki radio seçimi ilə):
- **🌐 Agent Ümumi Baxış** — hər iki modulun xülasəsi, agent axını diaqramı
- **📦 Olist — check_order** — 4 tab: Ümumi Baxış, Ətraflı Analitika, Model Performansı, Risk Kalkulyatoru
- **💬 Bitext — search_faq/create_ticket** — 4 tab: Ümumi Baxış, Ətraflı Analitika, Model Performansı, Canlı Agent Demo

Yenidən öyrətmək üçün:
```bash
python3 train_models.py
```
