"""
LLM Müştəri Dəstək Agent-i — Tam Dashboard (Olist + Bitext)
================================================================
Data Science Final Layihəsi | Streamlit Dashboard

İşlətmək üçün:
    streamlit run app.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="LLM Agent Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATE_ROOTS = [HERE, os.path.dirname(HERE)]

def find_path(*parts):
    for root in CANDIDATE_ROOTS:
        p = os.path.join(root, *parts)
        if os.path.exists(p):
            return p
    return os.path.join(CANDIDATE_ROOTS[0], *parts)

MODELS_DIR = find_path("models")
OLIST_PATH = find_path("data", "olist_clean.parquet")
BITEXT_PATH = find_path("data", "bitext_clean.parquet")

FLAG_LABELS = {
    "morphological_variation": "Morfoloji variasiya", "semantic_variation": "Semantik variasiya",
    "basic_syntax": "Sadə sintaksis", "interrogative": "Sual forması",
    "coordinated_syntax": "Mürəkkəb cümlə", "negation": "İnkar", "polite": "Nəzakətli",
    "colloquial": "Danışıq dili", "offensive": "Kobud dil", "keyword_mode": "Açar söz rejimi",
    "abbreviation": "Abreviatura", "typo_error": "Orfoqrafik səhv",
}
TOOL_LABELS = {"check_order": "🔍 check_order", "create_ticket": "🎫 create_ticket", "search_faq": "📚 search_faq"}


@st.cache_data(show_spinner="Data yüklənir...")
def load_data():
    olist = pd.read_parquet(OLIST_PATH)
    olist["order_purchase_timestamp"] = pd.to_datetime(olist["order_purchase_timestamp"])
    bitext = pd.read_parquet(BITEXT_PATH)
    return olist, bitext


@st.cache_resource(show_spinner="Modellər yüklənir...")
def load_models():
    m = {}
    try:
        m["tfidf"] = joblib.load(f"{MODELS_DIR}/tfidf_vectorizer.pkl")
        m["intent_clf"] = joblib.load(f"{MODELS_DIR}/intent_classifier.pkl")
        m["cat_clf"] = joblib.load(f"{MODELS_DIR}/category_classifier.pkl")
        m["tool_clf"] = joblib.load(f"{MODELS_DIR}/tool_router_classifier.pkl")
        m["late_clf"] = joblib.load(f"{MODELS_DIR}/olist_late_delivery_classifier.pkl")
        m["late_encoders"] = joblib.load(f"{MODELS_DIR}/olist_late_encoders.pkl")
        m["late_features"] = joblib.load(f"{MODELS_DIR}/olist_late_feature_order.pkl")
        m["rev_clf"] = joblib.load(f"{MODELS_DIR}/olist_bad_review_classifier.pkl")
        m["rev_encoders"] = joblib.load(f"{MODELS_DIR}/olist_review_encoders.pkl")
        m["rev_features"] = joblib.load(f"{MODELS_DIR}/olist_review_feature_order.pkl")
        with open(f"{MODELS_DIR}/metrics_report.json") as f:
            m["metrics"] = json.load(f)
    except FileNotFoundError as e:
        st.warning(f"Bəzi model faylları tapılmadı: {e}. Əvvəlcə `train_models.py`-ı işə salın.")
    return m


olist_full, bitext_full = load_data()
models = load_models()

CAT_FEATURES_OLIST = ["customer_state", "seller_state", "product_category_name_english", "payment_type"]

# ---------------------------------------------------------------------
# HEADER + top-level module switch
# ---------------------------------------------------------------------
st.title("🤖 LLM Müştəri Dəstək Agent-i — Tam Dashboard")
st.caption("Data Science Final Layihəsi · Olist (check_order) + Bitext (search_faq/create_ticket)")

module = st.sidebar.radio("📂 Modul seçin:", ["🌐 Agent Ümumi Baxış", "📦 Olist — check_order", "💬 Bitext — search_faq/create_ticket"])

st.sidebar.markdown("---")
st.sidebar.caption(
    "Bu dashboard 2 real dataset üzərində qurulub: **Olist** (95,621 real "
    "Braziliya sifarişi, 2016-2018) və **Bitext** (26,872 real-tərz müştəri "
    "mesajı). Onlar birlikdə LLM Agent-in `check_order`, `create_ticket` və "
    "`search_faq` alətlərinin data-driven backend-ini təşkil edir."
)

# =======================================================================
# MODULE 0 — AGENT OVERVIEW
# =======================================================================
if module == "🌐 Agent Ümumi Baxış":
    st.header("🌐 Agent Arxitekturası — Ümumi Baxış")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        #### 📦 Olist Dataset → `check_order`
        - **95,621** real çatdırılmış sifariş (2016-2018)
        - Model D: Gecikmə riski proqnozu (ROC-AUC)
        - Model E: Narazılıq (pis rəy) riski proqnozu (ROC-AUC)
        - Real korrelyasiya: gecikmə ↔ aşağı rəy balı (**r ≈ -0.27**)
        """)
    with col2:
        st.markdown("""
        #### 💬 Bitext Dataset → `search_faq` / `create_ticket`
        - **26,872** real-tərz müştəri mesajı, 27 intent
        - Model A: İntent təsnifatı (99.5% dəqiqlik)
        - Model B: Kateqoriya təsnifatı (99.8% dəqiqlik)
        - Model C: Tool-routing modeli (99.6% dəqiqlik)
        """)

    st.markdown("---")
    if "metrics" in models:
        met = models["metrics"]
        st.subheader("📊 Bütün Modellərin Xülasəsi")
        summary = pd.DataFrame([
            {"Model": "A. İntent (Bitext)", "Metrik": "Accuracy", "Dəyər": met["intent_classifier"]["accuracy"],
             "Baseline": met["intent_classifier"]["baseline"]},
            {"Model": "B. Kateqoriya (Bitext)", "Metrik": "Accuracy", "Dəyər": met["category_classifier"]["accuracy"],
             "Baseline": met["category_classifier"]["baseline"]},
            {"Model": "C. Tool-Routing (Bitext)", "Metrik": "Accuracy", "Dəyər": met["tool_router_classifier"]["accuracy"],
             "Baseline": met["tool_router_classifier"]["baseline"]},
            {"Model": "D. Gecikmə Riski (Olist)", "Metrik": "ROC-AUC", "Dəyər": met["late_delivery_classifier"]["roc_auc"],
             "Baseline": 0.50},
            {"Model": "E. Narazılıq Riski (Olist)", "Metrik": "ROC-AUC", "Dəyər": met["bad_review_classifier"]["roc_auc"],
             "Baseline": 0.50},
        ])
        fig = px.bar(summary, x="Model", y=["Dəyər", "Baseline"], barmode="group",
                     title="Model Performansı vs Baseline", color_discrete_sequence=["#2E86AB", "#C0C0C0"])
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔄 Agent Axını")
    st.markdown("""
    ```
    İstifadəçi mesajı
          │
          ▼
    Model C: Tool-Routing (Bitext, mətn əsaslı)
          │
          ├─── check_order ────► Model D + E (Olist, sifariş məlumatları əsaslı)
          │                       → gecikmə/narazılıq riski qiymətləndirməsi
          │
          ├─── create_ticket ──► İnsan operatora / dəstək komandasına yönləndirmə
          │
          └─── search_faq ─────► Model A: İntent → FAQ bazasından cavab
    ```
    Canlı simulyasiya üçün **"🤖 Canlı Agent Demo"** bölməsinə keçin (Bitext modulunun daxilində).
    """)

# =======================================================================
# MODULE 1 — OLIST
# =======================================================================
elif module == "📦 Olist — check_order":
    st.sidebar.markdown("### Olist Filtrləri")
    sel_state = st.sidebar.multiselect("Müştəri Ştatı", sorted(olist_full["customer_state"].unique()),
                                         default=sorted(olist_full["customer_state"].unique()))
    sel_cat = st.sidebar.multiselect("Məhsul Kateqoriyası",
                                       sorted(olist_full["product_category_name_english"].unique()),
                                       default=sorted(olist_full["product_category_name_english"].unique()))

    mask = olist_full["customer_state"].isin(sel_state) & olist_full["product_category_name_english"].isin(sel_cat)
    olist = olist_full[mask]
    if olist.empty:
        st.warning("Seçilmiş filtrlərə uyğun sifariş tapılmadı.")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Ümumi Baxış", "🔍 Ətraflı Analitika", "📈 Model Performansı", "🧮 Risk Kalkulyatoru"])

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ümumi Sifariş", f"{len(olist):,}")
        c2.metric("Gecikmə Nisbəti", f"{olist['is_late'].mean()*100:.1f}%")
        c3.metric("Pis Rəy Nisbəti", f"{olist['is_bad_review'].mean()*100:.1f}%")
        c4.metric("Orta Rəy Balı", f"{olist['review_score'].mean():.2f} / 5")

        col1, col2 = st.columns(2)
        with col1:
            vol = olist.set_index("order_purchase_timestamp").resample("ME").size().reset_index(name="count")
            fig = px.line(vol, x="order_purchase_timestamp", y="count", markers=True, title="Aylıq Sifariş Həcmi")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.histogram(olist, x="review_score", title="Rəy Balı Bölgüsü", color_discrete_sequence=["coral"])
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            state_counts = olist["customer_state"].value_counts().head(10).reset_index()
            state_counts.columns = ["state", "count"]
            fig = px.bar(state_counts, x="count", y="state", orientation="h", title="Top 10 Ştat — Sifariş Sayı",
                         color="count", color_continuous_scale="Blues")
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
        with col4:
            pay_counts = olist["payment_type"].value_counts().reset_index()
            pay_counts.columns = ["payment_type", "count"]
            fig = px.pie(pay_counts, names="payment_type", values="count", hole=0.45, title="Ödəniş Növü Bölgüsü")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("🔑 Əsas Tapıntı: Gecikmə → Narazılıq Əlaqəsi")
        col1, col2 = st.columns(2)
        with col1:
            late_review = olist.groupby("is_late")["review_score"].mean().reset_index()
            late_review["is_late"] = late_review["is_late"].map({0: "Vaxtında", 1: "Gecikmiş"})
            fig = px.bar(late_review, x="is_late", y="review_score", color="is_late",
                         title="Orta Rəy Balı — Çatdırılma Statusuna görə",
                         color_discrete_sequence=["seagreen", "indianred"])
            fig.update_layout(yaxis_range=[0, 5])
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            corr = olist["delivery_delay_days"].corr(olist["review_score"])
            st.metric("Korrelyasiya (gecikmə ↔ rəy balı)", f"{corr:.3f}")
            fig = px.box(olist[olist["delivery_delay_days"].between(-30, 30)],
                         x="review_score", y="delivery_delay_days", color="review_score",
                         title="Gecikmə Günü Bölgüsü — Rəy Balına görə")
            fig.add_hline(y=0, line_dash="dash")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Region üzrə Gecikmə Analizi")
        col3, col4 = st.columns(2)
        with col3:
            state_late = olist.groupby("customer_state")["is_late"].mean().sort_values(ascending=False).head(10).reset_index()
            fig = px.bar(state_late, x="is_late", y="customer_state", orientation="h",
                         title="Ən Yüksək Gecikmə Nisbəti — Top 10 Ştat", color="is_late",
                         color_continuous_scale="Oranges")
            st.plotly_chart(fig, use_container_width=True)
        with col4:
            cross_late = olist.groupby("cross_state")["is_late"].mean().reset_index()
            cross_late["cross_state"] = cross_late["cross_state"].map({0: "Eyni ştat", 1: "Fərqli ştat"})
            fig = px.bar(cross_late, x="cross_state", y="is_late", color="cross_state",
                         title="Gecikmə Nisbəti — Satıcı/Müştəri Eyni Ştatdadırmı?")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Məhsul Kateqoriyası üzrə Məmnuniyyət")
        cat_review = olist.groupby("product_category_name_english")["review_score"].agg(["mean", "count"])
        cat_review = cat_review[cat_review["count"] >= 20].sort_values("mean").reset_index()
        fig = px.bar(pd.concat([cat_review.head(8), cat_review.tail(8)]), x="mean", y="product_category_name_english",
                     orientation="h", title="Ən Aşağı / Ən Yüksək Orta Rəy Balı (min. 20 sifariş)",
                     color="mean", color_continuous_scale="RdYlGn")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if "metrics" not in models:
            st.info("Model metrikaları tapılmadı.")
        else:
            met = models["metrics"]
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### D. Gecikmə Riski Modeli")
                st.metric("ROC-AUC", f"{met['late_delivery_classifier']['roc_auc']:.3f}")
                st.metric("F1", f"{met['late_delivery_classifier']['f1']:.3f}")
                st.metric("Accuracy", f"{met['late_delivery_classifier']['accuracy']*100:.1f}%")
                st.caption(f"Majority baseline: {met['late_delivery_classifier']['baseline']*100:.1f}% "
                           "(diqqət: accuracy baseline-dan aşağı ola bilər — aşağıdakı qeydə bax)")
            with col2:
                st.markdown("#### E. Narazılıq Riski Modeli")
                st.metric("ROC-AUC", f"{met['bad_review_classifier']['roc_auc']:.3f}")
                st.metric("F1", f"{met['bad_review_classifier']['f1']:.3f}")
                st.metric("Accuracy", f"{met['bad_review_classifier']['accuracy']*100:.1f}%")
                st.caption(f"Majority baseline: {met['bad_review_classifier']['baseline']*100:.1f}%")

            st.markdown("---")
            st.markdown("""
            #### 🔎 Metodoloji Qeyd — Accuracy Tələsi
            Hər iki modelin accuracy-si "həmişə əksəriyyət sinifini proqnozlaşdır"
            baseline-ından **aşağıdır** (məsələn gecikmə nadir hadisədir — cəmi ~6.8%,
            ona görə "heç vaxt gecikmə olmayacaq" desək belə 93%+ accuracy əldə edilər).
            Bu, pis model demək DEYİL: `class_weight="balanced"` bilərəkdən accuracy-ni
            qurban verib az-rast-gəlinən sinifi tuta bilmək üçün optimallaşdırılıb.
            **ROC-AUC (0.75-0.76)** həqiqi proqnozlaşdırma gücünü göstərir — bu, 0.50
            (təsadüfi) səviyyəsindən əhəmiyyətli dərəcədə yuxarıdır.
            """)

            feat_df = pd.DataFrame(met["late_delivery_classifier"]["top_features"])
            fig = px.bar(feat_df, x="importance", y="feature", orientation="h",
                         title="Feature Importance — Gecikmə Riski Modeli")
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("🧮 Canlı Risk Kalkulyatoru")
        st.caption("Sifariş parametrlərini daxil edin — model gecikmə və narazılıq riskini real vaxtda hesablayacaq.")

        if not all(k in models for k in ["late_clf", "rev_clf"]):
            st.info("Modellər yüklənməyib.")
        else:
            colA, colB, colC = st.columns(3)
            with colA:
                in_state = st.selectbox("Müştəri ştatı", sorted(olist_full["customer_state"].unique()))
                in_seller_state = st.selectbox("Satıcı ştatı", sorted(olist_full["seller_state"].dropna().unique()))
                in_category = st.selectbox("Məhsul kateqoriyası", sorted(olist_full["product_category_name_english"].unique()))
            with colB:
                in_payment = st.selectbox("Ödəniş növü", sorted(olist_full["payment_type"].unique()))
                in_installments = st.slider("Taksit sayı", 1, 24, 3)
                in_items = st.slider("Məhsul sayı", 1, 10, 1)
            with colC:
                in_price = st.slider("Ümumi qiymət ($)", 10, 2000, 150)
                in_freight = st.slider("Çatdırılma haqqı ($)", 0, 200, 25)
                in_approval_delay = st.slider("Təsdiq gecikməsi (saat)", 0, 72, 5)

            if st.button("🎯 Riski hesabla"):
                cross_state = int(in_state != in_seller_state)
                base_row = {
                    "n_items": in_items, "total_price": float(in_price), "total_freight": float(in_freight),
                    "avg_price": float(in_price) / in_items, "payment_installments": in_installments,
                    "approval_delay_hours": float(in_approval_delay), "cross_state": cross_state,
                    "purchase_month": 6, "purchase_dow": 2,
                    "customer_state": in_state, "seller_state": in_seller_state,
                    "product_category_name_english": in_category, "payment_type": in_payment,
                }

                late_row = pd.DataFrame([base_row])[models["late_features"]].copy()
                for c in CAT_FEATURES_OLIST:
                    late_row[c] = models["late_encoders"][c].transform(late_row[[c]])
                late_risk = models["late_clf"].predict_proba(late_row)[0][1]

                rev_row_dict = dict(base_row)
                rev_row_dict["delivery_delay_days"] = 3 if late_risk > 0.5 else -5
                rev_row_dict["purchase_to_delivery_days"] = 12
                rev_row = pd.DataFrame([rev_row_dict])[models["rev_features"]].copy()
                for c in CAT_FEATURES_OLIST:
                    rev_row[c] = models["rev_encoders"][c].transform(rev_row[[c]])
                bad_review_risk = models["rev_clf"].predict_proba(rev_row)[0][1]

                col1, col2 = st.columns(2)
                with col1:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number", value=late_risk * 100, title={"text": "Gecikmə Riski (%)"},
                        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "darkred" if late_risk > 0.3 else "seagreen"}}))
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number", value=bad_review_risk * 100, title={"text": "Narazılıq Riski (%)"},
                        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "darkred" if bad_review_risk > 0.2 else "seagreen"}}))
                    st.plotly_chart(fig, use_container_width=True)

                if late_risk > 0.3:
                    st.warning("⚠️ Yüksək gecikmə riski — agent müştəriyə proaktiv bildiriş göndərməyi tövsiyə edir.")

# =======================================================================
# MODULE 2 — BITEXT
# =======================================================================
else:
    st.sidebar.markdown("### Bitext Filtrləri")
    sel_tool = st.sidebar.multiselect("Agent Aləti", sorted(bitext_full["agent_tool"].unique()),
                                        default=sorted(bitext_full["agent_tool"].unique()))
    sel_category = st.sidebar.multiselect("Kateqoriya", sorted(bitext_full["category"].unique()),
                                            default=sorted(bitext_full["category"].unique()))

    mask = bitext_full["agent_tool"].isin(sel_tool) & bitext_full["category"].isin(sel_category)
    bt = bitext_full[mask]
    if bt.empty:
        st.warning("Seçilmiş filtrlərə uyğun mesaj tapılmadı.")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Ümumi Baxış", "🔍 Ətraflı Analitika", "📈 Model Performansı", "🤖 Canlı Agent Demo"])

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ümumi Mesaj", f"{len(bt):,}")
        c2.metric("İntent Sayı", f"{bt['intent'].nunique()}")
        c3.metric("Kateqoriya Sayı", f"{bt['category'].nunique()}")
        c4.metric("Orta Söz Sayı", f"{bt['instruction_word_count'].mean():.1f}")

        col1, col2 = st.columns(2)
        with col1:
            tool_counts = bt["agent_tool"].value_counts().reset_index()
            tool_counts.columns = ["tool", "count"]
            fig = px.pie(tool_counts, names="tool", values="count", hole=0.45, title="Agent Tool Bölgüsü")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            cat_counts = bt["category"].value_counts().reset_index()
            cat_counts.columns = ["category", "count"]
            fig = px.bar(cat_counts, x="count", y="category", orientation="h", title="Kateqoriya üzrə Mesaj Sayı",
                         color="count", color_continuous_scale="Blues")
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

        intent_counts = bt["intent"].value_counts().reset_index()
        intent_counts.columns = ["intent", "count"]
        fig = px.bar(intent_counts, x="count", y="intent", orientation="h", color="count",
                     color_continuous_scale="Teal", height=650, title="İntent Bölgüsü (27 sinif)")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Mesaj Uzunluğu — Kateqoriya üzrə")
        fig = px.box(bt, x="category", y="instruction_word_count", color="category", title="Söz Sayı Bölgüsü")
        fig.update_layout(showlegend=False)
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Dil Variasiyası Bayraqları")
        flag_cols = [f"flag_{k}" for k in FLAG_LABELS]
        flag_means = (bt[flag_cols].mean() * 100).sort_values()
        flag_df = pd.DataFrame({"flag": [FLAG_LABELS[c.replace("flag_", "")] for c in flag_means.index],
                                  "pct": flag_means.values})
        fig = px.bar(flag_df, x="pct", y="flag", orientation="h", title="Dil Variasiyası Tezliyi (%)",
                     color="pct", color_continuous_scale="Oranges")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Nümunə Mesajlar")
        sample_intent = st.selectbox("İntent seç:", sorted(bt["intent"].unique()))
        st.dataframe(bt[bt["intent"] == sample_intent][["instruction", "category", "agent_tool", "flags"]].head(10),
                     use_container_width=True)

    with tab3:
        if "metrics" not in models:
            st.info("Model metrikaları tapılmadı.")
        else:
            met = models["metrics"]
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("#### A. İntent (27 sinif)")
                st.metric("Accuracy", f"{met['intent_classifier']['accuracy']*100:.1f}%")
                st.caption(f"Baseline: {met['intent_classifier']['baseline']*100:.1f}%")
            with c2:
                st.markdown("#### B. Kateqoriya (11 sinif)")
                st.metric("Accuracy", f"{met['category_classifier']['accuracy']*100:.1f}%")
                st.caption(f"Baseline: {met['category_classifier']['baseline']*100:.1f}%")
            with c3:
                st.markdown("#### C. Tool-Routing (3 sinif)")
                st.metric("Accuracy", f"{met['tool_router_classifier']['accuracy']*100:.1f}%")
                st.caption(f"Baseline: {met['tool_router_classifier']['baseline']*100:.1f}% "
                           f"| NB: {met['tool_router_classifier']['naive_bayes_accuracy']*100:.1f}%")

            st.markdown("---")
            cm = np.array(met["tool_router_classifier"]["confusion_matrix"])
            labels = met["tool_router_classifier"]["classes"]
            fig = px.imshow(cm, text_auto=True, x=labels, y=labels, color_continuous_scale="Greens",
                             labels=dict(x="Təxmin edilən", y="Həqiqi", color="Say"),
                             title="Confusion Matrix — Tool Routing")
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("🤖 Canlı LLM Agent Simulyasiyası")
        if not all(k in models for k in ["tfidf", "intent_clf", "tool_clf"]):
            st.info("Modellər yüklənməyib.")
        else:
            example_messages = [
                "where is my order, it hasn't arrived yet",
                "I was charged twice, I want to file a complaint",
                "what is your policy on refunds",
                "can u cancel my order plz",
                "I need to talk to a real human right now",
                "how do I reset my password",
            ]
            chosen = st.selectbox("Nümunə mesaj seçin (və ya aşağıda özünüz yazın):", ["— seçin —"] + example_messages)
            default_text = "" if chosen == "— seçin —" else chosen
            user_msg = st.text_area("Müştəri mesajı:", value=default_text, height=100)

            if st.button("🚀 Agent-i çağır") and user_msg.strip():
                vec = models["tfidf"].transform([user_msg])
                intent_pred = models["intent_clf"].predict(vec)[0]
                intent_proba = models["intent_clf"].predict_proba(vec)[0].max()
                tool_pred = models["tool_clf"].predict(vec)[0]
                tool_proba = models["tool_clf"].predict_proba(vec)[0]
                tool_classes = models["tool_clf"].classes_

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("##### 🎯 Təxmin edilən İntent")
                    st.success(f"**{intent_pred}**  ({intent_proba*100:.1f}% etibar)")
                    st.markdown("##### 🔧 Seçilmiş Agent Aləti")
                    st.info(f"**{TOOL_LABELS.get(tool_pred, tool_pred)}**")
                    if tool_pred == "check_order":
                        st.code(f'check_order(order_id="...", intent="{intent_pred}")\n'
                                f'  → sonra Olist modulundakı Risk Kalkulyatoru ilə\n'
                                f'    gecikmə/narazılıq riski qiymətləndirilir', language="python")
                    elif tool_pred == "create_ticket":
                        st.code(f'create_ticket(category="{intent_pred}", priority="normal")', language="python")
                    else:
                        st.code(f'search_faq(topic="{intent_pred}")', language="python")
                with col2:
                    proba_df = pd.DataFrame({"tool": tool_classes, "proba": tool_proba}).sort_values("proba")
                    fig = px.bar(proba_df, x="proba", y="tool", orientation="h", title="Alət seçimi ehtimalları")
                    fig.update_layout(xaxis_range=[0, 1])
                    st.plotly_chart(fig, use_container_width=True)

                if tool_proba.max() < 0.6:
                    st.warning("⚠️ Aşağı etibar balı — real sistemdə insan operatora yönləndirmə tövsiyə olunur.")

st.markdown("---")
st.caption("Data Science Final Layihəsi · LLM Agent — Olist (check_order) + Bitext (search_faq/create_ticket)")
