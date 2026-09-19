"""
LLM Customer Support Agent — Combined Model Training Pipeline
=================================================================
Two real, independent data sources power the 3 agent tools:

  OLIST (99k real Brazilian e-commerce orders) -> powers `check_order`
     D) Late-delivery risk classifier   (pre-delivery features only)
     E) Review/satisfaction risk classifier (post-delivery features)

  BITEXT (26.9k real customer-support utterances) -> powers `search_faq`
  and `create_ticket`, plus the overall tool-routing "brain"
     A) Intent classifier      (27-class)
     B) Category classifier    (11-class)
     C) Tool-routing classifier (3-class: check_order/create_ticket/search_faq)

All artifacts are saved to ./models/ as .pkl (joblib) files so the
Streamlit dashboard can load them instantly without retraining.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import time

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score
)

BASE = "/home/claude/project3"
OLIST_DIR = f"{BASE}/data/olist"
BITEXT_DIR = f"{BASE}/data/bitext"
MODELS_DIR = f"{BASE}/models"
DATA_DIR = f"{BASE}/data"
os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_STATE = 42
metrics_report = {}

# =======================================================================
# PART 1 — OLIST: check_order backend
# =======================================================================
print("=" * 70)
print("PART 1 — OLIST (real e-commerce orders) -> check_order backend")
print("=" * 70)

orders = pd.read_csv(f"{OLIST_DIR}/olist_orders_dataset.csv", parse_dates=[
    "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
    "order_delivered_customer_date", "order_estimated_delivery_date"])
items = pd.read_csv(f"{OLIST_DIR}/olist_order_items_dataset.csv")
payments = pd.read_csv(f"{OLIST_DIR}/olist_order_payments_dataset.csv")
reviews = pd.read_csv(f"{OLIST_DIR}/olist_order_reviews_dataset.csv")
customers = pd.read_csv(f"{OLIST_DIR}/olist_customers_dataset.csv")
products = pd.read_csv(f"{OLIST_DIR}/olist_products_dataset.csv")
sellers = pd.read_csv(f"{OLIST_DIR}/olist_sellers_dataset.csv")
cat_trans = pd.read_csv(f"{OLIST_DIR}/product_category_name_translation.csv")

item_agg = items.groupby("order_id").agg(
    n_items=("order_item_id", "count"),
    total_price=("price", "sum"),
    total_freight=("freight_value", "sum"),
    avg_price=("price", "mean"),
).reset_index()

first_item = items.sort_values("order_item_id").groupby("order_id").first().reset_index()
first_item = first_item.merge(products[["product_id", "product_category_name"]], on="product_id", how="left")
first_item = first_item.merge(cat_trans, on="product_category_name", how="left")
first_item = first_item.merge(sellers[["seller_id", "seller_state"]], on="seller_id", how="left")

pay_agg = payments.groupby("order_id").agg(
    payment_installments=("payment_installments", "max"),
    payment_type=("payment_type", lambda x: x.mode()[0] if len(x) > 0 else "unknown"),
).reset_index()

olist = orders.merge(customers[["customer_id", "customer_state", "customer_city"]], on="customer_id", how="left")
olist = olist.merge(item_agg, on="order_id", how="left")
olist = olist.merge(first_item[["order_id", "product_category_name_english", "seller_state"]], on="order_id", how="left")
olist = olist.merge(pay_agg, on="order_id", how="left")
olist = olist.merge(reviews[["order_id", "review_score"]], on="order_id", how="left")

olist = olist[olist["order_status"] == "delivered"].dropna(
    subset=["order_delivered_customer_date", "review_score", "product_category_name_english"]
)
olist["delivery_delay_days"] = (olist["order_delivered_customer_date"] - olist["order_estimated_delivery_date"]).dt.days
olist["is_late"] = (olist["delivery_delay_days"] > 0).astype(int)
olist["purchase_to_delivery_days"] = (olist["order_delivered_customer_date"] - olist["order_purchase_timestamp"]).dt.days
olist["approval_delay_hours"] = (olist["order_approved_at"] - olist["order_purchase_timestamp"]).dt.total_seconds() / 3600
olist["cross_state"] = (olist["customer_state"] != olist["seller_state"]).astype(int)
olist["purchase_month"] = olist["order_purchase_timestamp"].dt.month
olist["purchase_dow"] = olist["order_purchase_timestamp"].dt.dayofweek
olist["is_bad_review"] = (olist["review_score"] <= 2).astype(int)
olist["product_category_name_english"] = olist["product_category_name_english"].fillna("unknown")
olist["payment_type"] = olist["payment_type"].fillna("unknown")
olist["seller_state"] = olist["seller_state"].fillna("unknown")

print("Cleaned Olist shape:", olist.shape)

dash_cols_olist = [
    "order_id", "order_status", "order_purchase_timestamp", "customer_state",
    "customer_city", "seller_state", "product_category_name_english",
    "payment_type", "payment_installments", "n_items", "total_price",
    "total_freight", "avg_price", "delivery_delay_days", "is_late",
    "purchase_to_delivery_days", "approval_delay_hours", "cross_state",
    "purchase_month", "purchase_dow", "review_score", "is_bad_review",
]
olist[dash_cols_olist].to_parquet(f"{DATA_DIR}/olist_clean.parquet", index=False)
print("Saved cleaned Olist dashboard dataset.")

# ---------------------------------------------------------------------
# TASK D — Late delivery risk (PRE-delivery features only — no leakage)
# ---------------------------------------------------------------------
print("\n=== TASK D: Late-delivery risk classifier (Random Forest) ===")
t0 = time.time()

pre_delivery_num = ["n_items", "total_price", "total_freight", "avg_price",
                     "payment_installments", "approval_delay_hours",
                     "cross_state", "purchase_month", "purchase_dow"]
pre_delivery_cat = ["customer_state", "seller_state", "product_category_name_english", "payment_type"]
pre_delivery_features = pre_delivery_num + pre_delivery_cat

olist_model_df = olist.dropna(subset=pre_delivery_features + ["is_late"]).copy()
X_late = olist_model_df[pre_delivery_features].copy()
y_late = olist_model_df["is_late"]

late_encoders = {}
for c in pre_delivery_cat:
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_late[c] = oe.fit_transform(X_late[[c]])
    late_encoders[c] = oe

Xtr, Xte, ytr, yte = train_test_split(X_late, y_late, test_size=0.2, random_state=RANDOM_STATE, stratify=y_late)

clf_late = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=10,
                                   n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced")
clf_late.fit(Xtr, ytr)
pred_late = clf_late.predict(Xte)
proba_late = clf_late.predict_proba(Xte)[:, 1]

acc_late = accuracy_score(yte, pred_late)
f1_late = f1_score(yte, pred_late)
auc_late = roc_auc_score(yte, proba_late)
print(f"Accuracy: {acc_late:.4f} | F1: {f1_late:.4f} | ROC-AUC: {auc_late:.4f} | time: {time.time()-t0:.1f}s")
print(f"Baseline (majority class): {max(y_late.mean(), 1-y_late.mean()):.4f}")

joblib.dump(clf_late, f"{MODELS_DIR}/olist_late_delivery_classifier.pkl")
joblib.dump(late_encoders, f"{MODELS_DIR}/olist_late_encoders.pkl")
joblib.dump(pre_delivery_features, f"{MODELS_DIR}/olist_late_feature_order.pkl")

late_importance = sorted(
    zip(pre_delivery_features, clf_late.feature_importances_.tolist()), key=lambda x: -x[1]
)[:10]

metrics_report["late_delivery_classifier"] = {
    "accuracy": round(acc_late, 4),
    "f1": round(f1_late, 4),
    "roc_auc": round(auc_late, 4),
    "baseline": round(max(y_late.mean(), 1 - y_late.mean()), 4),
    "top_features": [{"feature": f, "importance": round(v, 4)} for f, v in late_importance],
}

# ---------------------------------------------------------------------
# TASK E — Bad-review risk (post-delivery features, incl. delay)
# ---------------------------------------------------------------------
print("\n=== TASK E: Bad-review risk classifier (Random Forest) ===")
t0 = time.time()

post_delivery_num = pre_delivery_num + ["delivery_delay_days", "purchase_to_delivery_days"]
post_delivery_features = post_delivery_num + pre_delivery_cat

X_rev = olist_model_df[post_delivery_features].copy()
y_rev = olist_model_df["is_bad_review"]

rev_encoders = {}
for c in pre_delivery_cat:
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_rev[c] = oe.fit_transform(X_rev[[c]])
    rev_encoders[c] = oe

Xtr, Xte, ytr, yte = train_test_split(X_rev, y_rev, test_size=0.2, random_state=RANDOM_STATE, stratify=y_rev)

clf_rev = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=10,
                                  n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced")
clf_rev.fit(Xtr, ytr)
pred_rev = clf_rev.predict(Xte)
proba_rev = clf_rev.predict_proba(Xte)[:, 1]

acc_rev = accuracy_score(yte, pred_rev)
f1_rev = f1_score(yte, pred_rev)
auc_rev = roc_auc_score(yte, proba_rev)
print(f"Accuracy: {acc_rev:.4f} | F1: {f1_rev:.4f} | ROC-AUC: {auc_rev:.4f} | time: {time.time()-t0:.1f}s")
print(f"Baseline (majority class): {max(y_rev.mean(), 1-y_rev.mean()):.4f}")

joblib.dump(clf_rev, f"{MODELS_DIR}/olist_bad_review_classifier.pkl")
joblib.dump(rev_encoders, f"{MODELS_DIR}/olist_review_encoders.pkl")
joblib.dump(post_delivery_features, f"{MODELS_DIR}/olist_review_feature_order.pkl")

rev_importance = sorted(
    zip(post_delivery_features, clf_rev.feature_importances_.tolist()), key=lambda x: -x[1]
)[:10]

metrics_report["bad_review_classifier"] = {
    "accuracy": round(acc_rev, 4),
    "f1": round(f1_rev, 4),
    "roc_auc": round(auc_rev, 4),
    "baseline": round(max(y_rev.mean(), 1 - y_rev.mean()), 4),
    "top_features": [{"feature": f, "importance": round(v, 4)} for f, v in rev_importance],
}

metrics_report["olist_dataset_info"] = {
    "n_orders": int(olist_model_df.shape[0]),
    "late_rate": round(float(y_late.mean()), 4),
    "bad_review_rate": round(float(y_rev.mean()), 4),
    "source": "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce",
}

# =======================================================================
# PART 2 — BITEXT: search_faq / create_ticket backend + tool routing
# =======================================================================
print("\n" + "=" * 70)
print("PART 2 — BITEXT (real customer-support utterances) -> search_faq / create_ticket")
print("=" * 70)

bt = pd.read_csv(f"{BITEXT_DIR}/bitext_customer_support_raw.csv")
bt["instruction_clean"] = bt["instruction"].str.strip()
bt["instruction_len"] = bt["instruction_clean"].str.len()
bt["instruction_word_count"] = bt["instruction_clean"].str.split().str.len()

FLAG_MAP = {
    "M": "morphological_variation", "L": "semantic_variation",
    "B": "basic_syntax", "I": "interrogative", "C": "coordinated_syntax",
    "N": "negation", "P": "polite", "Q": "colloquial", "W": "offensive",
    "K": "keyword_mode", "E": "abbreviation", "Z": "typo_error",
}
for code, name in FLAG_MAP.items():
    bt[f"flag_{name}"] = bt["flags"].str.contains(code, regex=False)

TOOL_MAP = {
    "track_order": "check_order", "place_order": "check_order",
    "cancel_order": "check_order", "change_order": "check_order",
    "track_refund": "check_order",
    "complaint": "create_ticket", "contact_human_agent": "create_ticket",
    "payment_issue": "create_ticket", "registration_problems": "create_ticket",
    "delete_account": "create_ticket",
    "check_invoice": "search_faq", "get_invoice": "search_faq",
    "check_refund_policy": "search_faq", "get_refund": "search_faq",
    "check_payment_methods": "search_faq", "check_cancellation_fee": "search_faq",
    "delivery_options": "search_faq", "delivery_period": "search_faq",
    "create_account": "search_faq", "edit_account": "search_faq",
    "recover_password": "search_faq", "switch_account": "search_faq",
    "change_shipping_address": "search_faq", "set_up_shipping_address": "search_faq",
    "newsletter_subscription": "search_faq", "review": "search_faq",
    "contact_customer_service": "search_faq",
}
bt["agent_tool"] = bt["intent"].map(TOOL_MAP)
assert bt["agent_tool"].isnull().sum() == 0

dash_cols_bt = ["instruction", "category", "intent", "response", "flags", "agent_tool",
                "instruction_len", "instruction_word_count"] + [f"flag_{n}" for n in FLAG_MAP.values()]
bt[dash_cols_bt].to_parquet(f"{DATA_DIR}/bitext_clean.parquet", index=False)
print("Saved cleaned Bitext dashboard dataset.")

# --- Task A: Intent ---
print("\n=== TASK A: Intent classification ===")
X_text = bt["instruction_clean"].astype(str)
y_intent = bt["intent"].astype(str)
Xtr, Xte, ytr, yte = train_test_split(X_text, y_intent, test_size=0.2, random_state=RANDOM_STATE, stratify=y_intent)

tfidf = TfidfVectorizer(max_features=6000, ngram_range=(1, 2), min_df=2)
Xtr_vec = tfidf.fit_transform(Xtr)
Xte_vec = tfidf.transform(Xte)

clf_intent = LogisticRegression(max_iter=300, C=5.0)
clf_intent.fit(Xtr_vec, ytr)
pred_intent = clf_intent.predict(Xte_vec)
acc_intent = accuracy_score(yte, pred_intent)
f1_intent = f1_score(yte, pred_intent, average="macro")
print(f"Accuracy: {acc_intent:.4f} | Macro-F1: {f1_intent:.4f} | Baseline: {1/y_intent.nunique():.4f}")

joblib.dump(tfidf, f"{MODELS_DIR}/tfidf_vectorizer.pkl")
joblib.dump(clf_intent, f"{MODELS_DIR}/intent_classifier.pkl")

labels_intent = sorted(y_intent.unique())
metrics_report["intent_classifier"] = {
    "accuracy": round(acc_intent, 4), "macro_f1": round(f1_intent, 4),
    "baseline": round(1 / y_intent.nunique(), 4), "classes": labels_intent,
    "confusion_matrix": confusion_matrix(yte, pred_intent, labels=labels_intent).tolist(),
}

# --- Task B: Category ---
print("\n=== TASK B: Category classification ===")
y_cat = bt["category"].astype(str)
Xtr, Xte, ytr, yte = train_test_split(X_text, y_cat, test_size=0.2, random_state=RANDOM_STATE, stratify=y_cat)
Xtr_vec = tfidf.transform(Xtr)
Xte_vec = tfidf.transform(Xte)
clf_cat = LogisticRegression(max_iter=300, C=5.0)
clf_cat.fit(Xtr_vec, ytr)
pred_cat = clf_cat.predict(Xte_vec)
acc_cat = accuracy_score(yte, pred_cat)
f1_cat = f1_score(yte, pred_cat, average="macro")
print(f"Accuracy: {acc_cat:.4f} | Macro-F1: {f1_cat:.4f} | Baseline: {1/y_cat.nunique():.4f}")

joblib.dump(clf_cat, f"{MODELS_DIR}/category_classifier.pkl")
labels_cat = sorted(y_cat.unique())
metrics_report["category_classifier"] = {
    "accuracy": round(acc_cat, 4), "macro_f1": round(f1_cat, 4),
    "baseline": round(1 / y_cat.nunique(), 4), "classes": labels_cat,
    "confusion_matrix": confusion_matrix(yte, pred_cat, labels=labels_cat).tolist(),
}

# --- Task C: Tool routing ---
print("\n=== TASK C: Agent tool-routing classification ===")
y_tool = bt["agent_tool"].astype(str)
Xtr, Xte, ytr, yte = train_test_split(X_text, y_tool, test_size=0.2, random_state=RANDOM_STATE, stratify=y_tool)
Xtr_vec = tfidf.transform(Xtr)
Xte_vec = tfidf.transform(Xte)
clf_tool = LogisticRegression(max_iter=300, C=5.0, class_weight="balanced")
clf_tool.fit(Xtr_vec, ytr)
pred_tool = clf_tool.predict(Xte_vec)
acc_tool = accuracy_score(yte, pred_tool)
f1_tool = f1_score(yte, pred_tool, average="macro")

clf_nb = MultinomialNB()
clf_nb.fit(Xtr_vec, ytr)
acc_nb = accuracy_score(yte, clf_nb.predict(Xte_vec))
print(f"Accuracy: {acc_tool:.4f} | Macro-F1: {f1_tool:.4f} | Baseline: {1/y_tool.nunique():.4f} | NB: {acc_nb:.4f}")

joblib.dump(clf_tool, f"{MODELS_DIR}/tool_router_classifier.pkl")
labels_tool = sorted(y_tool.unique())
metrics_report["tool_router_classifier"] = {
    "accuracy": round(acc_tool, 4), "macro_f1": round(f1_tool, 4),
    "baseline": round(1 / y_tool.nunique(), 4), "naive_bayes_accuracy": round(acc_nb, 4),
    "classes": labels_tool,
    "confusion_matrix": confusion_matrix(yte, pred_tool, labels=labels_tool).tolist(),
}

metrics_report["bitext_dataset_info"] = {
    "n_rows": int(bt.shape[0]), "n_intents": int(bt["intent"].nunique()),
    "n_categories": int(bt["category"].nunique()),
    "tool_distribution": bt["agent_tool"].value_counts().to_dict(),
    "source": "https://github.com/bitext/customer-support-llm-chatbot-training-dataset",
}

# =======================================================================
# SAVE COMBINED METRICS
# =======================================================================
with open(f"{MODELS_DIR}/metrics_report.json", "w") as f:
    json.dump(metrics_report, f, indent=2)

print("\nAll models & artifacts saved to", MODELS_DIR)
print(json.dumps({k: v for k, v in metrics_report.items() if "dataset_info" in k}, indent=2, default=str))
