import streamlit as st
import pandas as pd
import joblib
import streamlit.components.v1 as components
import os
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc

# ─── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="Returns Fraud & Warehouse Analytics",
    page_icon="📦",
    layout="wide"
)

# ─── LOAD MODELS ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    model_proc  = joblib.load("processing_model.pkl")
    model_fraud = joblib.load("fraud_model.pkl")
    encoders    = joblib.load("encoders.pkl")
    
    # Check if target_encoder exists
    if os.path.exists("target_encoder.pkl"):
        target_encoder = joblib.load("target_encoder.pkl")
    else:
        target_encoder = None
        
    return model_proc, model_fraud, encoders, target_encoder

model_proc, model_fraud, encoders, target_encoder = load_models()

# ─── LOAD DASHBOARD HTML ───────────────────────────────────────
@st.cache_data
def load_html():
    if os.path.exists("ecommerce_returns_dashboard.html"):
        with open("ecommerce_returns_dashboard.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Dashboard HTML not found.</h1>"

html_content = load_html()

# ─── LOAD DATA FOR EVALUATION ──────────────────────────────────
@st.cache_data
def load_data():
    path = r"C:\Users\Admin\.gemini\antigravity\scratch\returns_fraud_project\returns_fraud_dataset.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

dataset = load_data()

# ─── TABS ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Prediction System", "📊 Analytics Dashboard", "📈 Model Performance"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — ML PREDICTION
# ══════════════════════════════════════════════════════════════
with tab1:

    st.title("📦 E-Commerce Returns: Fraud & Processing Predictor")
    st.write("Enter the details of a return request below.")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Return Details")

        product_category = st.selectbox(
            "Product Category",
            options=list(encoders["product_category_name"].classes_) if "product_category_name" in encoders else ["Unknown"],
            help="Type of product being returned"
        )

        return_reason = st.selectbox(
            "Return Reason",
            options=list(encoders["return_reason"].classes_) if "return_reason" in encoders else ["Unknown"],
            help="Reason stated by the customer for the return"
        )

        inspection_level = st.selectbox(
            "Inspection Level",
            options=list(encoders["inspection_level"].classes_) if "inspection_level" in encoders else ["Unknown"],
            help="Level of inspection assigned at the warehouse"
        )

        warehouse_load = st.selectbox(
            "Warehouse Load",
            options=list(encoders["warehouse_load"].classes_) if "warehouse_load" in encoders else ["Unknown"],
            help="Current operational load at the warehouse"
        )

    with col2:
        st.subheader("Prediction Results")

        if st.button("🔍 Predict", use_container_width=True, type="primary"):
            cat_enc  = encoders["product_category_name"].transform([product_category])[0] if "product_category_name" in encoders else 0
            reas_enc = encoders["return_reason"].transform([return_reason])[0] if "return_reason" in encoders else 0
            insp_enc = encoders["inspection_level"].transform([inspection_level])[0] if "inspection_level" in encoders else 0
            load_enc = encoders["warehouse_load"].transform([warehouse_load])[0] if "warehouse_load" in encoders else 0

            # ── Processing Category Prediction ──
            proc_input = pd.DataFrame(
                [[cat_enc, reas_enc, load_enc]],
                columns=["product_category_name", "return_reason", "warehouse_load"]
            )
            proc_pred_encoded = model_proc.predict(proc_input)[0]
            if target_encoder and hasattr(target_encoder, "inverse_transform"):
                proc_pred_label = target_encoder.inverse_transform([proc_pred_encoded])[0]
            else:
                proc_pred_label = "Low" if proc_pred_encoded == 0 else "Medium" if proc_pred_encoded == 1 else "High"
                
            # ── Fraud Risk Prediction ──
            fraud_input = pd.DataFrame(
                [[cat_enc, reas_enc, insp_enc, load_enc]],
                columns=["product_category_name", "return_reason", "inspection_level", "warehouse_load"]
            )
            fraud_prob = model_fraud.predict_proba(fraud_input)[0][1]

            proc_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
            proc_avg   = {"Low": "< 20 min", "Medium": "20–50 min", "High": "> 50 min"}
            st.metric(
                label="📊 Processing Category",
                value=f"{proc_color.get(proc_pred_label, '')} {proc_pred_label}",
                delta=proc_avg.get(proc_pred_label, "")
            )

            st.metric(label="🎯 Fraud Probability Score", value=f"{fraud_prob:.1%}")

            if fraud_prob >= 0.55:
                st.error(f"⚠️ **HIGH FRAUD RISK** ({fraud_prob:.1%})  \nRecommend: Intensive inspection before accepting this return.")
            elif fraud_prob >= 0.30:
                st.warning(f"⚡ **MEDIUM FRAUD RISK** ({fraud_prob:.1%})  \nRecommend: Manual inspection — verify product condition.")
            else:
                st.success(f"✅ **LOW FRAUD RISK** ({fraud_prob:.1%})  \nRecommend: Basic inspection — likely a genuine return.")
            st.divider()

        else:
            st.info("👈 Fill in the return details on the left and click **Predict**.")

# ══════════════════════════════════════════════════════════════
# TAB 2 — ANALYTICS DASHBOARD
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 📊 Full Analytics Dashboard")
    components.html(html_content, height=1100, scrolling=True)

# ══════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 📈 Fraud Model Diagnostic Metrics")
    st.write("This tab displays evaluation metrics for the Fraud Prediction model on a holdout or reference dataset.")

    if not dataset.empty:
        try:
            eval_df = dataset.copy()
            # Clean data mappings for prediction
            eval_df["product_category_name"] = eval_df["Product_Category"]
            eval_df["return_reason"] = eval_df["Return_Reason_Claimed"]
            eval_df["inspection_level"] = eval_df["Inspection_Level"]
            eval_df["warehouse_load"] = eval_df["Warehouse_Load"].apply(lambda x: "High" if x > 1000 else "Medium" if x > 500 else "Low")
            
            # Map Fraud Ground Truth
            eval_df["True_Fraud"] = eval_df["Fraud_Flag_Expert"].apply(lambda x: 1 if str(x).strip().lower() == "yes" else 0)

            # Drop missing values
            eval_df = eval_df.dropna(subset=["product_category_name", "return_reason", "inspection_level", "warehouse_load", "True_Fraud"])
            
            # Filter to labels that models have seen
            seen_pc = set(encoders["product_category_name"].classes_)
            seen_rr = set(encoders["return_reason"].classes_)
            seen_il = set(encoders["inspection_level"].classes_)
            seen_wl = set(encoders["warehouse_load"].classes_)
            
            eval_df = eval_df[
                eval_df["product_category_name"].isin(seen_pc) &
                eval_df["return_reason"].isin(seen_rr) &
                eval_df["inspection_level"].isin(seen_il) &
                eval_df["warehouse_load"].isin(seen_wl)
            ].sample(n=min(len(eval_df), 1500), random_state=42)

            X_cat = eval_df["product_category_name"].apply(lambda x: encoders["product_category_name"].transform([x])[0])
            X_reas = eval_df["return_reason"].apply(lambda x: encoders["return_reason"].transform([x])[0])
            X_insp = eval_df["inspection_level"].apply(lambda x: encoders["inspection_level"].transform([x])[0])
            X_load = eval_df["warehouse_load"].apply(lambda x: encoders["warehouse_load"].transform([x])[0])

            fraud_input = pd.DataFrame({
                "product_category_name": X_cat,
                "return_reason": X_reas,
                "inspection_level": X_insp,
                "warehouse_load": X_load
            })

            preds = model_fraud.predict(fraud_input)
            pred_probs = model_fraud.predict_proba(fraud_input)[:, 1]

            y_true = eval_df["True_Fraud"].values
            
            # Append predictions back to the eval dataframe
            eval_df["Predicted_Fraud_Prob"] = pred_probs
            eval_df["Predicted_Fraud_Class"] = preds

            # Layout metrics
            col1, col2, col3 = st.columns(3)
            
            f1 = f1_score(y_true, preds)
            col1.metric("⚡ F1 Score (Fraud)", f"{f1:.3f}")
            
            fpr, tpr, thresholds = roc_curve(y_true, pred_probs)
            roc_auc = auc(fpr, tpr)
            col2.metric("📈 AUC-ROC Score", f"{roc_auc:.3f}")
            
            acc = (preds == y_true).mean()
            col3.metric("🎯 Accuracy", f"{acc:.1%}")
            
            st.divider()
            
            col_left, col_right = st.columns(2)
            with col_left:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_true, preds)
                cm_df = pd.DataFrame(cm, 
                                     index=["Actual Genuine (0)", "Actual Fraud (1)"], 
                                     columns=["Predicted Genuine (0)", "Predicted Fraud (1)"])
                st.dataframe(cm_df.style.background_gradient(cmap='Blues'), use_container_width=True)
                
            with col_right:
                st.subheader("ROC Curve")
                roc_df = pd.DataFrame({"False Positive Rate": fpr, "True Positive Rate": tpr})
                roc_df = roc_df.set_index("False Positive Rate")
                st.line_chart(roc_df, color="#ff4b4b")
                st.caption(f"Area Under Curve (AUC): {roc_auc:.3f}")

            st.divider()
            st.subheader("Sample Predictions vs Ground Truth")
            display_df = eval_df[["Return_ID", "Product_Category", "Return_Reason_Claimed", "Warehouse_Load", "Inspection_Level", "True_Fraud", "Predicted_Fraud_Prob", "Predicted_Fraud_Class"]].head(20)
            
            st.dataframe(display_df.style.format({"Predicted_Fraud_Prob": "{:.1%}"})
                            .background_gradient(subset=["Predicted_Fraud_Prob"], cmap="Reds"),
                         use_container_width=True, hide_index=True)
            
        except Exception as e:
            st.error(f"Cannot compute metrics with dataset. Details: {str(e)}")
    else:
        st.info("No test dataset found to compute metrics.")
