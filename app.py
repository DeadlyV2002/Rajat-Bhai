
import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
import re

st.set_page_config(layout="wide", page_title="Universal Bank - Personal Loan Propensity", initial_sidebar_state="expanded")

st.title("Universal Bank — Personal Loan Propensity Dashboard")
st.markdown("This Streamlit dashboard trains Decision Tree, Random Forest and Gradient Boosting models to predict whether a customer will accept a personal loan. Upload your data or use the sample dataset.")

def normalize_cols(df):
    df = df.copy()
    df.columns = [re.sub(r'[^0-9a-zA-Z]', '', c).lower() for c in df.columns]
    return df

def preprocess(df):
    df = normalize_cols(df)
    target = None
    for c in df.columns:
        if 'personalloan' in c or (('personal' in c) and ('loan' in c)):
            target = c
            break
    if target is None:
        raise ValueError("Could not find target column 'Personal Loan' in uploaded data. Include a column named 'Personal Loan' or similar.")
    drop_cols = [c for c in df.columns if c=='id' or 'zip' in c]
    df = df.drop(columns=drop_cols, errors='ignore')
    for c in df.columns:
        if c==target: continue
        try:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        except:
            pass
    df = df.dropna(subset=[target])
    return df, target

def train_models(X_train, y_train):
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42)
    }
    trained = {}
    cv = 5
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = []
    for name, m in models.items():
        m.fit(X_train, y_train)
        trained[name] = m
        cv_scores = cross_val_score(m, X_train, y_train, cv=skf, scoring='accuracy')
        results.append((name, m, cv_scores.mean()))
    return trained, results

def eval_models(trained, X_train, y_train, X_test, y_test):
    rows = []
    for name, model in trained.items():
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_test_proba = model.predict_proba(X_test)[:,1]
        else:
            y_test_proba = model.decision_function(X_test)
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred, zero_division=0)
        recall = recall_score(y_test, y_test_pred, zero_division=0)
        f1 = f1_score(y_test, y_test_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_test_proba)
        rows.append({
            'Algorithm': name,
            'Training Accuracy': train_acc,
            'Testing Accuracy': test_acc,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'AUC': auc
        })
    return pd.DataFrame(rows).set_index('Algorithm')

st.sidebar.header("Data & Options")
use_sample = st.sidebar.checkbox("Use sample dataset (UniversalBank.csv)", value=True)
uploaded_file = st.sidebar.file_uploader("Or upload CSV file", type=['csv'])

if use_sample:
    df = pd.read_csv("UniversalBank.csv")
elif uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.sidebar.warning("Select sample dataset or upload a CSV to begin.")
    st.stop()

try:
    df_proc, target_col = preprocess(df)
except Exception as e:
    st.error(f"Error processing dataset: {e}")
    st.stop()

st.sidebar.markdown(f"**Detected target column:** `{target_col}`")
st.sidebar.write("Columns found:")
st.sidebar.write(list(df_proc.columns))

tabs = st.tabs(["Overview & Insights", "Train & Evaluate Models", "Predict New Data", "Download Sample"])

with tabs[0]:
    st.header("Customer Insights — 5 Strategic Charts")
    st.markdown("Actionable, marketing-focused visualizations for conversion improvements. Each chart includes a short insight.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Acceptance rate — Education x Family (heatmap)")
        if 'education' in df_proc.columns and 'family' in df_proc.columns:
            pivot = df_proc.groupby(['education','family'])[target_col].mean().unstack(fill_value=0)
            fig = px.imshow(pivot.values, x=[str(c) for c in pivot.columns], y=[str(i) for i in pivot.index], labels=dict(x="Family size", y="Education", color="Acceptance Rate"))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("**Insight:** Target education-family segments with below-average acceptance rates for tailored offers.")
        else:
            st.info("Education or Family column missing for this chart.")

    with col2:
        st.subheader("Income vs CCAvg — propensity scatter")
        if 'income' in df_proc.columns and 'ccavg' in df_proc.columns:
            fig = px.scatter(df_proc, x='income', y='ccavg', color=target_col.astype(str), labels={'color':'Personal Loan'}, hover_data=df_proc.columns, marginal_x="histogram", marginal_y="histogram")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("**Insight:** Identify high-income low-CCAvg segments to cross-sell credit services to increase conversion.")
        else:
            st.info("Income or CCAvg missing.")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Acceptance rate by Income Decile (lift)")
        if 'income' in df_proc.columns:
            df_proc['income_decile'] = pd.qcut(df_proc['income'], 10, labels=False, duplicates='drop')
            decile = df_proc.groupby('income_decile')[target_col].mean().reset_index()
            decile['decile_label'] = decile['income_decile']+1
            fig = px.bar(decile, x='decile_label', y=target_col, labels={'decile_label':'Income Decile','personalLoan':'Acceptance Rate'})
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("**Insight:** Focus campaigns on deciles with high acceptance but low reach; personalize messaging per income band.")
        else:
            st.info("Income missing.")

    with col4:
        st.subheader("Acceptance rate by Age band")
        if 'age' in df_proc.columns:
            bins = [20,30,40,50,60,80]
            labels = ['20-29','30-39','40-49','50-59','60+']
            df_proc['age_band'] = pd.cut(df_proc['age'], bins=bins, labels=labels, right=False)
            age_rate = df_proc.groupby('age_band')[target_col].mean().reset_index()
            fig = px.line(age_rate, x='age_band', y=target_col, markers=True, labels={target_col:'Acceptance Rate', 'age_band':'Age Band'})
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("**Insight:** Tailor loan terms and messaging by age cohort; younger cohorts may respond better to digital-first offers.")
        else:
            st.info("Age missing.")

    st.markdown("---")
    st.subheader("Feature correlation with Personal Loan (absolute)")
    numeric_cols = df_proc.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    corrs = df_proc[numeric_cols + [target_col]].corr()[target_col].drop(index=target_col).abs().sort_values(ascending=False)
    fig = px.bar(x=corrs.index, y=corrs.values, labels={'x':'Feature', 'y':'Absolute correlation with Personal Loan'})
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Insight:** Use highest-correlation features for prioritization in campaigns and for feature engineering.")

with tabs[1]:
    st.header("Train Models & View Performance")
    st.markdown("Click **Start Training** to train all three models (Decision Tree, Random Forest, Gradient Boosting) on the dataset and compute metrics, ROC, confusion matrices and feature importances.")

    colA, colB = st.columns([1,3])
    with colA:
        run_train = st.button("Start Training & Evaluation")
        test_size = st.slider("Test set size (%)", min_value=10, max_value=50, value=30, step=5)
    with colB:
        st.write("Cross-validation: Stratified 5-fold on training set. Models use reasonable default hyperparameters for a quick run.")

    if run_train:
        with st.spinner("Training models..."):
            X = df_proc.drop(columns=[target_col])
            y = df_proc[target_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=42, stratify=y)
            trained, cv_results = train_models(X_train, y_train)
            metrics_df = eval_models(trained, X_train, y_train, X_test, y_test)
            cv_map = {name:cv for (name,_,cv) in cv_results}
            metrics_df['CV Accuracy (5-fold, train)'] = metrics_df.index.map(lambda n: cv_map.get(n, np.nan))
            st.subheader("Metrics table")
            st.dataframe(metrics_df.style.format("{:.4f}"))
            st.download_button("Download metrics as CSV", metrics_df.to_csv().encode('utf-8'), file_name="model_metrics.csv")

            st.subheader("ROC Curves (all models)")
            fig = go.Figure()
            for name, model in trained.items():
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_test)[:,1]
                else:
                    probs = model.decision_function(X_test)
                fpr, tpr, _ = roc_curve(y_test, probs)
                auc = roc_auc_score(y_test, probs)
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{name} (AUC {auc:.3f})"))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(dash='dash')))
            fig.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', width=900, height=600)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Confusion Matrices and Feature Importances")
            for name, model in trained.items():
                st.markdown(f"**{name}**")
                cols = st.columns(2)
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                cm_train = confusion_matrix(y_train, y_train_pred)
                cm_test = confusion_matrix(y_test, y_test_pred)
                fig1 = px.imshow(cm_train, text_auto=True, labels=dict(x="Predicted", y="Actual"), title="Train Confusion Matrix")
                fig2 = px.imshow(cm_test, text_auto=True, labels=dict(x="Predicted", y="Actual"), title="Test Confusion Matrix")
                cols[0].plotly_chart(fig1, use_container_width=True)
                cols[1].plotly_chart(fig2, use_container_width=True)
                if hasattr(model, 'feature_importances_'):
                    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
                    figf = px.bar(x=importances.index, y=importances.values, labels={'x':'Feature','y':'Importance'}, title="Feature importances")
                    st.plotly_chart(figf, use_container_width=True)
            st.success("Training & evaluation completed.")

with tabs[2]:
    st.header("Upload New Data & Predict Personal Loan")
    st.markdown("Upload a CSV with the same columns (ID optional). The model below will train on the existing dataset and predict the `Personal Loan` label for the uploaded records. You can then download the predicted file.")

    uploaded_new = st.file_uploader("Upload CSV to predict", type=['csv'], key="predict")
    classifier_choice = st.selectbox("Model to use for prediction", ("Random Forest","Gradient Boosting","Decision Tree"))
    predict_btn = st.button("Train on current dataset and predict uploaded data")

    if predict_btn:
        if uploaded_new is None:
            st.warning("Please upload a CSV to predict.")
        else:
            X_full = df_proc.drop(columns=[target_col])
            y_full = df_proc[target_col]
            trained, _ = train_models(X_full, y_full)
            model = trained.get(classifier_choice)
            new_df = pd.read_csv(uploaded_new)
            new_proc, _ = preprocess(new_df)
            missing = [c for c in X_full.columns if c not in new_proc.columns]
            if missing:
                st.warning(f"The uploaded file is missing columns required for prediction: {missing}. Fill these columns and retry.")
            else:
                X_new = new_proc[X_full.columns]
                preds = model.predict(X_new)
                proba = model.predict_proba(X_new)[:,1] if hasattr(model, "predict_proba") else None
                new_proc['pred_personal_loan'] = preds
                if proba is not None:
                    new_proc['pred_prob_personal_loan'] = proba
                st.write("Preview of predictions:")
                st.dataframe(new_proc.head(20))
                csv = new_proc.to_csv(index=False).encode('utf-8')
                st.download_button("Download predictions CSV", csv, file_name="predictions.csv")

with tabs[3]:
    st.header("Download Sample Dataset")
    with open("UniversalBank.csv","rb") as f:
        st.download_button("Download sample UniversalBank.csv", f, file_name="UniversalBank.csv")
    st.markdown("You can upload this file in the 'Predict New Data' tab to test the prediction flow.")

st.markdown("---")
st.markdown("Built for Marketing Heads: the charts and model outputs are designed to give actionable segmentation, income/age targeting, and model-based prioritization for sales outreach.")
