import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# ========================
# Tampilan Header & Sidebar
# ========================
st.set_page_config(page_title="Deteksi Fraud Kartu Kredit", layout="wide")

with st.sidebar:
    st.image("image.png", width=80)
    st.title("🚀 Menu")
    uploaded_file = st.file_uploader("📁 Upload Dataset CSV", type="csv")
    st.markdown("---")
    st.info("Gunakan dataset Credit Card Fraud Detection (Kaggle)")

st.title("📊 Aplikasi Klasifikasi Transaksi Kartu Kredit")
st.markdown("**Algoritma: Logistic Regression** &nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp; _Streamlit by [Nama Kamu]_")

# ========================
# Proses Data
# ========================
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    tab1, tab2, tab3 = st.tabs(["📄 Dataset", "⚙️ Training & Model", "📈 Visualisasi"])

    with tab1:
        st.subheader("Preview Dataset")
        st.dataframe(df.head(), use_container_width=True)
        st.markdown(f"**Total Data:** {df.shape[0]} baris | **Fitur:** {df.shape[1]} kolom")
        st.write("Distribusi Kelas:")
        st.bar_chart(df['Class'].value_counts())

    with tab2:
        st.subheader("Training Model")
        X = df.drop('Class', axis=1)
        y = df['Class']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("🎯 Akurasi", f"{acc*100:.2f}%")
        with col2:
            st.metric("📉 ROC AUC", f"{roc_auc:.2f}")

        st.text("Classification Report:")
        st.code(classification_report(y_test, y_pred), language='text')

    with tab3:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig1, ax1 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", xticklabels=["Normal", "Fraud"], yticklabels=["Normal", "Fraud"])
        st.pyplot(fig1)

        st.subheader("ROC Curve")
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        ax2.plot([0, 1], [0, 1], 'k--')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve')
        ax2.legend()
        st.pyplot(fig2)
else:
    st.warning("Silakan upload file dataset terlebih dahulu.")
