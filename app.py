

# DASHBOARD


# IMPORT LIBRARIES


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import joblib
import os

# PAGE CONFIGURATION


st.set_page_config(
    page_title="AI-Powered Credit Card Fraud Detection System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)


# CUSTOM CSS


st.markdown("""
<style>

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

.main {
    background: linear-gradient(
        to right,
        #0f172a,
        #111827
    );
}

.block-container {
    padding-top: 2rem;
}

h1, h2, h3 {
    color: white;
    font-weight: 700;
}

section[data-testid="stSidebar"] {
    background: #020617;
    border-right: 1px solid #1e293b;
}

[data-testid="metric-container"] {
    background: linear-gradient(
        135deg,
        #1e293b,
        #334155
    );
    border: 1px solid #475569;
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
}

[data-testid="metric-container"] label {
    color: white !important;
}

.stButton>button {
    background: linear-gradient(
        135deg,
        #2563eb,
        #7c3aed
    );
    color: white;
    border-radius: 12px;
    border: none;
    padding: 12px 25px;
    font-size: 16px;
    font-weight: bold;
    width: 100%;
}

.stButton>button:hover {
    background: linear-gradient(
        135deg,
        #1d4ed8,
        #6d28d9
    );
    color: white;
}

div[data-baseweb="input"] {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# HEADER


st.title("💳 AI-Powered Credit Card Fraud Detection Dashboard")

st.markdown("""
### 🚀 Real-Time Credit Card Fraud Detection System
""")

st.markdown("---")

# FILE UPLOADER


uploaded_file = st.file_uploader(
    "📂 Upload Credit Card CSV File",
    type=["csv"]
)

# CHECK FILE


if uploaded_file is None:

    st.warning("⚠️ Please Upload creditcard.csv")

    st.stop()

# LOAD DATA

@st.cache_data
def load_data(file):

    dataframe = pd.read_csv(file)


    return dataframe

df = load_data(uploaded_file)


# LOAD MODEL


model = None

if os.path.exists("model.pkl"):

    model = joblib.load("model.pkl")


# SIDEBAR


st.sidebar.title("⚡ Navigation")

menu = st.sidebar.radio(
    "Select Dashboard",
    [
        "🏠 Dashboard",
        "📊 Dataset",
        "📈 Visual Analysis"
    ]
)

st.sidebar.markdown("---")



# DASHBOARD


if menu == "🏠 Dashboard":

    total_transactions = len(df)

    fraud_transactions = len(
        df[df["Class"] == 1]
    )

    normal_transactions = len(
        df[df["Class"] == 0]
    )

    fraud_percentage = (
        fraud_transactions / total_transactions
    ) * 100

    # METRICS
    

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "💰 Total Transactions",
        f"{total_transactions:,}"
    )

    col2.metric(
        "🚨 Fraud Cases",
        fraud_transactions
    )

    col3.metric(
        "✅ Legitimate",
        normal_transactions
    )

    col4.metric(
        "📉 Fraud %",
        f"{fraud_percentage:.4f}%"
    )

    st.markdown("---")

   
    # CHARTS
  

    col1, col2 = st.columns(2)

    with col1:

        pie_chart = px.pie(
            names=["Normal", "Fraud"],
            values=[
                normal_transactions,
                fraud_transactions
            ],
            hole=0.5,
            title="Fraud Distribution"
        )

        pie_chart.update_layout(
            template="plotly_dark"
        )

        st.plotly_chart(
            pie_chart,
            use_container_width=True
        )

    with col2:

        histogram = px.histogram(
            df,
            x="Amount",
            nbins=60,
            title="Transaction Amount Distribution"
        )

        histogram.update_layout(
            template="plotly_dark"
        )

        st.plotly_chart(
            histogram,
            use_container_width=True
        )


# DATASET


elif menu == "📊 Dataset":

    st.header("📊 Dataset Overview")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Dataset Shape")

        st.success(df.shape)

    with col2:

        st.subheader("Missing Values")

        st.success(
            df.isnull().sum().sum()
        )

    st.markdown("---")

    st.subheader("📁 Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("📈 Statistical Summary")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )


# VISUAL ANALYSIS

elif menu == "📈 Visual Analysis":

    st.header("📈 Fraud Analysis")

    st.markdown("---")

    # CORRELATION HEATMAP


    st.subheader("🔥 Correlation Heatmap")

    correlation = df.corr(
        numeric_only=True
    )

    heatmap = go.Figure(

        data=go.Heatmap(

            z=correlation.values,

            x=correlation.columns,

            y=correlation.columns,

            colorscale="Viridis"
        )
    )

    heatmap.update_layout(
        template="plotly_dark",
        height=700
    )

    st.plotly_chart(
        heatmap,
        use_container_width=True
    )

    st.markdown("---")

    
    # BOXPLOTS
   

    col1, col2 = st.columns(2)

    fraud_df = df[df["Class"] == 1]

    normal_df = df[df["Class"] == 0]

    with col1:

        fraud_box = px.box(
            fraud_df,
            y="Amount",
            title="Fraud Transaction Amount"
        )

        fraud_box.update_layout(
            template="plotly_dark"
        )

        st.plotly_chart(
            fraud_box,
            use_container_width=True
        )

    with col2:

        normal_box = px.box(
            normal_df,
            y="Amount",
            title="Normal Transaction Amount"
        )

        normal_box.update_layout(
            template="plotly_dark"
        )

        st.plotly_chart(
            normal_box,
            use_container_width=True
        )


# FOOTER

st.markdown("""
<!-- Load Font Awesome Icon Library -->
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">

<div class="footer" style="text-align: center; margin-top: 2rem; font-family: sans-serif;">
    Developedms❤️ by <span style="color: #38bdf8; font-weight: 600;">Shibam</span> | 
    <a href="https://github.com/myselfshibam" 
       target="_blank" 
       style="color: #38bdf8; text-decoration: none; font-weight: 600; margin-left: 5px;">
        <i class="fa-brands fa-github" style="margin-right: 5px;"></i>
    </a>  
    <a href="https://linkedin.com/in/shibammitra89" 
       target="_blank" 
       style="color: #38bdf8; text-decoration: none; font-weight: 600;">
        <i class="fa-brands fa-linkedin"></i>
    </a>
</div>
""", unsafe_allow_html=True)