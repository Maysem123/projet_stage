# =============================================================================
# Fichier : src/pages/2_👥_Customer_Segmentation.py
# Rôle    : Page de l'application Streamlit pour la segmentation des clients.
# =============================================================================

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np

# --- Configuration de la page ---
st.set_page_config(
    page_title="Segmentation des Clients",
    page_icon="👥",
    layout="wide"
)

# --- Fonctions de mise en cache ---
@st.cache_data
def load_data(path):
    """Charge les données depuis un fichier CSV."""
    df = pd.read_csv(path)
    # Correction pour 'TotalCharges'
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    return df

@st.cache_resource
def create_preprocessor(df, numeric_features, categorical_features):
    """Crée un pipeline de prétraitement pour les données."""
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    return preprocessor

@st.cache_data
def get_optimal_k(data):
    """Calcule l'inertie pour trouver le K optimal (Elbow Method)."""
    inertia = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)
    return k_range, inertia

@st.cache_data
def perform_clustering(_df_processed, k=4):
    """Exécute K-Means et PCA pour la visualisation."""
    # K-Means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(_df_processed)

    # PCA
    pca = PCA(n_components=2, random_state=42)
    decomposed_data = pca.fit_transform(_df_processed)
    
    return clusters, decomposed_data

# --- Chargement et préparation des données ---
df = load_data('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Sélection des caractéristiques pour la segmentation
features_for_clustering = ['tenure', 'MonthlyCharges', 'Contract', 'PaymentMethod', 'InternetService']
numeric_features = ['tenure', 'MonthlyCharges']
categorical_features = ['Contract', 'PaymentMethod', 'InternetService']

df_clustering = df[features_for_clustering].copy()

# Création et application du pipeline de prétraitement
preprocessor = create_preprocessor(df_clustering, numeric_features, categorical_features)
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
df_processed = pipeline.fit_transform(df_clustering)

# --- Interface Utilisateur ---
st.title(" Segmentation des Clients par Comportement")

# --- Add a professional navbar similar to the main app
st.markdown("""
<div class="main-header">
    <h1>👥 Segmentation des Clients par Comportement</h1>
    <p>Utilisez des outils avancés pour analyser et segmenter les clients en groupes distincts, afin d'améliorer les stratégies de rétention.</p>
</div>
""", unsafe_allow_html=True)

# --- Section: Introduction ---
st.markdown(
    "Cette page utilise un algorithme de **K-Means** pour grouper les clients en segments distincts "
    "basés sur leur comportement contractuel et financier. Cela permet d'identifier des profils types."
)

# --- Apply the same CSS theme from the main app
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #0059b3;
        --secondary-color: #f0f7ff;
        --accent-color: #004080;
        --text-color: #2c3e50;
        --light-bg: #ffffff;
        --border-color: #e1ecf4;
        --page-bg: #fafcff;
    }

    /* Main page background */
    .stApp {
        background: linear-gradient(135deg, var(--page-bg) 0%, #f8fbff 100%);
    }

    .main .block-container {
        background: transparent;
        padding-top: 2rem;
    }

    /* Sidebar background */
    .css-1d391kg, .css-1cypcdb {
        background: linear-gradient(180deg, #f0f7ff 0%, #e8f4ff 100%);
    }

    .stApp > header {
        background: transparent;
    }

    .stApp [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f0f7ff 0%, #e8f4ff 100%);
    }

    /* Ensure main content has proper background */
    section[data-testid="stSidebar"] > div {
        background: transparent;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--accent-color) 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0, 89, 179, 0.2);
    }

    .main-header h1 {
        color: white !important;
        margin-bottom: 0.5rem;
        font-size: 2.5rem;
        font-weight: 600;
    }

    .main-header p {
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 1.1rem;
        margin: 0;
    }

    /* Card styling */
    .metric-card {
        background: var(--light-bg);
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid var(--border-color);
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0, 89, 179, 0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 89, 179, 0.15);
    }

    .info-card {
        background: var(--secondary-color);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid var(--primary-color);
        margin-bottom: 1rem;
        color: var(--text-color) !important;
    }

    .info-card p, .info-card li, .info-card strong {
        color: var(--text-color) !important;
    }

    .prediction-card {
        background: linear-gradient(135deg, var(--light-bg) 0%, var(--secondary-color) 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid var(--primary-color);
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0, 89, 179, 0.1);
    }

    .section-header {
        background: var(--secondary-color);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        border-left: 4px solid var(--primary-color);
        margin: 2rem 0 1rem 0;
    }

    .section-header h2 {
        color: var(--primary-color) !important;
        margin: 0;
        font-size: 1.5rem;
        font-weight: 600;
    }

    /* Sidebar styling */
    .css-1d391kg, .css-1cypcdb {
        background: linear-gradient(180deg, #f0f7ff 0%, #e8f4ff 100%) !important;
    }

    .css-1d391kg .css-1v0mbdj {
        background: transparent;
    }

    /* Button styling */
    .stButton > button {
        background-color: var(--primary-color);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: background-color 0.3s ease;
    }

    .stButton > button:hover {
        background-color: var(--accent-color);
        border: none;
        color: white;
    }

    /* Metric styling */
    .metric-container {
        text-align: center;
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: var(--primary-color);
        margin-bottom: 0.5rem;
    }

    .metric-label {
        font-size: 1rem;
        color: var(--text-color);
        font-weight: 500;
    }

    /* Status indicators */
    .status-high-risk {
        background: linear-gradient(135deg, #ff4757, #ff3742);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
        margin: 1rem 0;
    }

    .status-loyal {
        background: linear-gradient(135deg, #2ed573, #1dd1a1);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
        margin: 1rem 0;
    }

    /* Table styling */
    .dataframe {
        border: 1px solid var(--border-color);
        border-radius: 8px;
        overflow: hidden;
    }

    /* Explanation box */
    .explanation-box {
        background: var(--light-bg);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 89, 179, 0.05);
        color: var(--text-color) !important;
    }

    .explanation-box h4 {
        color: var(--primary-color);
        margin-bottom: 1rem;
    }

    .explanation-box p, .explanation-box li, .explanation-box ul, .explanation-box strong, .explanation-box em {
        color: var(--text-color) !important;
    }

    /* Additional text color fixes */
    div[style*="background: #f0f7ff"] p,
    div[style*="background: #f0f7ff"] h4,
    div[style*="background: #f0f7ff"] strong {
        color: var(--text-color) !important;
    }

    /* Ensure all paragraph text is visible */
    .stMarkdown p {
        color: var(--text-color) !important;
    }

    /* Fix for dataframe display text */
    .stDataFrame {
        color: var(--text-color) !important;
    }
</style>
""", unsafe_allow_html=True)

# Apply custom CSS for tables
st.markdown("""
<style>
    .stDataFrame {
        border: 1px solid var(--border-color);
        border-radius: 8px;
        overflow: hidden;
        color: var(--text-color) !important;
        background-color: var(--light-bg);
    }
</style>
""", unsafe_allow_html=True)

# --- Section 1: Méthodologie (Elbow Method) ---
with st.expander("🔍 Comment trouver le nombre optimal de segments ? (Méthode du Coude)"):
    st.markdown(
        "Pour déterminer le nombre de segments (K) le plus pertinent, nous utilisons la méthode du coude. "
        "Nous calculons l'inertie du modèle pour différents K. Le 'coude' sur le graphique représente "
        "le meilleur compromis entre le nombre de clusters et la variance au sein de chaque cluster."
    )
    k_range, inertia = get_optimal_k(df_processed)
    
    fig_elbow = px.line(x=k_range, y=inertia, title='Méthode du Coude pour K Optimal', markers=True)
    fig_elbow.update_layout(
        xaxis_title="Nombre de Clusters (K)",
        yaxis_title="Inertie",
        template="plotly_white"
    )
    st.plotly_chart(fig_elbow, use_container_width=True, key="elbow_chart")
    st.info("Le graphique montre un 'coude' autour de K=4, ce qui suggère que 4 est un bon nombre de segments pour ces données.")

# --- Section 2: Visualisation des Segments ---
st.header("Visualisation des Segments de Clients")

clusters, decomposed_data = perform_clustering(df_processed, k=4)
df['Segment'] = clusters
df['PC1'] = decomposed_data[:, 0]
df['PC2'] = decomposed_data[:, 1]

fig_clusters = px.scatter(
    df,
    x='PC1',
    y='PC2',
    color='Segment',
    title="Segments de Clients (visualisés avec PCA)",
    hover_data=['customerID', 'tenure', 'MonthlyCharges', 'Contract', 'Churn']
)
fig_clusters.update_layout(
    xaxis_title="Composante Principale 1",
    yaxis_title="Composante Principale 2",
    legend_title="Segment",
    template="plotly_white"
)
st.plotly_chart(fig_clusters, use_container_width=True, key="pca_chart")

# --- Section 3: Analyse des Profils de Segments ---
st.header("Analyse des Profils de Segments")

segment_profiles = df.groupby('Segment').agg(
    Nombre_de_Clients=('customerID', 'count'),
    Ancienneté_Moyenne=('tenure', 'mean'),
    Facture_Moyenne=('MonthlyCharges', 'mean'),
    Taux_de_Churn=('Churn', lambda x: (x == 'Yes').mean())
).reset_index()

# Formatage pour un meilleur affichage
segment_profiles['Ancienneté_Moyenne'] = segment_profiles['Ancienneté_Moyenne'].round(1)
segment_profiles['Facture_Moyenne'] = segment_profiles['Facture_Moyenne'].round(2)
segment_profiles['Taux_de_Churn'] = (segment_profiles['Taux_de_Churn'] * 100).round(2)

# Personas pour chaque segment
personas = {
    0: "Nouveaux Clients à Risque",
    1: "Clients Stables à Faible Coût",
    2: "Clients Fidèles et Rentables",
    3: "Clients Engagés à Long Terme"
}
segment_profiles['Persona'] = segment_profiles['Segment'].map(personas)

# --- Improve layout with columns for better organization
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("""
    <div class="section-header">
        <h2>🔍 Méthode du Coude</h2>
    </div>
    """, unsafe_allow_html=True)
    st.plotly_chart(fig_elbow, use_container_width=True, key="elbow_chart_col1")

with col2:
    st.markdown("""
    <div class="section-header">
        <h2>📊 Visualisation des Segments</h2>
    </div>
    """, unsafe_allow_html=True)
    st.plotly_chart(fig_clusters, use_container_width=True, key="pca_chart_col2")

st.dataframe(
    segment_profiles[['Persona', 'Nombre_de_Clients', 'Ancienneté_Moyenne', 'Facture_Moyenne', 'Taux_de_Churn']],
    use_container_width=True,
    hide_index=True,
    column_config={
        "Persona": st.column_config.TextColumn("Profil du Segment"),
        "Taux_de_Churn": st.column_config.ProgressColumn(
            "Taux de Churn (%)",
            format="%.2f %%",
            min_value=0,
            max_value=segment_profiles['Taux_de_Churn'].max(),
        ),
    }
)

# --- Add a footer for consistency
st.markdown("""
<div style="text-align: center; margin-top: 2rem;">
    <p style="color: var(--text-color); font-size: 0.9rem;">© 2025 AI Churn Dashboard - All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)
