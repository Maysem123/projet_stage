# =============================================================================
# Fichier : src/app.py
# Rôle    : Contient l'application web Streamlit pour l'interface utilisateur.
# =============================================================================

# --- 1. Importation des bibliothèques nécessaires ---
import streamlit as st
import pandas as pd

# --- 2. Configuration de la page Streamlit ---
# Cette commande doit être la première instruction Streamlit de votre script.
st.set_page_config(
    page_title="Dashboard Churn IA",
    page_icon="🤖",
    layout="wide"
)

# --- 3. Chargement des données (avec mise en cache pour la performance) ---
# Cette fonction charge le fichier CSV. Le décorateur @st.cache_data
# garantit que les données ne sont lues qu'une seule fois.
@st.cache_data
def load_data(path):
    """Charge les données depuis un fichier CSV."""
    df = pd.read_csv(path)
    return df

# Appel de la fonction pour charger les données.
# Le chemin est direct car on considère que le script est lancé depuis la racine.
df_data = load_data('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')


# --- 4. Construction de l'Interface Utilisateur (UI) ---

# Titre principal et description de l'application
st.title("🤖 Tableau de Bord de Prédiction du Churn Client")
st.markdown("Cette application est une maquette fonctionnelle qui prédit le risque de départ d'un client et vise à expliquer les raisons derrière cette prédiction.")
st.markdown("---") # Ajoute une ligne de séparation visuelle

# --- BARRE LATÉRALE pour les contrôles de l'utilisateur ---
st.sidebar.header("⚙️ Paramètres de Sélection")

# Création du menu déroulant pour sélectionner un client
liste_ids_clients = df_data['customerID'].tolist()
selected_customer_id = st.sidebar.selectbox(
    "Sélectionnez un ID Client pour analyse :",
    liste_ids_clients
)


# --- PANNEAU PRINCIPAL pour l'affichage des résultats ---
st.header(f"📊 Diagnostic pour le Client : {selected_customer_id}")

# Récupération de la ligne de données pour le client sélectionné
client_info = df_data[df_data['customerID'] == selected_customer_id]

# Organisation de l'affichage en deux colonnes pour une meilleure lisibilité
col1, col2 = st.columns(2)

# Colonne de gauche : Affichage de la prédiction (actuellement factice)
with col1:
    st.subheader("🔮 Prédiction du Risque de Départ")
    
    # --- !! PLACEHOLDER / DONNÉES FACTICES !! ---
    # Ces valeurs seront remplacées par les vrais résultats du modèle plus tard.
    # C'est la partie qui sera connectée au backend.
    probabilite_churn = 0.85 # Exemple de valeur statique
    
    # Affichage de la métrique avec un formatage en pourcentage
    st.metric("Probabilité de Départ", f"{probabilite_churn:.0%}")
    
    # Logique conditionnelle pour afficher un statut visuel (rouge/vert)
    if probabilite_churn > 0.5:
        st.error("🔴 Statut Prédit : Client à Haut Risque de Churn")
    else:
        st.success("🟢 Statut Prédit : Client Fidèle")
    
    st.info("Cette prédiction sera générée par un modèle d'IA (XGBoost) après le développement du backend.", icon="ℹ️")

# Colonne de droite : Affichage des informations de base du client
with col2:
    st.subheader("👤 Informations Contractuelles")
    
    st.write(f"**Type de Contrat :** `{client_info['Contract'].iloc[0]}`")
    st.write(f"**Ancienneté (mois) :** `{client_info['tenure'].iloc[0]}`")
    st.write(f"**Facture Mensuelle :** `{client_info['MonthlyCharges'].iloc[0]:.2f} $`")
    # Conversion de TotalCharges en numérique pour éviter les erreurs avec les espaces
    total_charges = pd.to_numeric(client_info['TotalCharges'], errors='coerce').iloc[0]
    st.write(f"**Facture Totale :** `{total_charges:.2f} $`")

# Ligne de séparation avant la section suivante
st.markdown("---")

# Section pour l'explication XAI (avec un placeholder visuel)
st.header("🧠 Facteurs d'Influence (Analyse XAI)")
st.warning(
    "Cette section est en attente du développement du backend. "
    "Elle affichera un graphique expliquant les raisons de la prédiction du modèle.",
    icon="⚠️"
)

# Placeholder visuel pour le futur graphique SHAP
st.image(
    'https://miro.medium.com/v2/resize:fit:1400/1*j_3_2N9NArY9i-5n12t3zA.png',
    caption="Exemple de graphique d'explicabilité (SHAP) qui sera généré ici."
)