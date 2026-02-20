import os
import streamlit as st
import pandas as pd

# 1. Configuration de la page
st.set_page_config(page_title="RobotReview", page_icon="🧠", layout="wide")

# Style CSS pour rendre l'interface plus "Tech"
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stInfo { border-left: 5px solid #4B90CD; }
    </style>
    """, unsafe_allow_html=True) # <-- C'est ici que j'avais fait une faute

# 2. En-tête
st.title("🧠 RobotReview")
st.subheader("Hybrid Cloud/Local AI Product Analysis")
st.markdown("---")

# 3. Chargement des données
@st.cache_data
def load_data():
    return pd.read_csv("articles_blog.csv")

try:
    df_articles = load_data()
    
    # 4. Barre latérale (Sidebar)
    with st.sidebar:
        st.header("⚙️ Model Engine")
        choix_modele = st.radio(
            "Select Processing Intelligence:",
            ["Gemini 2.5 (Cloud API)", "Qwen 2.5 (Local RTX 5070)"]
        )
        
        st.markdown("---")
        st.header("🛒 Product Category")
        categorie_choisie = st.selectbox(
            "Choose a category to inspect:", 
            df_articles['Categorie'].unique()
        )
        
        st.markdown("---")
        st.caption("Developed with Scikit-Learn (K-Means) & Transformers.")

    # 5. Affichage Principal
    col1, col2 = st.columns([2, 1]) 

    # Dictionnaire d'images locales
    images_categories = {
        'Tablets Med (8")': "tabletMed8.jpg",
        'Tablets Mini (7")': "tabletMini.jpg",
        'Streaming & TV': "streamingTv.jpg",
        'Kids Tablets': "kidTablet.jpg",
        'Smart Home & Speakers': "speakers.jpg",
        'Batteries': "battery.jpg"
    }

    with col1:
        st.markdown(f"### Analysis for: **{categorie_choisie}**")
        
        # Affichage de l'image
        if categorie_choisie in images_categories:
            img_name = images_categories[categorie_choisie]
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            img_path = os.path.join(BASE_DIR, "img", img_name)
            
            if os.path.exists(img_path):
                st.image(img_path, width=450)
            else:
                st.warning(f"📸 Image '{img_name}' not found in /img folder.")

        # Affichage de l'article selon le modèle
        col_to_read = "Article_Gemini" if "Gemini" in choix_modele else "Article_Local"
        content = df_articles[df_articles['Categorie'] == categorie_choisie][col_to_read].values[0]
        
        st.info(content)

    with col2:
        st.markdown("### 📊 Methodology")
        st.success("**Positive Bias:** Filtered from 4-5 star ratings.")
        st.error("**Negative Clusters:** Extracted via K-Means on negative reviews.")
        
        st.markdown("---")
        st.markdown("### 🤖 Model Comparison")
        if "Gemini" in choix_modele:
            st.write("**Provider:** Google Cloud")
            st.write("**Latency:** Network dependent")
        else:
            st.write("**Provider:** Local GPU (RTX 5070)")
            st.write("**Latency:** 0ms (Inference pre-computed)")

except FileNotFoundError:
    st.error("🚨 'articles_blog.csv' is missing. Run your notebook first!")