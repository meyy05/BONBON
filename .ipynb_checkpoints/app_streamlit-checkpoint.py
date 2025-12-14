#!/usr/bin/env python
# coding: utf-8

# In[1]:


# app_streamlit.py
import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# --- Config page ---
st.set_page_config(page_title="Popularité d'un bonbon", page_icon="🍬", layout="centered")

# --- CSS pour le style (pastel / carte) ---
st.markdown(
    """
    <style>
    :root{
      --rose:#ffd6e0;
      --rose-2:#ffc0cb;
      --accent:#f7a6c2;
      --card:#fff5f8;
      --text:#5a2b3a;
    }
    body { background: linear-gradient(180deg, #fff7fb 0%, #fff 100%); }
    .main-card {
      background: linear-gradient(180deg,var(--card), #fff);
      border-radius: 16px;
      padding: 28px;
      box-shadow: 0 8px 24px rgba(200,120,150,0.12);
      color: var(--text);
    }
    .title {
      font-family: 'Helvetica', Arial;
      font-weight: 800;
      font-size: 34px;
      letter-spacing: 1px;
      color: #b23b66;
      margin-bottom: 6px;
    }
    .subtitle {
      color: #955065;
      margin-top: -6px;
      margin-bottom: 18px;
      font-style: italic;
    }
    .predict-btn {
      background: linear-gradient(90deg,var(--accent), #ff8fb9);
      color: white;
      padding: 8px 18px;
      border-radius: 10px;
      font-weight: 700;
      border: none;
    }
    .result-box {
      border: 2px solid #f2a1b9;
      padding: 12px;
      border-radius: 8px;
      background: #fff;
    }
    .tree-btn {
      background:#f7a6c2;
      color:white;
      padding:8px 14px;
      border-radius:10px;
      font-weight:700;
      border:none;
    }
    footer { visibility: hidden; } /* hide streamlit footer */
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Main card ---
st.markdown("<div class='main-card'>", unsafe_allow_html=True)
st.markdown("<div class='title'>POPULARITÉ<br><span style='font-size:20px;color:#e07a9b'>D'UN BONBON</span></div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Vous aimez votre bonbon</div>", unsafe_allow_html=True)

# --- Load or train model (simple) ---
MODEL_PATH = "modele_regressor.joblib"
CSV = "candy-data.csv"

def train_simple_model(csv_path=CSV, model_out=MODEL_PATH):
    # fonction minimaliste pour (re)entrainer le modèle si nécessaire
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor, export_text
    import numpy as np

    df = pd.read_csv(csv_path)
    # nettoyage identique à ce que nous avons utilisé
    cols_to_drop = ['competitorname', 'pricepercent', 'sugarpercent', 'pluribus', 'bar', 'hard']
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    df = df.rename(columns={
        "chocolate": "chocolat",
        "fruity": "fruité",
        "caramel": "caramel",
        "peanutyalmondy": "cacahuètes_amandes",
        "nougat": "nougat",
        "crispedricewafer": "riz_soufflé",
        "winpercent": "popularité"
    })
    X = df.drop(columns=["popularité"])
    y = df["popularité"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = DecisionTreeRegressor(max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    # export rules textuable
    try:
        text = export_text(model, feature_names=list(X.columns))
        with open("regles_arbre.txt", "w", encoding="utf-8") as f:
            f.write(text)
    except Exception:
        pass
    joblib.dump(model, model_out)
    return model, list(X.columns)

model = None
feature_names = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        # try to get feature names from regles file or csv
        if os.path.exists("regles_arbre.txt"):
            # fallback: infer from CSV
            df_tmp = pd.read_csv(CSV)
            df_tmp = df_tmp.rename(columns={
                "chocolate": "chocolat",
                "fruity": "fruité",
                "caramel": "caramel",
                "peanutyalmondy": "cacahuètes_amandes",
                "nougat": "nougat",
                "crispedricewafer": "riz_soufflé",
                "winpercent": "popularité"
            })
            feature_names = [c for c in df_tmp.columns if c != "popularité"]
        else:
            df_tmp = pd.read_csv(CSV)
            df_tmp = df_tmp.rename(columns={
                "chocolate": "chocolat",
                "fruity": "fruité",
                "caramel": "caramel",
                "peanutyalmondy": "cacahuètes_amandes",
                "nougat": "nougat",
                "crispedricewafer": "riz_soufflé",
                "winpercent": "popularité"
            })
            feature_names = [c for c in df_tmp.columns if c != "popularité"]
    except Exception:
        model, feature_names = train_simple_model()
else:
    model, feature_names = train_simple_model()

# --- Formulaire (radio buttons) ---
cols = feature_names or ["chocolat", "fruité", "caramel", "cacahuètes_amandes", "nougat", "riz_soufflé"]
# show options as radio or checkbox group; per design we use radio-like vertical choices (one by one)
st.write("")  # spacing

# We want multiple independent options (user can check each preference)
cols_display_map = {
    "chocolat": "Chocolaté",
    "fruité": "Fruité",
    "caramel": "Caramélisé",
    "cacahuètes_amandes": "Cacahuètes / Amandes",
    "riz_soufflé": "Riz soufflé",
    "nougat": "Nougat"
}

# Layout: two columns for checkboxes maybe
left, right = st.columns([1,1])
with left:
    chocolat = st.checkbox(cols_display_map.get("chocolat","chocolat"), value=False)
    fruite = st.checkbox(cols_display_map.get("fruité","fruité"), value=False)
    caramel = st.checkbox(cols_display_map.get("caramel","caramel"), value=False)
with right:
    cacahuetes = st.checkbox(cols_display_map.get("cacahuètes_amandes","cacahuètes_amandes"), value=False)
    riz = st.checkbox(cols_display_map.get("riz_soufflé","riz_soufflé"), value=False)
    nougat = st.checkbox(cols_display_map.get("nougat","nougat"), value=False)

st.write("")  # spacing

# Predict button
predict_clicked = st.button("Prédir", key="predict")

# Area to display results with custom HTML/CSS
if predict_clicked:
    # build input dict matching feature names order
    inp = {}
    for c in cols:
        # consistent keys: our features are binary 0/1
        if c in ["chocolat","chocolate"]:
            inp[c] = int(chocolat)
        elif c == "fruité" or c == "fruity":
            inp[c] = int(fruite)
        elif c == "caramel":
            inp[c] = int(caramel)
        elif c in ["cacahuètes_amandes","peanutyalmondy"]:
            inp[c] = int(cacahuetes)
        elif c in ["riz_soufflé","crispedricewafer"]:
            inp[c] = int(riz)
        elif c == "nougat":
            inp[c] = int(nougat)
        else:
            # default 0
            inp[c] = 0

    # create dataframe row
    row = pd.DataFrame([inp])[cols]  # preserve order
    try:
        pred = float(model.predict(row)[0])
        pred_text = f"{pred:.2f} %"
    except Exception as e:
        pred_text = "Erreur lors de la prédiction"

    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
    st.markdown("<strong>Résultat</strong><br>", unsafe_allow_html=True)
    st.markdown(f"<div style='margin-top:6px'>Score estimé : <span style='font-weight:700;color:#c03a68'>{pred_text}</span></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Button to show textual rules
if st.button("Afficher les règles de l'arbre"):
    if os.path.exists("regles_arbre.txt"):
        with open("regles_arbre.txt", "r", encoding="utf-8") as f:
            rules = f.read()
        st.text_area("Règles (arbre de décision)", value=rules, height=300)
    else:
        st.info("Aucune règle trouvée. Entraîne le modèle d'abord (le modèle a été entraîné automatiquement si nécessaire).")

st.markdown("<br>", unsafe_allow_html=True)

# Button to show the tree image (PNG) if present, otherwise plot inline
if st.button("Arbre de décision"):
    # prefer arbre_regressor.png or arbre.png
    if os.path.exists("arbre_regressor.png"):
        st.image("arbre_regressor.png", use_column_width=True)
    elif os.path.exists("arbre.png"):
        st.image("arbre.png", use_column_width=True)
    else:
        # create a plot inline from the model
        try:
            fig, ax = plt.subplots(figsize=(12,6))
            plot_tree(model, feature_names=cols, filled=True, rounded=True, ax=ax)
            st.pyplot(fig)
        except Exception as e:
            st.error("Impossible d'afficher l'arbre: " + str(e))

st.markdown("</div>", unsafe_allow_html=True)


# In[ ]:




