#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import base64
from io import BytesIO

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ----------------- CONFIG -----------------
# Chemin absolu vers ton image (selon ta demande)
BACKGROUND_IMAGE = r"C:\Users\NOUR HAMZA\BONBON\bg.jpg"
CSV = "candy-data.csv"
MODEL_PATH = "modele_regressor.joblib"
RANDOM_STATE = 42

st.set_page_config(page_title="Popularité d'un bonbon", page_icon="🍬", layout="centered")

# ----------------- HELPERS -----------------
def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(image_path):
    """
    Inject CSS to use the provided image_path as a page background.
    Returns True on success, False if image not found.
    """
    if not os.path.exists(image_path):
        return False

    img_b64 = get_base64_of_bin_file(image_path)

    # Note: we use double braces {{ }} inside f-string to produce single braces in final CSS
    css = f"""
    <style>
    .stApp {{
      background-image: url("data:image/jpeg;base64,{img_b64}");
      background-size: cover;
      background-position: center;
      background-repeat: no-repeat;
    }}

    /* light overlay to keep content readable (adjust alpha if needed) */
    .stApp::before {{
      content: "";
      position: fixed;
      inset: 0;
      background: rgba(255, 245, 248, 0.85);
      pointer-events: none;
    }}

    /* MAIN CARD */
    .main-card {{
      background: rgba(255, 243, 244, 0.96);
      border-radius: 18px;
      padding: 26px;
      box-shadow: 0 8px 30px rgba(190,120,150,0.12);
      color: #5a2b3a;
      max-width: 420px;
      margin: 48px auto;
    }}

    .title {{
      font-weight: 800;
      font-size: 28px;
      color: #b23b66;
      margin-bottom: 4px;
      letter-spacing: 0.5px;
    }}

    .subtitle {{
      color: #955065;
      margin-top: -4px;
      margin-bottom: 16px;
      font-style: italic;
    }}

    .choices-row {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      margin-bottom: 12px;
    }}

    .predict-btn {{
      background: linear-gradient(90deg,#f7a6c2,#ff8fb9);
      color: white;
      padding: 8px 18px;
      border-radius: 10px;
      font-weight: 700;
      border: none;
      margin-top: 8px;
    }}

    .result-box {{
      border: 2px solid #d88a96;
      padding: 12px;
      border-radius: 8px;
      background: rgba(255,255,255,0.98);
      margin-top: 12px;
    }}

    .tree-btn {{
      background: #f7a6c2;
      color: white;
      padding: 8px 12px;
      border-radius: 10px;
      font-weight: 700;
      border: none;
      margin-top: 12px;
    }}

    footer {{
      visibility: hidden;
    }}

    @media(max-width:600px) {{
      .main-card {{ margin: 12px; padding:18px; max-width: 360px; }}
      .title {{ font-size: 22px; }}
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    return True

# Apply background
bg_ok = set_background(BACKGROUND_IMAGE)
if not bg_ok:
    st.warning(f"Image de fond introuvable : '{BACKGROUND_IMAGE}'. Place-la à cet emplacement ou modifie le chemin.")

# ----------------- MODEL TRAIN / LOAD -----------------
def train_simple_model(csv_path=CSV, model_out=MODEL_PATH):
    df = pd.read_csv(csv_path)
    # nettoyage (comme dans ton test.py)
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

    if "popularité" not in df.columns:
        raise SystemExit("La colonne 'popularité' est introuvable dans le CSV.")

    X = df.drop(columns=["popularité"])
    y = df["popularité"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE)
    model = DecisionTreeRegressor(max_depth=4, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    # save textual rules
    try:
        rules_text = export_text(model, feature_names=list(X.columns))
        with open("regles_arbre.txt", "w", encoding="utf-8") as f:
            f.write(rules_text)
    except Exception:
        pass

    joblib.dump(model, model_out)
    return model, list(X.columns), (X_test, y_test)

# Load or train model
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        # infer feature names from CSV reliably
        tmp = pd.read_csv(CSV)
        tmp = tmp.rename(columns={
            "chocolate": "chocolat",
            "fruity": "fruité",
            "caramel": "caramel",
            "peanutyalmondy": "cacahuètes_amandes",
            "nougat": "nougat",
            "crispedricewafer": "riz_soufflé",
            "winpercent": "popularité"
        })
        feature_names = [c for c in tmp.columns if c != "popularité"]
    except Exception:
        model, feature_names, testset = train_simple_model()
else:
    model, feature_names, testset = train_simple_model()

# ----------------- UI -----------------
st.markdown("<div class='main-card'>", unsafe_allow_html=True)
st.markdown("<div class='title'>POPULARITÉ</div>", unsafe_allow_html=True)
st.markdown("<div class='title' style='font-size:18px;margin-top:-6px;color:#f09aa6'>D'UN BONBON</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Vous aimez votre bonbon</div>", unsafe_allow_html=True)

# Display choices as checkboxes (vertical like in your design)
cols = feature_names or ["chocolat", "fruité", "caramel", "cacahuètes_amandes", "riz_soufflé", "nougat"]
display_map = {
    "chocolat": "Chocolaté",
    "fruité": "Fruité",
    "caramel": "Caramélisé",
    "cacahuètes_amandes": "Cacahuètes / Amandes",
    "riz_soufflé": "Riz soufflé",
    "nougat": "Nougat"
}

# Two-column layout to approximate your screenshot
left_col, right_col = st.columns([1,1])
with left_col:
    chocolate = st.checkbox(display_map.get("chocolat"))
    fruity = st.checkbox(display_map.get("fruité"))
    caramel = st.checkbox(display_map.get("caramel"))
with right_col:
    peanuts = st.checkbox(display_map.get("cacahuètes_amandes"))
    rice = st.checkbox(display_map.get("riz_soufflé"))
    nougat = st.checkbox(display_map.get("nougat"))

# Predict button
if st.button("Prédire"):
    # Build input row respecting feature order
    inp = {}
    for c in cols:
        if c in ["chocolat","chocolate"]:
            inp[c] = int(chocolate)
        elif c in ["fruité","fruity"]:
            inp[c] = int(fruity)
        elif c == "caramel":
            inp[c] = int(caramel)
        elif c in ["cacahuètes_amandes","peanutyalmondy"]:
            inp[c] = int(peanuts)
        elif c in ["riz_soufflé","crispedricewafer"]:
            inp[c] = int(rice)
        elif c == "nougat":
            inp[c] = int(nougat)
        else:
            inp[c] = 0

    row = pd.DataFrame([inp])[cols]
    try:
        pred = float(model.predict(row)[0])
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        st.markdown(f"<strong>Résultat</strong><br><div style='margin-top:6px'>Score estimé : <span style='font-weight:700;color:#c03a68'>{pred:.2f} %</span></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error("Erreur lors de la prédiction : " + str(e))

# Button:Afficher règles
if st.button("Afficher les règles de l'arbre"):
    if os.path.exists("regles_arbre.txt"):
        with open("regles_arbre.txt", "r", encoding="utf-8") as f:
            rules = f.read()
        st.text_area("Règles (arbre de décision)", value=rules, height=300)
    else:
        st.info("Aucune règle trouvée. Entraîne le modèle d'abord (ou supprime/regen modele).")

# Button: Afficher arbre (image or inline)
if st.button("Arbre de décision"):
    # prefer saved image if present
    if os.path.exists("arbre_regressor.png"):
        st.image("arbre_regressor.png", use_column_width=True)
    elif os.path.exists("arbre.png"):
        st.image("arbre.png", use_column_width=True)
    else:
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            plot_tree(model, feature_names=cols, filled=True, rounded=True, ax=ax)
            st.pyplot(fig)
        except Exception as e:
            st.error("Impossible d'afficher l'arbre: " + str(e))

# Optionally show simple evaluation (MSE / R2) if testset exists
if 'testset' in locals():
    X_test, y_test = testset
    try:
        y_pred_test = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)
        st.markdown(f"<div style='margin-top:10px;color:#6b3742;font-size:13px'>MSE: {mse:.3f} — R²: {r2:.3f}</div>", unsafe_allow_html=True)
    except Exception:
        pass

st.markdown("</div>", unsafe_allow_html=True)

