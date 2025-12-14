#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import base64
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn import tree

# ---------------- CONFIG ----------------
BACKGROUND_IMAGE = r"bg1.jpg"  # place bg.jpg in same folder as notebook / .py after conversion
CSV = "candy-data.csv"
MODEL_PATH = "modele_regressor.joblib"
RANDOM_STATE = 42

st.set_page_config(page_title="Popularité d'un bonbon", page_icon="🍬", layout="centered")

# ---------------- HELPERS ----------------
def get_base64_of_file(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

def set_background(image_path):
    if not os.path.exists(image_path):
        return False
    img_b64 = get_base64_of_file(image_path)
    css = f"""
    <style>
    /* full page background */
    .stApp {{
      background-image: url('data:image/jpeg;base64,{img_b64}');
      background-position: center;
      background-repeat: no-repeat;
    }}

    /* overlay to keep text readable */
    .stApp::before {{
      content: "";
      position: fixed;
      inset: 0;
      background: rgba(255, 245, 248, 0.86);
      pointer-events: none;
    }}

   

    .title {{
      font-weight: 900;
      font-size: 30px;
      color: #b23b66;
      text-align: center;
      letter-spacing: 1px;
    }}

    .subtitle {{
      color: #955065;
      font-style: italic;
      text-align: center;
      margin-top: -6px;
      margin-bottom: 18px;
    }}

    .choices {{
      display: block;
      margin-left: 12px;
      margin-bottom: 8px;
    }}

    .predict-btn {{
      display:block;
      margin: 14px auto 6px auto;
      background: linear-gradient(90deg,#f7a6c2,#ff8fb9);
      color: white;
      padding: 10px 20px;
      border-radius: 12px;
      font-weight: 800;
      border: none;
      box-shadow: 0 6px 18px rgba(200,100,140,0.12);
    }}

    .result-box {{
      border: 2px solid #e79aa6;
      padding: 12px;
      border-radius: 8px;
      background: rgba(255,255,255,0.98);
      margin-top: 12px;
    }}

    .tree-btn {{
      display:block;
      margin: 10px auto 0 auto;
      background: #f7a6c2;
      color: white;
      padding: 8px 14px;
      border-radius: 10px;
      font-weight: 700;
      border:none;
    }}

    footer {{visibility: hidden;}}

    @media(max-width:600px) {{
      .main-card {{ margin: 12px; padding:18px; max-width: 360px; }}
      .title {{ font-size: 22px; }}
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def clean_and_prepare(csv_path):
    df = pd.read_csv(csv_path)
    # drop obviously irrelevant columns if present
    to_drop = ['competitorname','pricepercent','sugarpercent','pluribus','bar','hard']
    df = df.drop(columns=[c for c in to_drop if c in df.columns])
    df = df.rename(columns={
        'chocolate':'chocolat','fruity':'fruité','caramel':'caramel',
        'peanutyalmondy':'cacahuètes_amandes','nougat':'nougat',
        'crispedricewafer':'riz_soufflé','winpercent':'popularité'
    })
    # ensure numeric types
    for c in df.select_dtypes(include=['object']).columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except Exception:
            pass
    return df

def train_and_save(csv_path=CSV, model_path=MODEL_PATH):
    df = clean_and_prepare(csv_path)
    if 'popularité' not in df.columns:
        raise SystemExit("Colonne 'popularité' introuvable dans le CSV après nettoyage.")
    X = df.drop(columns=['popularité'])
    y = df['popularité']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)
    model = DecisionTreeRegressor(max_depth=4, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    # save model
    joblib.dump(model, model_path)
    # save arbre.png via matplotlib
    try:
        fig, ax = plt.subplots(figsize=(10,6))
        plot_tree(model, feature_names=list(X.columns), filled=True, rounded=True, ax=ax)
        fig.tight_layout()
        fig.savefig('arbre.png', dpi=150)
        plt.close(fig)
    except Exception as e:
        print('Warning: impossible de sauvegarder arbre.png:', e)
    return model, list(X.columns), (X_test, y_test)

# set background (if available)
if os.path.exists(BACKGROUND_IMAGE):
    set_background(BACKGROUND_IMAGE)
else:
    st.warning(f"Image de fond '{BACKGROUND_IMAGE}' introuvable — place-la dans le dossier ou change le chemin.")

# load or train model
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        df_probe = clean_and_prepare(CSV)
        feature_names = [c for c in df_probe.columns if c != 'popularité']
    except Exception:
        model, feature_names, testset = train_and_save()
else:
    model, feature_names, testset = train_and_save()

# make sure model.feature_names_in_ exists (fit in scikit-learn sets that)
try:
    expected = list(model.feature_names_in_)
except Exception:
    expected = None

# ---------------- UI (pages) ----------------
page = st.sidebar.selectbox('Navigation', ['Accueil', "À propos"]) 

if page == 'Accueil':
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("<div class='title'>POPULARITÉ</div>", unsafe_allow_html=True)
    st.markdown("<div class='title' style='font-size:18px;margin-top:-6px;color:#f09aa6'>D'UN BONBON</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Vous aimez votre bonbon</div>", unsafe_allow_html=True)
    
    # display checkboxes in two columns
    cols = feature_names or ['chocolat','fruité','caramel','cacahuètes_amandes','riz_soufflé','nougat']
    mapping = {'chocolat':'Chocolaté','fruité':'Fruité','caramel':'Caramélisé','cacahuètes_amandes':'Cacahuètes / Amandes','riz_soufflé':'Riz soufflé','nougat':'Nougat'}
    left, right = st.columns([1,1])
    with left:
        v_choc = st.checkbox(mapping.get('chocolat'))
        v_fru = st.checkbox(mapping.get('fruité'))
        v_car = st.checkbox(mapping.get('caramel'))
    with right:
        v_pea = st.checkbox(mapping.get('cacahuètes_amandes'))
        v_riz = st.checkbox(mapping.get('riz_soufflé'))
        v_nou = st.checkbox(mapping.get('nougat'))

    if st.button('Prédire'):
        # build input vector matching expected feature names
        if expected is not None:
            inp = {name: 0 for name in expected}
            # map known friendly names to potential expected names
            mapping_inputs = {
                'chocolat': int(v_choc),
                'fruité': int(v_fru),
                'caramel': int(v_car),
                'cacahuètes_amandes': int(v_pea),
                'riz_soufflé': int(v_riz),
                'nougat': int(v_nou)
            }
            for k,v in mapping_inputs.items():
                if k in inp:
                    inp[k]=v
                else:
                    # try alternatives (english names or without accents)
                    alt = k.replace('é','e')
                    if alt in inp:
                        inp[alt]=v
            row = pd.DataFrame([inp], columns=list(inp.keys()))
        else:
            # fallback: use feature_names order
            row = pd.DataFrame([{c:int((v_choc if c=='chocolat' else v_fru) ) for c in cols}])

        try:
            pred = float(model.predict(row)[0])
            st.markdown(f"<div class='result-box'><b>Résultat</b><br>Score estimé : {pred:.2f} %</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error('Erreur lors de la prédiction : ' + str(e))

    if st.button('Afficher les règles de l\'arbre'):
        try:
            rules = export_text(model, feature_names=feature_names)
            st.text_area('Règles (arbre de décision)', value=rules, height=300)
        except Exception as e:
            st.error('Impossible d\'exporter les règles : '+str(e))

    if st.button('Arbre de décision (PNG)'):
        if os.path.exists('arbre.png'):
            st.image('arbre.png', use_column_width=True)
        else:
            try:
                fig, ax = plt.subplots(figsize=(10,6))
                plot_tree(model, feature_names=feature_names, filled=True, rounded=True, ax=ax)
                fig.tight_layout()
                fig.savefig('arbre.png', dpi=150)
                plt.close(fig)
                st.image('arbre.png', use_column_width=True)
            except Exception as e:
                st.error('Impossible d\'afficher/générer l\'arbre: '+str(e))

    st.markdown('</div>', unsafe_allow_html=True)

else:
    # À propos page
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("<div class='title'>À PROPOS</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Informations sur le projet</div>", unsafe_allow_html=True)
    st.markdown('''
    <p>Ce projet est un système expert simple basé sur un arbre de décision entraîné sur le dataset <code>candy-data.csv</code>.
    Il prédit la <strong>popularité</strong> (winpercent) d'un bonbon à partir de caractéristiques binaires (chocolaté, fruité, etc.).</p>
    <ul>
    <li>Nettoyage : suppression des colonnes inutiles.</li>
    <li>Modèle : DecisionTreeRegressor (max_depth=4).</li>
    <li>Export : règles texte et image de l'arbre (arbre.png).</li>
    </ul>
    ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# In[ ]:




