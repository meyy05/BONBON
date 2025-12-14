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
BACKGROUND_IMAGE = r"bg2.jpg"  # place bg.jpg in same folder as notebook / .py after conversion
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
        return None
    return get_base64_of_file(image_path)

bg_image_base64 = set_background(BACKGROUND_IMAGE)


 
if bg_image_base64:
    css = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@400;600;700&family=Pacifico&display=swap');

        /* === PAGE BACKGROUND === */
        .stApp {{
            background-image: url("data:image/jpg;base64,{bg_image_base64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            font-family: 'Fredoka', sans-serif;
        }}

        /* === CARD PRINCIPALE === */
       

        @keyframes rotate {{
            from {{ transform: rotate(0deg); }}
            to {{ transform: rotate(360deg); }}
        }}

        /* === TITRES === */
        .title {{
            font-family: 'Pacifico', cursive;
            font-weight: 400;
            font-size: 68px;
            background: linear-gradient(135deg, #ff69b4, #ff1493, #ff69b4, #ff8fb9);
            background-size: 200% 200%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-align: center;
            letter-spacing: 2px;
            text-shadow: 2px 2px 4px rgba(255, 105, 180, 0.2);
            animation: gradient-shift 3s ease infinite;
            position: relative;
            z-index: 1;
        }}

        @keyframes gradient-shift {{
            0%, 100% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
        }}

        .subtitle {{
            color: #d05f87;
            font-style: italic;
            text-align: center;
            margin-top: -6px;
            margin-bottom: 28px;
            font-size: 22px;
            font-weight: 600;
            text-shadow: 1px 1px 2px rgba(255, 182, 193, 0.5);
            position: relative;
            z-index: 1;
        }}

        /* === CHECKBOXES STYLISÉES === */
        .stCheckbox {{
            background: rgba(255, 255, 255, 0.7);
            padding: 12px 16px;
            border-radius: 15px;
            margin: 8px 0;
            border: 2px solid #ffb6c9;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(255, 105, 180, 0.1);
        }}

        .stCheckbox:hover {{
            transform: translateX(5px);
            background: rgba(255, 240, 245, 0.9);
            border-color: #ff8fb9;
            box-shadow: 0 6px 20px rgba(255, 105, 180, 0.2);
        }}

        .stCheckbox label {{
            font-weight: 600;
            color: #7a3a4a;
            font-size: 16px;
        }}

        /* === BOUTONS === */
        .stButton > button {{
            display: block;
            margin: 16px auto;
            background: linear-gradient(135deg, #ff69b4, #ff8fb9, #ffb6c9);
            color: white;
            padding: 14px 32px;
            border-radius: 25px;
            font-weight: 800;
            font-size: 18px;
            border: none;
            cursor: pointer;
            box-shadow: 0 8px 20px rgba(255, 105, 180, 0.3),
                        inset 0 -2px 10px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
            position: relative;
            overflow: hidden;
        }}

        .stButton > button::before {{
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.3);
            transform: translate(-50%, -50%);
            transition: width 0.6s, height 0.6s;
        }}

        .stButton > button:hover::before {{
            width: 300px;
            height: 300px;
        }}

        .stButton > button:hover {{
            transform: translateY(-3px) scale(1.05);
            box-shadow: 0 12px 30px rgba(255, 105, 180, 0.4),
                        inset 0 -2px 10px rgba(0, 0, 0, 0.1);
        }}

        .stButton > button:active {{
            transform: translateY(0) scale(0.98);
        }}

        /* === RESULTAT === */
        .result-box {{
            border: 3px solid #ff8fb9;
            padding: 20px;
            border-radius: 20px;
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.98), rgba(255, 240, 245, 0.98));
            margin-top: 20px;
            color: #7a3a4a;
            font-size: 20px;
            font-weight: 600;
            text-align: center;
            box-shadow: 0 10px 30px rgba(255, 105, 180, 0.2),
                        inset 0 2px 10px rgba(255, 182, 193, 0.3);
            animation: pop-in 0.5s ease;
        }}

        @keyframes pop-in {{
            0% {{ transform: scale(0.8); opacity: 0; }}
            50% {{ transform: scale(1.05); }}
            100% {{ transform: scale(1); opacity: 1; }}
        }}

        .result-box b {{
            font-size: 24px;
            color: #ff1493;
            display: block;
            margin-bottom: 10px;
        }}

        /* === TEXT AREA === */
        .stTextArea textarea {{
            border: 2px solid #ffb6c9;
            border-radius: 15px;
            background: rgba(255, 255, 255, 0.9);
            font-family: 'Courier New', monospace;
            color: #7a3a4a;
            padding: 15px;
        }}

        /* === SIDEBAR === */
        .css-1d391kg {{
            background: linear-gradient(180deg, rgba(255, 240, 245, 0.95), rgba(255, 228, 240, 0.95));
            border-right: 3px solid #ffb6c9;
        }}

        /* === IMAGES === */
        .stImage {{
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(255, 105, 180, 0.3);
            border: 3px solid #ffb6c9;
            overflow: hidden;
        }}

        /* === EMOJI DECORATIFS === */
        .stApp::after {{
            content: '🍬 🍭 🍫 🍰';
            position: fixed;
            bottom: 20px;
            right: 20px;
            font-size: 30px;
            opacity: 0.3;
            pointer-events: none;
            animation: float 3s ease-in-out infinite;
        }}

        @keyframes float {{
            0%, 100% {{ transform: translateY(0px); }}
            50% {{ transform: translateY(-15px); }}
        }}

        /* Cacher le footer Streamlit */
        footer {{
            visibility: hidden;
        }}

        /* === VERSION MOBILE === */
        @media(max-width:600px) {{
            .title {{
                font-size: 42px;
            }}
            .subtitle {{
                font-size: 18px;
            }}
            .main-card {{
                padding: 25px 20px;
            }}
        }}

    </style>
    """
else:
    css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@400;600;700&family=Pacifico&display=swap');
        
        .stApp {{
            background: linear-gradient(135deg, #fff5f8, #ffe4f0, #ffd4e8);
            font-family: 'Fredoka', sans-serif;
        }}
        
        .main-card {{
            background: rgba(255, 255, 255, 0.9);
            border-radius: 30px;
            padding: 40px 30px;
            box-shadow: 0 20px 60px rgba(255, 105, 180, 0.3);
            border: 3px solid #ffb6c9;
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





# In[ ]:




