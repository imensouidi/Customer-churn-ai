import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import os
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, f1_score,
    recall_score, precision_score, accuracy_score,
    roc_curve, precision_recall_curve, average_precision_score
)
 
# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnGuard · Rétention Client",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# ─────────────────────────────────────────────
#  DESIGN TOKENS
# ─────────────────────────────────────────────
PRIMARY    = "#1B2B4B"   # Marine profond
ACCENT     = "#E63946"   # Rouge alerte
SUCCESS    = "#2DC653"   # Vert rétention
WARNING    = "#F4A261"   # Orange risque
NEUTRAL    = "#8D99AE"   # Gris texte secondaire
SURFACE    = "#F8F9FA"   # Fond carte
DARK       = "#0D1B2A"   # Fond sidebar
 
# ─────────────────────────────────────────────
#  CSS GLOBAL
# ─────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap%27);
 
  html, body, [class*="css"] {{
      font-family: 'Inter', sans-serif;
      background-color: #F0F2F6;
  }}
 
  /* Sidebar */
  [data-testid="stSidebar"] {{
      background-color: {DARK};
  }}
  [data-testid="stSidebar"] * {{
      color: #E2E8F0 !important;
  }}
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stSlider label,
  [data-testid="stSidebar"] .stNumberInput label {{
      color: {NEUTRAL} !important;
      font-size: 0.78rem !important;
      text-transform: uppercase;
      letter-spacing: 0.06em;
  }}
 
  /* Hero header */
  .hero {{
      background: linear-gradient(135deg, {PRIMARY} 0%, #2E4A7A 100%);
      border-radius: 16px;
      padding: 2rem 2.5rem;
      margin-bottom: 1.5rem;
      color: white;
  }}
  .hero h1 {{
      font-family: 'Space Grotesk', sans-serif;
      font-size: 2rem;
      font-weight: 700;
      margin: 0;
      letter-spacing: -0.02em;
  }}
  .hero p {{
      font-size: 0.95rem;
      color: #A8C0E0;
      margin: 0.4rem 0 0 0;
  }}
 
  /* KPI cards */
  .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 1rem;
      margin-bottom: 1.5rem;
  }}
  .kpi-card {{
      background: white;
      border-radius: 12px;
      padding: 1.2rem 1.4rem;
      border-left: 4px solid {PRIMARY};
      box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }}
  .kpi-card.danger  {{ border-left-color: {ACCENT}; }}
  .kpi-card.success {{ border-left-color: {SUCCESS}; }}
  .kpi-card.warning {{ border-left-color: {WARNING}; }}
  .kpi-label {{
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: {NEUTRAL};
      font-weight: 500;
  }}
  .kpi-value {{
      font-family: 'Space Grotesk', sans-serif;
      font-size: 1.9rem;
      font-weight: 700;
      color: {PRIMARY};
      line-height: 1.1;
      margin: 0.2rem 0 0 0;
  }}
  .kpi-sub {{
      font-size: 0.78rem;
      color: {NEUTRAL};
      margin-top: 0.1rem;
  }}
 
  /* Section card */
  .section-card {{
      background: white;
      border-radius: 12px;
      padding: 1.4rem 1.6rem;
      box-shadow: 0 1px 4px rgba(0,0,0,0.06);
      margin-bottom: 1rem;
  }}
  .section-title {{
      font-family: 'Space Grotesk', sans-serif;
      font-size: 1rem;
      font-weight: 700;
      color: white;
      margin-bottom: 1rem;
      padding-bottom: 0.5rem;
      border-bottom: 2px solid #EEF2F7;
  }}
 
  /* Prediction badge */
  .pred-badge {{
      display: inline-block;
      padding: 0.5rem 1.2rem;
      border-radius: 50px;
      font-weight: 700;
      font-size: 1rem;
      letter-spacing: 0.02em;
  }}
  .pred-churn  {{ background: #FEE2E2; color: {ACCENT}; }}
  .pred-stable {{ background: #DCFCE7; color: #15803D; }}
 
  /* Risk gauge text */
  .prob-text {{
      font-family: 'Space Grotesk', sans-serif;
      font-size: 3rem;
      font-weight: 700;
      text-align: center;
  }}
 
  /* Tab styling */
  .stTabs [data-baseweb="tab"] {{
      font-weight: 500;
      color: {NEUTRAL};
  }}
  .stTabs [aria-selected="true"] {{
      color: {PRIMARY} !important;
      border-bottom-color: {PRIMARY} !important;
  }}
 
  /* Hide Streamlit branding */
  #MainMenu, footer, header {{ visibility: hidden; }}
  .block-container {{ padding-top: 1.5rem; }}
</style>
""", unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────
#  CHARGEMENT DES MODÈLES ET DONNÉES
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
DATA_DIR   = os.path.join(BASE_DIR, "..", "data")
 
@st.cache_resource
def load_models():
    models = {}
    for name, fname in [
        ("Régression Logistique", "logistic_regression.pkl"),
        ("Random Forest",         "random_forest.pkl"),
        ("XGBoost",               "xgboost.pkl"),
        ("MLP (Deep Learning)",   "mlp.pkl"),
    ]:
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models
 
@st.cache_resource
def load_preprocessor():
    path = os.path.join(MODELS_DIR, "preprocessor.pkl")
    return joblib.load(path) if os.path.exists(path) else None
 
@st.cache_data
def load_test_data():
    X = np.load(os.path.join(MODELS_DIR, "X_test_prepared.npy"))
    y = np.load(os.path.join(MODELS_DIR, "y_test.npy"))
    return X, y
 
@st.cache_data
def load_full_data():
    path = os.path.join(DATA_DIR, "customer_churn_business_dataset.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None
 
@st.cache_resource
def get_feature_names(_preprocessor):
    if _preprocessor is None:
        return []
    num_features = list(_preprocessor.transformers_[0][2])
    cat_features = _preprocessor.transformers_[1][1]['onehot'].get_feature_names_out(
        _preprocessor.transformers_[1][2]
    ).tolist()
    return num_features + cat_features
 
models       = load_models()
preprocessor = load_preprocessor()
X_test, y_test = load_test_data()
df_full      = load_full_data()
feature_names = get_feature_names(preprocessor)
#best_model   = models.get("XGBoost", list(models.values())[0] if models else None)
best_model = models.get("Régression Logistique", list(models.values())[0] if models else None)
 
# ─────────────────────────────────────────────
#  SIDEBAR — NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0 1.5rem 0;">
        <div style="font-family:'Space Grotesk',sans-serif; font-size:1.3rem;
                    font-weight:700; color:white; letter-spacing:-0.01em;">
            🛡️ ChurnGuard
        </div>
        <div style="font-size:0.75rem; color:#64748B; margin-top:0.2rem;">
            Plateforme de rétention client
        </div>
    </div>
    """, unsafe_allow_html=True)
 
    page = st.radio(
        "Navigation",
        ["📊 Vue d'ensemble", "🔮 Prédiction client", "📈 Performance des modèles", "🔍 Variables influentes"],
        label_visibility="collapsed"
    )
 
    st.markdown("---")
    st.markdown(f"""
    <div style="font-size:0.72rem; color:#475569; line-height:1.6;">
        <div style="color:#94A3B8; font-weight:600; margin-bottom:0.4rem;">MODÈLE ACTIF</div>
        XGBoost<br>
        <span style="color:#2DC653;">● Opérationnel</span>
    </div>
    """, unsafe_allow_html=True)
 
    if df_full is not None:
        st.markdown("---")
        st.markdown(f"""
        <div style="font-size:0.72rem; color:#475569; line-height:1.8;">
            <div style="color:#94A3B8; font-weight:600; margin-bottom:0.4rem;">DATASET</div>
            {len(df_full):,} clients<br>
            {df_full['churn'].sum():,} churners ({df_full['churn'].mean()*100:.1f}%)
        </div>
        """, unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────
#  PAGE 1 — VUE D'ENSEMBLE
# ─────────────────────────────────────────────
if page == "📊 Vue d'ensemble":
 
    st.markdown("""
    <div class="hero">
        <h1>🛡️ ChurnGuard — Rétention Client</h1>
        <p>Système intelligent de détection du risque de résiliation · Dataset Business SaaS · 10 000 clients</p>
    </div>
    """, unsafe_allow_html=True)
 
    # KPIs globaux
    if df_full is not None and best_model is not None:
        y_proba_all = best_model.predict_proba(X_test)[:, 1]
        n_at_risk   = int((y_proba_all > 0.5).sum())
        pct_risk    = n_at_risk / len(y_test) * 100
        total_churn = int(y_test.sum())
 
        # Revenu à risque (si colonne disponible)
        if 'monthly_charges' in df_full.columns:
            avg_revenue = df_full['monthly_charges'].mean()
            revenue_at_risk = n_at_risk * avg_revenue
        else:
            revenue_at_risk = n_at_risk * 85  # estimation
 
        auc_score = roc_auc_score(y_test, y_proba_all)
 
        st.markdown(f"""
        <div class="kpi-grid">
            <div class="kpi-card danger">
                <div class="kpi-label">Clients à risque</div>
                <div class="kpi-value">{n_at_risk:,}</div>
                <div class="kpi-sub">{pct_risk:.1f}% du portefeuille test</div>
            </div>
            <div class="kpi-card warning">
                <div class="kpi-label">Revenu mensuel à risque</div>
                <div class="kpi-value">{revenue_at_risk:,.0f} €</div>
                <div class="kpi-sub">Estimation basée sur charges moyennes</div>
            </div>
            <div class="kpi-card success">
                <div class="kpi-label">ROC-AUC (XGBoost)</div>
                <div class="kpi-value">{auc_score:.3f}</div>
                <div class="kpi-sub">Modèle final sélectionné</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Churners réels (test)</div>
                <div class="kpi-value">{total_churn:,}</div>
                <div class="kpi-sub">{total_churn/len(y_test)*100:.1f}% du jeu de test</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
 
    # Graphiques EDA
    if df_full is not None:
        col1, col2 = st.columns(2)
 
        with col1:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Répartition du churn</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 3.5))
            counts = df_full['churn'].value_counts()
            colors_pie = [SUCCESS, ACCENT]
            wedges, texts, autotexts = ax.pie(
                counts, labels=['Fidèles', 'Churners'],
                colors=colors_pie, autopct='%1.1f%%',
                startangle=90, pctdistance=0.75,
                wedgeprops=dict(edgecolor='white', linewidth=2)
            )
            for at in autotexts:
                at.set_fontsize(11)
                at.set_fontweight('bold')
                at.set_color('white')
            ax.set_facecolor('none')
            fig.patch.set_alpha(0)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.markdown('</div>', unsafe_allow_html=True)
 
        with col2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Probabilités de churn — Distribution</div>', unsafe_allow_html=True)
            if best_model is not None:
                fig, ax = plt.subplots(figsize=(5, 3.5))
                y_proba_all = best_model.predict_proba(X_test)[:, 1]
                ax.hist(y_proba_all[y_test == 0], bins=40, alpha=0.6,
                        color=SUCCESS, label='Fidèles', density=True)
                ax.hist(y_proba_all[y_test == 1], bins=40, alpha=0.6,
                        color=ACCENT,   label='Churners', density=True)
                ax.axvline(x=0.5, color=PRIMARY, linestyle='--', linewidth=1.5, label='Seuil 0.5')
                ax.set_xlabel('Probabilité de churn prédite', fontsize=9)
                ax.set_ylabel('Densité', fontsize=9)
                ax.legend(fontsize=8)
                ax.set_facecolor(SURFACE)
                fig.patch.set_alpha(0)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            st.markdown('</div>', unsafe_allow_html=True)
 
        # Distribution par contract_type si disponible
        if 'contract_type' in df_full.columns:
            col3, col4 = st.columns(2)
            with col3:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Taux de churn par type de contrat</div>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(5, 3.2))
                ct = df_full.groupby('contract_type')['churn'].mean().sort_values(ascending=False)
                bars = ax.barh(ct.index, ct.values * 100,
                               color=[PRIMARY, WARNING, SUCCESS],
                               edgecolor='none', height=0.5)
            
                for bar, val in zip(bars, ct.values):
                    ax.text(val * 100 - 0.5, bar.get_y() + bar.get_height()/2,
                             f'{val*100:.1f}%', va='center', ha='right',
            fontsize=9, fontweight='bold', color='white')            
                #ax.set_xlabel('Taux de churn (%)', fontsize=9)
                ax.set_xlabel('Taux de churn (%)', fontsize=9)
                ax.set_xlim(0, 12)
                plt.subplots_adjust(left=0.2)
                ax.set_facecolor(SURFACE)
                plt.setp(ax.get_yticklabels(), color='white')
                fig.patch.set_alpha(0)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                st.markdown('</div>', unsafe_allow_html=True)
 
            with col4:
                if 'monthly_charges' in df_full.columns:
                    st.markdown('<div class="section-card">', unsafe_allow_html=True)
                    st.markdown('<div class="section-title">Charges mensuelles — Churners vs Fidèles</div>', unsafe_allow_html=True)
                    fig, ax = plt.subplots(figsize=(5, 3.2))
                    df_full[df_full['churn']==0]['monthly_charges'].hist(
                        ax=ax, bins=30, alpha=0.6, color=SUCCESS, density=True, label='Fidèles')
                    df_full[df_full['churn']==1]['monthly_charges'].hist(
                        ax=ax, bins=30, alpha=0.6, color=ACCENT,  density=True, label='Churners')
                    ax.set_xlabel('Charges mensuelles (€)', fontsize=9)
                    ax.set_ylabel('Densité', fontsize=9)
                    ax.legend(fontsize=8)
                    ax.set_facecolor(SURFACE)
                    fig.patch.set_alpha(0)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    st.markdown('</div>', unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────
#  PAGE 2 — PRÉDICTION CLIENT
# ─────────────────────────────────────────────
elif page == "🔮 Prédiction client":
 
    st.markdown("""
    <div class="hero">
        <h1>🔮 Simulateur de risque client</h1>
        <p>Renseignez le profil d'un client pour obtenir sa probabilité de churn en temps réel</p>
    </div>
    """, unsafe_allow_html=True)
 
    if preprocessor is None or best_model is None:
        st.error("Modèle ou preprocessor introuvable. Vérifiez le dossier models/.")
    else:
        # Récupérer les colonnes attendues
        num_cols = list(preprocessor.transformers_[0][2])
        cat_cols = list(preprocessor.transformers_[1][2])
 
        col_form, col_result = st.columns([1.2, 1])
 
        with col_form:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Profil client</div>', unsafe_allow_html=True)
 
            input_data = {}
 
            # Champs numériques
            num_defaults = {
                'age': (18, 80, 35),
                'tenure_months': (0, 120, 24),
                'monthly_charges': (10.0, 300.0, 65.0),
                'total_revenue': (0.0, 50000.0, 2000.0),
                'payment_failures': (0, 20, 0),
                'support_tickets': (0, 30, 1),
                'login_frequency': (0, 100, 15),
                'session_duration': (0.0, 300.0, 30.0),
                'nps_score': (0, 10, 7),
                'csat_score': (0, 10, 7),
            }
 
            labels = {
                'age': 'Âge',
                'tenure_months': 'Ancienneté (mois)',
                'monthly_charges': 'Charges mensuelles (€)',
                'total_revenue': 'Revenu total (€)',
                'payment_failures': 'Échecs de paiement',
                'support_tickets': 'Tickets support',
                'login_frequency': 'Connexions / mois',
                'session_duration': 'Durée session moy. (min)',
                'nps_score': 'NPS Score (0-10)',
                'csat_score': 'CSAT Score (0-10)',
            }
 
            c1, c2 = st.columns(2)
            num_items = [col for col in num_cols if col in num_defaults]
 
            for i, col in enumerate(num_items):
                target = c1 if i % 2 == 0 else c2
                mn, mx, default = num_defaults.get(col, (0, 100, 0))
                label = labels.get(col, col.replace('_', ' ').title())
                if isinstance(default, float):
                    input_data[col] = target.number_input(label, min_value=float(mn), max_value=float(mx), value=float(default), step=0.5)
                else:
                    input_data[col] = target.number_input(label, min_value=int(mn), max_value=int(mx), value=int(default))
 
            # Champs catégoriels
            cat_defaults = {
                'contract_type': ['Month-to-Month', 'One Year', 'Two Year'],
                'gender': ['Male', 'Female'],
                'payment_method': ['Credit Card', 'Bank Transfer', 'Electronic Check', 'Mailed Check'],
            }
 
            for col in cat_cols:
                if col in cat_defaults:
                    label = col.replace('_', ' ').title()
                    input_data[col] = st.selectbox(label, cat_defaults[col])
 
            # Remplir les colonnes restantes avec des valeurs par défaut
            for col in num_cols:
                if col not in input_data:
                    input_data[col] = 0
            for col in cat_cols:
                if col not in input_data:
                    input_data[col] = 'Unknown'
 
            predict_btn = st.button("🔮 Analyser ce client", type="primary", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
 
        with col_result:
            st.markdown('<div class="section-card" style="min-height:400px;">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Résultat de l\'analyse</div>', unsafe_allow_html=True)
 
            if predict_btn:
                input_df = pd.DataFrame([input_data])
                X_input  = preprocessor.transform(input_df)
                proba    = best_model.predict_proba(X_input)[0, 1]
                pred     = int(proba > 0.5)
 
                # Jauge visuelle
                if pred == 1:
                    color_gauge = ACCENT
                    badge_class = "pred-churn"
                    badge_text  = "⚠️ RISQUE DE CHURN"
                    msg = "Ce client présente un risque élevé de résiliation. Une action de rétention est recommandée."
                else:
                    color_gauge = SUCCESS
                    badge_class = "pred-stable"
                    badge_text  = "✅ CLIENT STABLE"
                    msg = "Ce client présente un faible risque de résiliation. Maintenir le niveau de service."
 
                st.markdown(f"""
                <div style="text-align:center; padding: 1rem 0;">
                    <div class="prob-text" style="color:{color_gauge};">{proba*100:.1f}%</div>
                    <div style="font-size:0.85rem; color:{NEUTRAL}; margin-bottom:1rem;">
                        Probabilité de churn
                    </div>
                    <span class="pred-badge {badge_class}">{badge_text}</span>
                </div>
                """, unsafe_allow_html=True)
 
                # Barre de risque
                fig, ax = plt.subplots(figsize=(5, 0.8))
                ax.barh([0], [1], color='#E5E7EB', height=0.5)
                ax.barh([0], [proba], color=color_gauge, height=0.5)
                ax.axvline(x=0.5, color=PRIMARY, linewidth=2, linestyle='--')
                ax.set_xlim(0, 1)
                ax.axis('off')
                fig.patch.set_alpha(0)
                st.pyplot(fig)
                plt.close()
 
                st.info(msg)
 
                # Facteurs de risque détectés
                st.markdown("**Facteurs détectés :**")
                risk_factors = []
                if input_data.get('payment_failures', 0) >= 2:
                    risk_factors.append(f"🔴 {input_data['payment_failures']} échec(s) de paiement")
                if input_data.get('nps_score', 10) <= 5:
                    risk_factors.append(f"🔴 NPS faible ({input_data['nps_score']}/10)")
                if input_data.get('support_tickets', 0) >= 3:
                    risk_factors.append(f"🟠 {input_data['support_tickets']} tickets support")
                if input_data.get('tenure_months', 99) <= 6:
                    risk_factors.append(f"🟠 Client récent ({input_data['tenure_months']} mois)")
                if input_data.get('login_frequency', 99) <= 5:
                    risk_factors.append("🟡 Faible fréquence de connexion")
                if not risk_factors:
                    risk_factors.append("✅ Aucun signal d'alerte majeur détecté")
                for f in risk_factors:
                    st.markdown(f"- {f}")
 
            else:
                st.markdown("""
                <div style="text-align:center; padding:3rem 1rem; color:#94A3B8;">
                    <div style="font-size:3rem;">🎯</div>
                    <div style="margin-top:0.5rem; font-size:0.9rem;">
                        Renseignez le profil client et cliquez sur Analyser
                    </div>
                </div>
                """, unsafe_allow_html=True)
 
            st.markdown('</div>', unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────
#  PAGE 3 — PERFORMANCE DES MODÈLES
# ─────────────────────────────────────────────
elif page == "📈 Performance des modèles":
 
    st.markdown("""
    <div class="hero">
        <h1>📈 Comparaison des modèles</h1>
        <p>Évaluation rigoureuse des 4 algorithmes sur le jeu de test — Métriques, ROC, Precision-Recall</p>
    </div>
    """, unsafe_allow_html=True)
 
    if not models:
        st.error("Aucun modèle trouvé dans le dossier models/.")
    else:
        # Tableau comparatif
        rows = []
        for name, model in models.items():
            y_pred  = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            rows.append({
                "Modèle"    : name,
                "Accuracy"  : round(accuracy_score(y_test, y_pred), 4),
                "Precision" : round(precision_score(y_test, y_pred, zero_division=0), 4),
                "Recall"    : round(recall_score(y_test, y_pred, zero_division=0), 4),
                "F1-Score"  : round(f1_score(y_test, y_pred, zero_division=0), 4),
                "ROC-AUC"   : round(roc_auc_score(y_test, y_proba), 4),
                "PR-AUC"    : round(average_precision_score(y_test, y_proba), 4),
            })
 
        df_results = pd.DataFrame(rows).set_index("Modèle")
 
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Tableau des performances</div>', unsafe_allow_html=True)
 
        def color_max(s):
            is_max = s == s.max()
            return ['background-color:#DCFCE7; font-weight:700' if v else '' for v in is_max]
 
        st.dataframe(
            df_results.style.apply(color_max, axis=0).format("{:.4f}"),
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
 
        # Courbes ROC + PR côte à côte
        col1, col2 = st.columns(2)
 
        colors_models = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
 
        with col1:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Courbes ROC</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5.5, 4))
            for (name, model), color in zip(models.items(), colors_models):
                y_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                auc = roc_auc_score(y_test, y_proba)
                ax.plot(fpr, tpr, color=color, linewidth=2, label=f'{name} ({auc:.3f})')
            ax.plot([0,1],[0,1],'k--', alpha=0.4, label='Aléatoire')
            ax.set_xlabel('FPR', fontsize=9)
            ax.set_ylabel('TPR (Recall)', fontsize=9)
            ax.legend(fontsize=7.5, loc='lower right')
            ax.set_facecolor(SURFACE)
            fig.patch.set_alpha(0)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.markdown('</div>', unsafe_allow_html=True)
 
        with col2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Courbes Precision-Recall</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5.5, 4))
            for (name, model), color in zip(models.items(), colors_models):
                y_proba = model.predict_proba(X_test)[:, 1]
                prec, rec, _ = precision_recall_curve(y_test, y_proba)
                ap = average_precision_score(y_test, y_proba)
                ax.plot(rec, prec, color=color, linewidth=2, label=f'{name} (AP={ap:.3f})')
            ax.axhline(y=y_test.mean(), color='black', linestyle='--', alpha=0.4,
                       label=f'Baseline ({y_test.mean():.2f})')
            ax.set_xlabel('Recall', fontsize=9)
            ax.set_ylabel('Precision', fontsize=9)
            ax.legend(fontsize=7.5, loc='upper right')
            ax.set_facecolor(SURFACE)
            fig.patch.set_alpha(0)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.markdown('</div>', unsafe_allow_html=True)
 
        # Matrices de confusion
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Matrices de confusion</div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
        if len(models) == 1:
            axes = [axes]
        for ax, ((name, model), color) in zip(axes, zip(models.items(), colors_models)):
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                        cmap=sns.light_palette(color, as_cmap=True),
                        xticklabels=['Reste', 'Part'],
                        yticklabels=['Reste', 'Part'],
                        linewidths=1, linecolor='white')
            ax.set_title(name, fontsize=10, fontweight='bold')
            ax.set_xlabel('Prédit', fontsize=9)
            ax.set_ylabel('Réel', fontsize=9)
        fig.patch.set_alpha(0)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────
#  PAGE 4 — VARIABLES INFLUENTES
# ─────────────────────────────────────────────
elif page == "🔍 Variables influentes":
 
    st.markdown("""
    <div class="hero">
        <h1>🔍 Variables influentes</h1>
        <p>Comprendre pourquoi le modèle prédit le churn — Feature Importance & Permutation Importance</p>
    </div>
    """, unsafe_allow_html=True)
 
    if best_model is None or not feature_names:
        st.error("Modèle ou features introuvables.")
    else:
        from sklearn.inspection import permutation_importance as perm_imp_fn
 
        tab1, tab2 = st.tabs(["📊 Feature Importance (XGBoost)", "🔄 Permutation Importance"])
 
        with tab1:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Importance native — Gain moyen sur tous les splits</div>', unsafe_allow_html=True)
 
            xgb_model = models.get("XGBoost") or models.get("XGBoost ".strip())
            if xgb_model is None:
                 st.error("Modèle XGBoost introuvable.")
            else:
             importances = xgb_model.feature_importances_
            feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})\
               .sort_values('Importance', ascending=False).head(20)
 
            fig, ax = plt.subplots(figsize=(8, 7))
            bars = ax.barh(feat_df['Feature'][::-1], feat_df['Importance'][::-1],
                           color=[ACCENT if i < 3 else PRIMARY if i < 8 else NEUTRAL
                                  for i in range(len(feat_df)-1, -1, -1)],
                           edgecolor='none', height=0.65)
            ax.set_xlabel('Importance (Gain)', fontsize=9)
            plt.setp(ax.get_yticklabels(), color='white')
            ax.xaxis.label.set_color('white')
            ax.tick_params(axis='x', colors='white')
            ax.set_facecolor(SURFACE)
            fig.patch.set_alpha(0)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
 
            st.markdown("**Lecture :** 🔴 Top 3 variables · 🔵 Top 8 · ⬜ Autres")
 
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Top 10 variables :**")
                st.dataframe(
                    feat_df.head(10)[['Feature','Importance']].reset_index(drop=True)
                    .style.format({'Importance': '{:.4f}'}),
                    use_container_width=True
                )
            with col2:
                st.markdown("**Interprétation métier :**")
                st.markdown("""
                | Variable | Signal |
                |---|---|
                | `payment_failures` | Friction financière critique |
                | `nps_score` | Satisfaction globale |
                | `tenure_months` | Fidélité installée |
                | `support_tickets` | Insatisfaction produit |
                | `login_frequency` | Engagement comportemental |
                """)
            st.markdown('</div>', unsafe_allow_html=True)
 
        with tab2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Permutation Importance — Perte réelle de ROC-AUC</div>', unsafe_allow_html=True)
 
            with st.spinner("Calcul en cours (10 répétitions)..."):
                perm = perm_imp_fn(
                    best_model, X_test, y_test,
                    scoring='roc_auc',
                    n_repeats=10,
                    random_state=42,
                    n_jobs=-1
                )
 
            perm_df = pd.DataFrame({
                'Feature': feature_names,
                'mean': perm.importances_mean,
                'std' : perm.importances_std
            }).sort_values('mean', ascending=False).head(20)
 
            fig, ax = plt.subplots(figsize=(8, 7))
            ax.barh(perm_df['Feature'][::-1], perm_df['mean'][::-1],
                    xerr=perm_df['std'][::-1],
                    color=PRIMARY, alpha=0.85, edgecolor='none', height=0.6,
                    error_kw=dict(ecolor=ACCENT, capsize=3))
            ax.set_xlabel('Perte de ROC-AUC (± écart-type)', fontsize=9)
            plt.setp(ax.get_yticklabels(), color='white')
            ax.xaxis.label.set_color('white')
            ax.tick_params(axis='x', colors='white')
            ax.set_facecolor(SURFACE)
            fig.patch.set_alpha(0)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
 
            st.info("La Permutation Importance est agnostique au modèle : elle mesure l'impact réel de chaque variable sur la performance, en la mélangeant aléatoirement. Plus la perte est grande, plus la variable est indispensable.")
            st.markdown('</div>', unsafe_allow_html=True)