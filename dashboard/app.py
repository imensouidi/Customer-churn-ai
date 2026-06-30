import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
PRIMARY  = "#1B2B4B"
ACCENT   = "#E63946"
SUCCESS  = "#2DC653"
WARNING  = "#F4A261"
NEUTRAL  = "#8D99AE"
SURFACE  = "#F8F9FA"
DARK     = "#0D1B2A"
 
# ─────────────────────────────────────────────
#  CSS GLOBAL
# ─────────────────────────────────────────────
st.markdown(f"""
<style>
  html, body, [class*="css"] {{
      font-family: 'Inter', sans-serif;
      background-color: #F0F2F6;
  }}
  [data-testid="stSidebar"] {{
      background-color: {DARK};
  }}
  [data-testid="stSidebar"] * {{
      color: #E2E8F0 !important;
  }}
  .hero {{
      background: linear-gradient(135deg, {PRIMARY} 0%, #2E4A7A 100%);
      border-radius: 16px;
      padding: 2rem 2.5rem;
      margin-bottom: 1.5rem;
      color: white;
  }}
  .hero h1 {{
      font-size: 2rem;
      font-weight: 700;
      margin: 0;
  }}
  .hero p {{
      font-size: 0.95rem;
      color: #A8C0E0;
      margin: 0.4rem 0 0 0;
  }}
  .section-card {{
      background: white;
      border-radius: 12px;
      padding: 1.4rem 1.6rem;
      box-shadow: 0 1px 4px rgba(0,0,0,0.06);
      margin-bottom: 1rem;
  }}
  .section-title {{
      font-size: 1rem;
      font-weight: 700;
      color: {PRIMARY};
      margin-bottom: 1rem;
      padding-bottom: 0.5rem;
      border-bottom: 2px solid #EEF2F7;
  }}
  .pred-badge {{
      display: inline-block;
      padding: 0.5rem 1.2rem;
      border-radius: 50px;
      font-weight: 700;
      font-size: 1rem;
  }}
  .pred-churn  {{ background: #FEE2E2; color: {ACCENT}; }}
  .pred-stable {{ background: #DCFCE7; color: #15803D; }}
  .prob-text {{
      font-size: 3rem;
      font-weight: 700;
      text-align: center;
  }}
  #MainMenu, footer, header {{ visibility: hidden; }}
  .block-container {{ padding-top: 1.5rem; }}
</style>
""", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────
#  CHARGEMENT
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
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
    return pd.read_csv(path) if os.path.exists(path) else None
 
@st.cache_resource
def get_feature_names(_preprocessor):
    if _preprocessor is None:
        return []
    num_features = list(_preprocessor.transformers_[0][2])
    cat_features = _preprocessor.transformers_[1][1]['onehot'].get_feature_names_out(
        _preprocessor.transformers_[1][2]
    ).tolist()
    return num_features + cat_features
 
models        = load_models()
preprocessor  = load_preprocessor()
X_test, y_test = load_test_data()
df_full       = load_full_data()
feature_names = get_feature_names(preprocessor)
best_model    = models.get("Random Forest", list(models.values())[0] if models else None)
 
# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 1.5rem 0;">
        <div style="font-size:1.3rem; font-weight:700; color:white;">🛡️ ChurnGuard</div>
        <div style="font-size:0.75rem; color:#64748B; margin-top:0.2rem;">Plateforme de rétention client</div>
    </div>
    """, unsafe_allow_html=True)
 
    page = st.radio(
        "Navigation",
        ["📊 Vue d'ensemble", "🔮 Prédiction client",
         "🔍 Variables influentes"],
        label_visibility="collapsed"
    )
 
 
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
 
    if df_full is not None and best_model is not None:
        y_proba_all    = best_model.predict_proba(X_test)[:, 1]
        n_at_risk      = int((y_proba_all > 0.5).sum())
        pct_risk       = n_at_risk / len(y_test) * 100
        total_churn    = int(y_test.sum())
        revenue_at_risk = n_at_risk * 85
        auc_score      = roc_auc_score(y_test, y_proba_all)
 
        # KPIs avec st.metric (natif Streamlit → fiable)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🚨 Clients à risque", f"{n_at_risk:,}", f"{pct_risk:.1f}% du portefeuille")
        c2.metric("💰 Revenu mensuel à risque", f"{revenue_at_risk:,.0f} €", "Estimation charges moyennes")
        c3.metric("📊 ROC-AUC (Random Forest)", f"{auc_score:.3f}", "Modèle final sélectionné")
        c4.metric("👥 Churners réels (test)", f"{total_churn:,}", f"{total_churn/len(y_test)*100:.1f}% du jeu de test")
 
        st.markdown("---")
 
        # Graphiques
        col1, col2 = st.columns(2)
 
        with col1:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Répartition du churn</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 3.5))
            counts = df_full['churn'].value_counts()
            wedges, texts, autotexts = ax.pie(
                counts,
                labels=['Fidèles', 'Churners'],
                colors=[SUCCESS, ACCENT],
                autopct='%1.1f%%',
                startangle=90,
                pctdistance=0.75,
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
            st.markdown('<div class="section-title">Distribution des probabilités de churn</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.hist(y_proba_all[y_test == 0], bins=40, alpha=0.6,
                    color=SUCCESS, label='Fidèles', density=True)
            ax.hist(y_proba_all[y_test == 1], bins=40, alpha=0.6,
                    color=ACCENT, label='Churners', density=True)
            ax.axvline(x=0.5, color=PRIMARY, linestyle='--', linewidth=1.5, label='Seuil 0.5')
            ax.set_xlabel('Probabilité de churn prédite', fontsize=10, color='black')
            ax.set_ylabel('Densité', fontsize=10, color='black')
            ax.tick_params(colors='black')
            ax.legend(fontsize=9)
            ax.set_facecolor(SURFACE)
            fig.patch.set_alpha(0)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.markdown('</div>', unsafe_allow_html=True)
 
        # ── Encart Impact Business ──
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">💼 Impact business estimé — Stratégie de rétention</div>', unsafe_allow_html=True)
 
        # Hypothèses métier
        taux_retention = 0.30   # 30% des clients contactés sont retenus
        valeur_client  = 85 * 12  # 85€/mois × 12 mois = 1020€/an
 
        clients_retenus    = int(n_at_risk * taux_retention)
        revenus_sauves     = clients_retenus * valeur_client
        cout_campagne      = n_at_risk * 15  # 15€ par contact marketing
 
        bi1, bi2, bi3, bi4 = st.columns(4)
        bi1.metric("🎯 Clients à contacter", f"{n_at_risk:,}", "Détectés par le modèle")
        bi2.metric("✅ Clients retenus estimés", f"{clients_retenus:,}", f"Taux de rétention : 30%")
        bi3.metric("💰 Revenus annuels sauvés", f"{revenus_sauves:,.0f} €", "Valeur vie client × retenus")
        bi4.metric("📉 ROI campagne estimé", f"{((revenus_sauves - cout_campagne) / cout_campagne * 100):.0f}%", f"Coût campagne : {cout_campagne:,.0f} €")
 
        st.caption("⚠️ Estimations basées sur des hypothèses métier : taux de rétention 30%, valeur client annuelle 1 020 €, coût contact 15 €.")
        st.markdown('</div>', unsafe_allow_html=True)
 
        st.markdown("---")
 
        # Graphiques ligne 2
        col3, col4 = st.columns(2)
 
        with col3:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Taux de churn par type de contrat</div>', unsafe_allow_html=True)
            if 'contract_type' in df_full.columns:
                fig, ax = plt.subplots(figsize=(5, 3))
                ct = df_full.groupby('contract_type')['churn'].mean().sort_values(ascending=True)
                colors_bar = [SUCCESS, WARNING, ACCENT]
                bars = ax.barh(ct.index, ct.values * 100,
                               color=colors_bar[:len(ct)],
                               edgecolor='none', height=0.5)
                for bar, val in zip(bars, ct.values):
                    ax.text(bar.get_width() / 2,
                            bar.get_y() + bar.get_height() / 2,
                            f'{val*100:.1f}%',
                            va='center', ha='center',
                            fontsize=11, fontweight='bold', color='white')
                ax.set_xlabel('Taux de churn (%)', fontsize=10, color='black')
                ax.tick_params(axis='x', colors='black')
                ax.tick_params(axis='y', colors='black')
                ax.set_xlim(0, 15)
                ax.set_facecolor(SURFACE)
                fig.patch.set_alpha(0)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            st.markdown('</div>', unsafe_allow_html=True)
 
        with col4:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Taux de churn par segment client</div>', unsafe_allow_html=True)
            if 'customer_segment' in df_full.columns:
                fig, ax = plt.subplots(figsize=(5, 3))
                seg = df_full.groupby('customer_segment')['churn'].mean().sort_values(ascending=True)
                bars = ax.barh(seg.index, seg.values * 100,
                               color=[SUCCESS, WARNING, ACCENT][:len(seg)],
                               edgecolor='none', height=0.5)
                for bar, val in zip(bars, seg.values):
                    ax.text(bar.get_width() / 2,
                            bar.get_y() + bar.get_height() / 2,
                            f'{val*100:.1f}%',
                            va='center', ha='center',
                            fontsize=11, fontweight='bold', color='white')
                ax.set_xlabel('Taux de churn (%)', fontsize=10, color='black')
                ax.tick_params(axis='x', colors='black')
                ax.tick_params(axis='y', colors='black')
                ax.set_xlim(0, 15)
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
        st.error("Modèle ou preprocessor introuvable.")
    else:
        num_cols = list(preprocessor.transformers_[0][2])
        cat_cols = list(preprocessor.transformers_[1][2])
 
        col_form, col_result = st.columns([1.2, 1])
 
        with col_form:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Profil client</div>', unsafe_allow_html=True)
 
            input_data = {}
 
            num_defaults = {
                'age': (18, 80, 35),
                'tenure_months': (0, 120, 24),
                'monthly_fee': (10.0, 300.0, 65.0),
                'total_revenue': (0.0, 50000.0, 2000.0),
                'payment_failures': (0, 20, 0),
                'support_tickets': (0, 30, 1),
                'monthly_logins': (0, 100, 15),
                'avg_session_time': (0.0, 300.0, 30.0),
                'nps_score': (-100, 100, 20),
                'csat_score': (1, 5, 4),
            }
 
            labels = {
                'age': 'Âge',
                'tenure_months': 'Ancienneté (mois)',
                'monthly_fee': 'Charges mensuelles (€)',
                'total_revenue': 'Revenu total (€)',
                'payment_failures': 'Échecs de paiement',
                'support_tickets': 'Tickets support',
                'monthly_logins': 'Connexions / mois',
                'avg_session_time': 'Durée session (min)',
                'nps_score': 'NPS Score (-100 à 100)',
                'csat_score': 'CSAT Score (1 à 5)',
            }
 
            c1, c2 = st.columns(2)
            num_items = [col for col in num_cols if col in num_defaults]
 
            for i, col in enumerate(num_items):
                target = c1 if i % 2 == 0 else c2
                mn, mx, default = num_defaults.get(col, (0, 100, 0))
                label = labels.get(col, col.replace('_', ' ').title())
                if isinstance(default, float):
                    input_data[col] = target.number_input(
                        label, min_value=float(mn), max_value=float(mx),
                        value=float(default), step=0.5)
                else:
                    input_data[col] = target.number_input(
                        label, min_value=int(mn), max_value=int(mx), value=int(default))
 
            cat_defaults = {
                'contract_type': ['Month-to-Month', 'One Year', 'Two Year'],
                'gender': ['Male', 'Female'],
                'payment_method': ['Credit Card', 'Bank Transfer', 'Electronic Check', 'Mailed Check'],
                'customer_segment': ['SME', 'Individual', 'Enterprise'],
                'signup_channel': ['Web', 'Mobile', 'Referral'],
            }
 
            for col in cat_cols:
                if col in cat_defaults:
                    label = col.replace('_', ' ').title()
                    input_data[col] = st.selectbox(label, cat_defaults[col])
 
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
                <div style="text-align:center; padding:1rem 0;">
                    <div class="prob-text" style="color:{color_gauge};">{proba*100:.1f}%</div>
                    <div style="font-size:0.85rem; color:{NEUTRAL}; margin-bottom:1rem;">
                        Probabilité de churn
                    </div>
                    <span class="pred-badge {badge_class}">{badge_text}</span>
                </div>
                """, unsafe_allow_html=True)
 
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
 
                st.markdown("**🔎 Facteurs détectés :**")
                risk_factors = []
                if input_data.get('payment_failures', 0) >= 2:
                    risk_factors.append(f"🔴 {input_data['payment_failures']} échec(s) de paiement")
                if input_data.get('nps_score', 10) <= 0:
                    risk_factors.append(f"🔴 NPS négatif ({input_data['nps_score']})")
                if input_data.get('csat_score', 5) <= 2:
                    risk_factors.append(f"🔴 CSAT très faible ({input_data['csat_score']}/5)")
                if input_data.get('support_tickets', 0) >= 3:
                    risk_factors.append(f"🟠 {input_data['support_tickets']} tickets support")
                if input_data.get('tenure_months', 99) <= 6:
                    risk_factors.append(f"🟠 Client récent ({input_data['tenure_months']} mois)")
                if input_data.get('monthly_logins', 99) <= 5:
                    risk_factors.append("🟡 Faible fréquence de connexion")
                if not risk_factors:
                    risk_factors.append("✅ Aucun signal d'alerte majeur détecté")
                for f in risk_factors:
                    st.markdown(f"- {f}")
 
                # ── Recommandations actions CRM ──
                st.markdown("---")
                st.markdown("**📋 Actions recommandées pour l'équipe CRM :**")
 
                if pred == 1:
                    urgence = "🔴 URGENT" if proba > 0.75 else "🟠 PRIORITAIRE"
 
                    actions = []
 
                    if input_data.get('payment_failures', 0) >= 2:
                        actions.append("💳 Contacter le service facturation — résoudre les échecs de paiement")
                    if input_data.get('csat_score', 5) <= 2:
                        actions.append("📞 Appel de satisfaction — identifier les problèmes et proposer une solution")
                    if input_data.get('tenure_months', 99) <= 6:
                        actions.append("🎁 Offrir une remise de fidélité de 10% sur le prochain mois")
                    if input_data.get('support_tickets', 0) >= 3:
                        actions.append("🛠️ Escalader vers le support premium — résoudre les problèmes en attente")
                    if input_data.get('monthly_logins', 99) <= 5:
                        actions.append("📧 Envoyer une campagne d'engagement — tutoriels et nouvelles fonctionnalités")
                    if not actions:
                        actions.append("📞 Contacter le client pour comprendre ses besoins")
                        actions.append("🎁 Proposer une offre personnalisée de rétention")
 
                    st.error(f"{urgence} — Probabilité de churn : {proba*100:.1f}%")
                    for action in actions:
                        st.markdown(f"→ {action}")
 
                    st.markdown(f"""
                    <div style="background:#FEF3C7; padding:0.8rem 1rem; border-radius:8px;
                                border-left:4px solid #F59E0B; margin-top:0.8rem;">
                        <b>💼 Valeur à risque estimée :</b> {input_data.get('total_revenue', 0) * 0.3:.0f} €
                        (30% de la valeur client si non retenu)
                    </div>
                    """, unsafe_allow_html=True)
 
                else:
                    st.success("✅ Aucune action urgente requise")
                    st.markdown("→ 📊 Maintenir le niveau de service actuel")
                    st.markdown("→ 🔄 Surveiller le prochain cycle de facturation")
                    st.markdown("→ 📧 Inclure dans les campagnes de fidélisation standard")
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
 
        def color_min(s):
            is_min = s == s.min()
            return ['background-color:#FEE2E2;' if v else '' for v in is_min]
 
        st.dataframe(
            df_results.style
                .apply(color_max, axis=0)
                .apply(color_min, axis=0)
                .format("{:.4f}"),
            use_container_width=True
        )
        st.caption("🟢 Vert = meilleure valeur · 🔴 Rouge = moins bonne valeur")
        st.markdown('</div>', unsafe_allow_html=True)
 
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
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Aléatoire')
            ax.set_xlabel('Taux de faux positifs (FPR)', fontsize=9, color='black')
            ax.set_ylabel('Taux de vrais positifs (Recall)', fontsize=9, color='black')
            ax.tick_params(colors='black')
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
            ax.set_xlabel('Recall', fontsize=9, color='black')
            ax.set_ylabel('Precision', fontsize=9, color='black')
            ax.tick_params(colors='black')
            ax.legend(fontsize=7.5, loc='upper right')
            ax.set_facecolor(SURFACE)
            fig.patch.set_alpha(0)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.markdown('</div>', unsafe_allow_html=True)
 
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
            ax.set_title(name, fontsize=10, fontweight='bold', color='black')
            ax.set_xlabel('Prédit', fontsize=9, color='black')
            ax.set_ylabel('Réel', fontsize=9, color='black')
            ax.tick_params(colors='black')
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
        tab1, tab2 = st.tabs(["📊 Feature Importance (Random Forest)", "🔄 Permutation Importance"])
 
        with tab1:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Importance native — Random Forest</div>', unsafe_allow_html=True)
 
            importances = best_model.feature_importances_
            feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})\
                        .sort_values('Importance', ascending=False).head(20)
 
            fig, ax = plt.subplots(figsize=(8, 7))
            colors_fi = [ACCENT if i < 3 else PRIMARY if i < 8 else NEUTRAL
                         for i in range(len(feat_df) - 1, -1, -1)]
            ax.barh(feat_df['Feature'][::-1], feat_df['Importance'][::-1],
                    color=colors_fi, edgecolor='none', height=0.65)
            ax.set_xlabel('Importance (Gain moyen)', fontsize=10, color='black')
            ax.tick_params(axis='x', colors='black')
            ax.tick_params(axis='y', colors='black')
            ax.set_facecolor(SURFACE)
            fig.patch.set_alpha(0)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
 
            st.markdown("**Lecture :** 🔴 Top 3 variables · 🔵 Top 4-8 · ⬜ Autres")
 
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Top 10 variables :**")
                st.dataframe(
                    feat_df.head(10)[['Feature', 'Importance']].reset_index(drop=True)
                    .style.format({'Importance': '{:.4f}'}),
                    use_container_width=True
                )
            with col2:
                st.markdown("**Interprétation métier :**")
                st.markdown("""
                | Variable | Signal métier |
                |---|---|
                | `csat_score` | Satisfaction globale client |
                | `payment_failures` | Friction financière critique |
                | `tenure_months` | Fidélité installée |
                | `monthly_logins` | Engagement comportemental |
                | `support_tickets` | Insatisfaction produit |
                """)
            st.markdown('</div>', unsafe_allow_html=True)
 
        with tab2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Permutation Importance — Perte réelle de ROC-AUC</div>', unsafe_allow_html=True)
 
            with st.spinner("Calcul en cours (10 répétitions)..."):
                from sklearn.metrics import roc_auc_score as ras
                y_proba_base = best_model.predict_proba(X_test)[:, 1]
                base_score   = ras(y_test, y_proba_base)
                importances_list = []
                for i in range(X_test.shape[1]):
                    scores = []
                    for _ in range(10):
                        X_perm = X_test.copy()
                        np.random.shuffle(X_perm[:, i])
                        score = ras(y_test, best_model.predict_proba(X_perm)[:, 1])
                        scores.append(base_score - score)
                    importances_list.append({
                        'mean': np.mean(scores),
                        'std':  np.std(scores)
                    })
 
            perm_df = pd.DataFrame({
                'Feature': feature_names,
                'mean': [x['mean'] for x in importances_list],
                'std':  [x['std']  for x in importances_list]
            }).sort_values('mean', ascending=False).head(20)
 
            fig, ax = plt.subplots(figsize=(8, 7))
            ax.barh(perm_df['Feature'][::-1], perm_df['mean'][::-1],
                    xerr=perm_df['std'][::-1],
                    color=PRIMARY, alpha=0.85, edgecolor='none', height=0.6,
                    error_kw=dict(ecolor=ACCENT, capsize=3))
            ax.set_xlabel('Perte de ROC-AUC (± écart-type)', fontsize=10, color='black')
            ax.tick_params(axis='x', colors='black')
            ax.tick_params(axis='y', colors='black')
            ax.set_facecolor(SURFACE)
            fig.patch.set_alpha(0)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
 
            st.info("Plus la barre est longue → plus la variable est indispensable au modèle.")
            st.markdown('</div>', unsafe_allow_html=True)
 
        # ── Encart Stratégie d'intégration IA ──
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🏢 Stratégie d\'intégration de l\'IA dans le processus métier (C4.1)</div>', unsafe_allow_html=True)
 
        col_a, col_b, col_c = st.columns(3)
 
        with col_a:
            st.markdown("""
            **📥 Entrée — Données client**
            - Comportement d'usage (logins, sessions)
            - Historique de paiement
            - Satisfaction (CSAT, NPS)
            - Tickets support
            - Ancienneté et contrat
            """)
 
        with col_b:
            st.markdown("""
            **🤖 Traitement — Modèle IA**
            - Random Forest + RUS
            - Pipeline sklearn anti-leakage
            - Recall : 0.83 → détecte 83% des churners
            - ROC-AUC : 0.80
            - Seuil décision : 0.5
            """)
 
        with col_c:
            st.markdown("""
            **📤 Sortie — Décision CRM**
            - Score de risque (0% → 100%)
            - Classification (stable / à risque)
            - Facteurs de risque identifiés
            - Actions de rétention recommandées
            - Estimation de la valeur à risque
            """)
 
        st.markdown("---")
        st.markdown("""
        **🔄 Intégration dans le processus métier :**
 
        `Données CRM` → `Pipeline preprocessing` → `Random Forest` → `Score de churn` → `Équipe marketing` → `Action de rétention` → `Suivi des résultats`
 
        Ce système transforme des données brutes en **décisions opérationnelles concrètes** pour les équipes CRM,
        permettant de prioriser les actions de rétention et d'optimiser le budget marketing.
        """)
        st.markdown('</div>', unsafe_allow_html=True)