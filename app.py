"""
Simulateur de mobilité décarbonée
Nemotron × EMP × Scénarios fiscaux
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG PAGE
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Simulateur mobilité décarbonée",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Police + fond */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    /* Titre principal */
    .main-title {
        font-size: 1.6rem; font-weight: 700;
        color: #1a1a2e; letter-spacing: -0.5px;
        border-left: 4px solid #00b4d8;
        padding-left: 12px; margin-bottom: 0.2rem;
    }
    .main-subtitle {
        color: #555; font-size: 0.9rem; margin-bottom: 1.5rem;
        padding-left: 16px;
    }

    /* Cartes métriques custom */
    .metric-card {
        background: #f8faff;
        border: 1px solid #e0e8f5;
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.9rem; font-weight: 700;
        color: #0077b6; line-height: 1.1;
    }
    .metric-delta-pos { color: #2d9e6b; font-size: 0.8rem; }
    .metric-delta-neg { color: #e05252; font-size: 0.8rem; }
    .metric-label { color: #666; font-size: 0.78rem; margin-top: 2px; }

    /* Section headers */
    .section-header {
        font-size: 1rem; font-weight: 600;
        color: #1a1a2e; border-bottom: 2px solid #00b4d8;
        padding-bottom: 4px; margin: 1.2rem 0 0.8rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #1a1a2e;
    }
    section[data-testid="stSidebar"] * {
        color: #e8eaf6 !important;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label {
        color: #90caf9 !important;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .sidebar-section {
        border-top: 1px solid #2d3561;
        margin-top: 12px; padding-top: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def charger_donnees(dossier):
    """Charge les fichiers Parquet exportés par le notebook Colab."""
    data = {}
    fichiers = {
        'dept':     'agg_departement.parquet',
        'csp':      'agg_csp.parquet',
        'zone':     'agg_zone.parquet',
        'rte':      'profil_rte.parquet',
        'personas': 'personas_enrichis.parquet',
    }
    for cle, nom in fichiers.items():
        chemin = os.path.join(dossier, nom)
        if os.path.exists(chemin):
            data[cle] = pd.read_parquet(chemin)

    params_path = os.path.join(dossier, 'params_run.json')
    if os.path.exists(params_path):
        with open(params_path) as f:
            data['params'] = json.load(f)

    return data


def generer_donnees_demo():
    """Données synthétiques si aucun fichier n'est uploadé."""
    np.random.seed(42)
    depts = [
        'Paris', 'Rhône', 'Nord', 'Gironde', 'Bouches-du-Rhône',
        'Loire-Atlantique', 'Haute-Garonne', 'Isère', 'Pas-de-Calais',
        'Hérault', 'Var', 'Ille-et-Vilaine', 'Bas-Rhin', 'Loire',
        'Finistère', 'Creuse', 'Lozère', 'Cantal', 'Meuse', 'Corrèze'
    ]
    n = len(depts)
    dept = pd.DataFrame({
        'departement':      depts,
        'nb_personas':      np.random.randint(500, 5000, n),
        'pct_ve':           np.random.uniform(0.15, 0.50, n),
        'pct_l6l7':         np.random.uniform(0.02, 0.10, n),
        'pct_velo':         np.random.uniform(0.01, 0.08, n),
        'pct_thermique':    np.random.uniform(0.35, 0.70, n),
        'cout_moyen_eur':   np.random.uniform(80, 280, n),
        'co2_evite_kg_mois':np.random.uniform(5000, 50000, n),
        'pct_report_tc':    np.random.uniform(0.02, 0.15, n),
        'pct_report_veli':  np.random.uniform(0.01, 0.08, n),
        'pct_report_velo':  np.random.uniform(0.005, 0.04, n),
    })

    csps = [
        'Cadres et professions intellectuelles supérieures',
        'Professions intermédiaires', 'Employés', 'Ouvriers',
        'Artisans, commerçants, chefs d\'entreprise',
        'Retraités', 'Autres sans activité professionnelle'
    ]
    csp = pd.DataFrame({
        'occupation':        csps,
        'nb_personas':       [8000, 12000, 15000, 10000, 5000, 9000, 4000],
        'pct_ve':            [0.45, 0.32, 0.22, 0.18, 0.28, 0.20, 0.12],
        'pct_l6l7':          [0.04, 0.06, 0.05, 0.04, 0.09, 0.03, 0.02],
        'dist_moyenne_km':   [32, 26, 20, 22, 38, 14, 11],
        'cout_moyen_eur':    [185, 145, 125, 118, 165, 90, 68],
        'sensibilite_moyenne':[1.8, 2.6, 3.4, 4.1, 2.3, 3.2, 4.6],
        'pct_bascule':       [0.03, 0.06, 0.10, 0.13, 0.05, 0.08, 0.16],
    })

    zone = pd.DataFrame({
        'type_zone': ['urbain']*4 + ['periurbain']*4 + ['rural']*4,
        'mode': ['vp_thermique','vp_electrique','tc','velo'] * 3,
        'part': [0.30, 0.35, 0.25, 0.10,
                 0.45, 0.30, 0.10, 0.15,
                 0.55, 0.20, 0.05, 0.20]
    })

    rte = pd.DataFrame({
        'heure':         range(24),
        'solaire_ete':   [0,0,0,0,0,0.02,0.08,0.18,0.38,0.58,0.75,0.88,
                          0.95,1.0,0.95,0.82,0.62,0.40,0.18,0.05,0.01,0,0,0],
        'solaire_hiver': [0,0,0,0,0,0,0.01,0.06,0.18,0.38,0.55,0.65,
                          0.68,0.65,0.55,0.38,0.15,0.02,0,0,0,0,0,0],
        'co2_g_kwh':     [18,15,12,10,10,12,15,20,45,80,60,50,
                          48,50,55,70,95,180,280,350,200,80,40,22]
    })

    params = {
        'horizon': '2035', 'scenario_ve': 'tendanciel',
        'scenario_l6l7': 'moyen', 'scenario_velo': 'faible',
        'scenario_tarif': 'combine', 'taxe_km_cents': 3.0,
        'n_personas': 50000, 'part_ve': 0.30,
        'part_l6l7': 0.05, 'part_velo': 0.02,
        'perte_ticpe_ratio': 0.35
    }
    return {'dept': dept, 'csp': csp, 'zone': zone,
            'rte': rte, 'params': params}


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — UPLOAD + PARAMÈTRES
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚡ Simulateur mobilité")
    st.markdown("*Nemotron × EMP × Fiscalité*")

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("**📂 Données**")
    uploaded = st.file_uploader(
        "Charger les fichiers Parquet (Colab)",
        type=['parquet'],
        accept_multiple_files=True,
        help="Dépose les fichiers exportés par le notebook Colab"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Chargement données
    if uploaded:
        import tempfile
        tmpdir = tempfile.mkdtemp()
        for f in uploaded:
            with open(os.path.join(tmpdir, f.name), 'wb') as out:
                out.write(f.read())
        data = charger_donnees(tmpdir)
        # Fallback pour chaque clé manquante (fichier non uploadé)
        demo = generer_donnees_demo()
        for cle in ['dept', 'csp', 'zone', 'rte', 'params']:
            if cle not in data:
                data[cle] = demo[cle]
        st.success(f"✅ {len(uploaded)} fichiers chargés")
        # Signaler les fichiers manquants
        noms = [f.name for f in uploaded]
        attendus = {
            'agg_departement': 'dept',
            'agg_csp': 'csp',
            'agg_zone': 'zone',
            'profil_rte': 'rte',
        }
        manquants = [k for k in attendus if not any(k in n for n in noms)]
        if manquants:
            st.warning(f"⚠️ Non trouvés (démo) : {', '.join(manquants)}")
    else:
        data = generer_donnees_demo()
        st.info("📊 Mode démo — charge tes fichiers Parquet pour les vraies données")

    # ── SCÉNARIOS ────────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("**🗓 Horizon**")
    horizon = st.radio("", ["2035", "2050"], horizontal=True,
                       index=0 if data.get('params', {}).get('horizon', '2035') == '2035' else 1)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("**🚗 Électrification du parc**")
    scenario_ve = st.selectbox(
        "Scénario VP électriques",
        ["tendanciel", "accéléré", "lent"],
        index=0
    )
    PART_VE = {'tendanciel': {'2035': 0.30, '2050': 0.70},
               'accéléré':   {'2035': 0.50, '2050': 0.90},
               'lent':       {'2035': 0.18, '2050': 0.50}}
    part_ve = PART_VE[scenario_ve][horizon]

    part_l6l7 = st.select_slider(
        "Quadricycles L6/L7 (% du parc)",
        options=[0.02, 0.05, 0.10],
        value=0.05,
        format_func=lambda x: f"{x:.0%}"
    )
    part_velo = st.select_slider(
        "Vélo (% des flux)",
        options=[0.02, 0.05, 0.10],
        value=0.02,
        format_func=lambda x: f"{x:.0%}"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("**⚡ Tarification recharge**")
    scenario_tarif = st.selectbox(
        "Scénario tarifaire",
        ["Actuel (HP/HC)", "Pic solaire réduit", "Combiné solaire + nuit"],
        index=2
    )
    TARIFS_LABEL = {
        "Actuel (HP/HC)":           {'HC': 0.175, 'HP': 0.255, 'solaire': 0.255},
        "Pic solaire réduit":       {'HC': 0.175, 'HP': 0.255, 'solaire': 0.12},
        "Combiné solaire + nuit":   {'HC': 0.140, 'HP': 0.255, 'solaire': 0.10},
    }
    tarif_actif = TARIFS_LABEL[scenario_tarif]
    st.caption(f"HC : {tarif_actif['HC']:.3f} €/kWh  |  "
               f"Solaire : {tarif_actif['solaire']:.3f} €/kWh")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("**💰 Taxe kilométrique**")
    taxe_km = st.slider(
        "Montant (c€/km)",
        min_value=0.0, max_value=10.0,
        value=3.0, step=0.5,
        help="TICPE actuelle ≈ 5 c€/km. La taxe compense la perte fiscale liée à l'électrification."
    )
    TICPE_REF = 5.0
    perte_ticpe = (part_ve + part_l6l7) * TICPE_REF
    st.caption(f"Perte TICPE estimée : {perte_ticpe:.2f} c€/km  |  "
               f"Compensation : {taxe_km/perte_ticpe:.0%}" if perte_ticpe > 0 else "")
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CALCULS DYNAMIQUES
# ─────────────────────────────────────────────────────────────────────────────

def recalculer(data, part_ve, part_l6l7, part_velo, tarif_actif, taxe_km, horizon):
    """Recalcule les métriques clés selon les paramètres sliders."""
    part_thermique = max(0, 1 - part_ve - part_l6l7 - part_velo)

    # Coût recharge moyen (mix profils)
    CONSO_VP   = 18.0   # kWh/100km
    CONSO_L6L7 =  9.0
    KM_MOIS    = 24 * 21.5  # distance moyenne × jours ouvrés

    prix_moy_ve   = (0.4 * tarif_actif['HC'] +
                     0.3 * tarif_actif['solaire'] +
                     0.3 * tarif_actif['HP'])
    cout_mois_ve  = KM_MOIS * CONSO_VP   / 100 * prix_moy_ve
    cout_mois_l6  = KM_MOIS * CONSO_L6L7 / 100 * prix_moy_ve

    # Coût thermique
    PRIX_CARB_CENTS = 12.6  # c€/km
    cout_mois_th  = KM_MOIS * PRIX_CARB_CENTS / 100

    # Taxe km (s'applique à tous)
    surcoût_taxe  = KM_MOIS * taxe_km / 100

    # Coût total moyen pondéré
    cout_total = (part_thermique * (cout_mois_th + surcoût_taxe) +
                  part_ve        * (cout_mois_ve  + surcoût_taxe) +
                  part_l6l7      * (cout_mois_l6  + surcoût_taxe) +
                  part_velo      * 0)

    # CO2 évité (kgCO2/an par conducteur électrique)
    CO2_KWPH_MOY_NUIT    = 15  # gCO2/kWh recharge nuit
    CO2_KWPH_MOY_SOLAIRE = 50
    CO2_THERMIQUE_G_KM   = 120
    co2_kwh_moy = 0.5 * CO2_KWPH_MOY_NUIT + 0.5 * CO2_KWPH_MOY_SOLAIRE
    co2_ve_km   = CONSO_VP   / 100 * co2_kwh_moy
    co2_l6_km   = CONSO_L6L7 / 100 * co2_kwh_moy
    km_an = KM_MOIS * 12

    co2_evite_an_par_ve  = (CO2_THERMIQUE_G_KM - co2_ve_km)  * km_an / 1e6  # tCO2
    co2_evite_an_par_l6  = (CO2_THERMIQUE_G_KM - co2_l6_km)  * km_an / 1e6

    N_CONDUCTEURS = 38_000_000
    co2_evite_total_mt = (part_ve   * N_CONDUCTEURS * co2_evite_an_par_ve +
                          part_l6l7 * N_CONDUCTEURS * co2_evite_an_par_l6) / 1e6  # MtCO2

    # Recettes taxe km
    km_france_an = N_CONDUCTEURS * km_an
    recettes_taxe_mrd = km_france_an * taxe_km / 100 / 1e9

    # Perte TICPE
    perte_ticpe_mrd = (part_ve + part_l6l7) * km_france_an * TICPE_REF / 100 / 1e9

    # Part bascule modale (proxy sensibilité)
    # Simplifié : part des personas dont coût > 12% revenu médian (1900€)
    REVENU_MEDIAN = 1900
    SEUIL_MOBILITE = REVENU_MEDIAN * 0.12
    pct_bascule = max(0, min(0.30,
        (cout_total - SEUIL_MOBILITE) / SEUIL_MOBILITE * 0.5))

    # Décomposition reports modaux
    report_tc   = pct_bascule * 0.45 * (1 - part_velo)
    report_veli = pct_bascule * 0.35 * (1 - part_velo)
    report_velo_sup = pct_bascule * 0.20

    return {
        'part_thermique':   part_thermique,
        'part_ve':          part_ve,
        'part_l6l7':        part_l6l7,
        'part_velo':        part_velo,
        'cout_total':       cout_total,
        'cout_mois_ve':     cout_mois_ve + surcoût_taxe,
        'cout_mois_th':     cout_mois_th + surcoût_taxe,
        'co2_evite_mt':     co2_evite_total_mt,
        'recettes_taxe':    recettes_taxe_mrd,
        'perte_ticpe':      perte_ticpe_mrd,
        'balance_fiscale':  recettes_taxe_mrd - perte_ticpe_mrd,
        'pct_bascule':      pct_bascule,
        'report_tc':        report_tc,
        'report_veli':      report_veli,
        'report_velo':      part_velo + report_velo_sup,
    }

metr = recalculer(data, part_ve, part_l6l7, part_velo,
                  tarif_actif, taxe_km, horizon)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE PRINCIPALE
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    f'<div class="main-title">Simulateur de mobilité décarbonée</div>'
    f'<div class="main-subtitle">Horizon {horizon} — '
    f'Scénario VE : {scenario_ve} | L6/L7 : {part_l6l7:.0%} | '
    f'Taxe : {taxe_km} c€/km | Tarif : {scenario_tarif}</div>',
    unsafe_allow_html=True
)

# ── MÉTRIQUES CLÉS ──────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Indicateurs clés</div>',
            unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

def card(col, value, label, delta=None, delta_pos=True):
    delta_html = ""
    if delta is not None:
        cls = "metric-delta-pos" if delta_pos else "metric-delta-neg"
        arrow = "▲" if delta_pos else "▼"
        delta_html = f'<div class="{cls}">{arrow} {delta}</div>'
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        {delta_html}
        <div class="metric-label">{label}</div>
    </div>""", unsafe_allow_html=True)

card(col1, f"{metr['part_ve']:.0%}",    "VP électriques dans le parc")
card(col2, f"{metr['co2_evite_mt']:.1f} Mt", "CO₂ évité/an (France)")
card(col3, f"{metr['cout_total']:.0f} €", "Coût mobilité/mois (moy.)")
card(col4,
     f"{metr['balance_fiscale']:+.1f} Md€",
     "Balance fiscale annuelle",
     delta=f"Taxe {metr['recettes_taxe']:.1f} Md€ / Perte TICPE {metr['perte_ticpe']:.1f} Md€",
     delta_pos=metr['balance_fiscale'] >= 0)
card(col5, f"{metr['pct_bascule']:.1%}", "Personas en arbitrage modal")

st.markdown("<br>", unsafe_allow_html=True)

# ── GRAPHIQUES ──────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🚗 Composition du parc",
    "⚡ Recharge & CO₂",
    "💰 Impact fiscal",
    "🚲 Reports modaux"
])

# ── TAB 1 : PARC ────────────────────────────────────────────────────────────
with tab1:
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown('<div class="section-header">Répartition du parc</div>',
                    unsafe_allow_html=True)

        parc_data = pd.DataFrame({
            'Mode': ['Thermique', 'VP électrique', 'L6/L7 électrique', 'Vélo'],
            'Part': [
                metr['part_thermique'],
                metr['part_ve'],
                metr['part_l6l7'],
                metr['part_velo']
            ]
        })

        try:
            import plotly.express as px
            fig = px.pie(
                parc_data, values='Part', names='Mode',
                color_discrete_sequence=['#e05252', '#00b4d8', '#0077b6', '#2d9e6b'],
                hole=0.45
            )
            fig.update_traces(textposition='outside', textinfo='percent+label')
            fig.update_layout(
                showlegend=False, margin=dict(t=20, b=20, l=20, r=20),
                height=300, paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12)
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.bar_chart(parc_data.set_index('Mode'))

    with col_b:
        st.markdown('<div class="section-header">Évolution temporelle (scénario)</div>',
                    unsafe_allow_html=True)

        EVOL = {
            'tendanciel': {'2025': 0.03, '2030': 0.15, '2035': 0.30, '2040': 0.50, '2050': 0.70},
            'accéléré':   {'2025': 0.05, '2030': 0.25, '2035': 0.50, '2040': 0.70, '2050': 0.90},
            'lent':       {'2025': 0.02, '2030': 0.08, '2035': 0.18, '2040': 0.32, '2050': 0.50},
        }
        evol = pd.DataFrame({
            'Année': [int(a) for a in EVOL['tendanciel'].keys()],
            'Tendanciel': list(EVOL['tendanciel'].values()),
            'Accéléré':   list(EVOL['accéléré'].values()),
            'Lent':       list(EVOL['lent'].values()),
        })

        try:
            fig2 = px.line(
                evol, x='Année', y=['Tendanciel', 'Accéléré', 'Lent'],
                color_discrete_map={
                    'Tendanciel': '#00b4d8',
                    'Accéléré':   '#2d9e6b',
                    'Lent':       '#e09a2d'
                },
                labels={'value': 'Part VP électriques', 'variable': 'Scénario'}
            )
            fig2.update_layout(
                height=300, margin=dict(t=20, b=40, l=20, r=20),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(tickformat='.0%', gridcolor='#eee'),
                xaxis=dict(gridcolor='#eee'),
                legend=dict(orientation='h', y=-0.25)
            )
            # Marquer l'horizon sélectionné
            fig2.add_vline(
                x=int(horizon), line_dash='dash',
                line_color='#888', annotation_text=f"Horizon {horizon}"
            )
            st.plotly_chart(fig2, use_container_width=True)
        except ImportError:
            st.line_chart(evol.set_index('Année'))

    # Tableau CSP
    st.markdown('<div class="section-header">Pénétration VE par catégorie socio-professionnelle</div>',
                unsafe_allow_html=True)
    df_csp_display = data['csp'][['occupation', 'pct_ve', 'pct_l6l7',
                                   'dist_moyenne_km', 'cout_moyen_eur']].copy()
    df_csp_display.columns = ['CSP', 'Part VE', 'Part L6/L7',
                               'Distance moy. (km/j)', 'Coût moy. (€/mois)']
    df_csp_display['Part VE']    = df_csp_display['Part VE'].map('{:.1%}'.format)
    df_csp_display['Part L6/L7'] = df_csp_display['Part L6/L7'].map('{:.1%}'.format)
    df_csp_display['Distance moy. (km/j)'] = df_csp_display['Distance moy. (km/j)'].map('{:.0f}'.format)
    df_csp_display['Coût moy. (€/mois)']   = df_csp_display['Coût moy. (€/mois)'].map('{:.0f} €'.format)
    st.dataframe(df_csp_display, use_container_width=True, hide_index=True)


# ── TAB 2 : RECHARGE & CO₂ ──────────────────────────────────────────────────
with tab2:
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown('<div class="section-header">Profil solaire & fenêtres de recharge</div>',
                    unsafe_allow_html=True)
        df_rte = data['rte'].copy()

        # Zones tarifaires
        try:
            fig3 = px.area(
                df_rte, x='heure',
                y=['solaire_ete', 'solaire_hiver'],
                color_discrete_map={
                    'solaire_ete':   '#f5a623',
                    'solaire_hiver': '#90caf9'
                },
                labels={'value': 'Production solaire (normalisée)',
                        'variable': 'Saison', 'heure': 'Heure'}
            )
            # Fenêtre solaire
            fig3.add_vrect(x0=11, x1=15, fillcolor='#f5a62340',
                           line_width=0, annotation_text="Pic solaire")
            # Fenêtre nuit
            fig3.add_vrect(x0=22, x1=24, fillcolor='#0077b620',
                           line_width=0)
            fig3.add_vrect(x0=0, x1=6, fillcolor='#0077b620',
                           line_width=0, annotation_text="Nuit")
            fig3.update_layout(
                height=300, margin=dict(t=30, b=40, l=20, r=20),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(gridcolor='#eee'), xaxis=dict(gridcolor='#eee'),
                legend=dict(orientation='h', y=-0.3)
            )
            st.plotly_chart(fig3, use_container_width=True)
        except ImportError:
            st.line_chart(df_rte.set_index('heure')[['solaire_ete', 'solaire_hiver']])

    with col_b:
        st.markdown('<div class="section-header">CO₂ du kWh selon l\'heure</div>',
                    unsafe_allow_html=True)
        try:
            fig4 = px.bar(
                df_rte, x='heure', y='co2_g_kwh',
                color='co2_g_kwh',
                color_continuous_scale=['#2d9e6b', '#f5a623', '#e05252'],
                labels={'co2_g_kwh': 'gCO₂/kWh', 'heure': 'Heure'}
            )
            fig4.update_layout(
                height=300, margin=dict(t=20, b=40, l=20, r=20),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                coloraxis_showscale=False,
                yaxis=dict(gridcolor='#eee'), xaxis=dict(gridcolor='#eee')
            )
            st.plotly_chart(fig4, use_container_width=True)
        except ImportError:
            st.bar_chart(df_rte.set_index('heure')['co2_g_kwh'])

    # Comparaison coûts
    st.markdown('<div class="section-header">Comparaison coût mensuel par type de motorisation</div>',
                unsafe_allow_html=True)
    comp = pd.DataFrame({
        'Motorisation': ['Thermique', 'VP électrique', 'L6/L7 électrique'],
        'Énergie':  [
            24 * 21.5 * 12.6 / 100,
            24 * 21.5 * 18 / 100 * tarif_actif['HP'] * 0.3
            + 24 * 21.5 * 18 / 100 * tarif_actif['HC'] * 0.4
            + 24 * 21.5 * 18 / 100 * tarif_actif['solaire'] * 0.3,
            24 * 21.5 * 9 / 100 * tarif_actif['HP'] * 0.3
            + 24 * 21.5 * 9 / 100 * tarif_actif['HC'] * 0.4
            + 24 * 21.5 * 9 / 100 * tarif_actif['solaire'] * 0.3,
        ],
        'Taxe km': [24 * 21.5 * taxe_km / 100] * 3,
    })
    comp['Total'] = comp['Énergie'] + comp['Taxe km']

    try:
        fig5 = px.bar(
            comp.melt(id_vars='Motorisation', value_vars=['Énergie', 'Taxe km']),
            x='Motorisation', y='value', color='variable',
            barmode='stack',
            color_discrete_map={'Énergie': '#0077b6', 'Taxe km': '#e09a2d'},
            labels={'value': '€/mois', 'variable': 'Composante'}
        )
        fig5.update_layout(
            height=280, margin=dict(t=20, b=40, l=20, r=20),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(gridcolor='#eee'), xaxis=dict(gridcolor='#eee'),
            legend=dict(orientation='h', y=-0.3)
        )
        st.plotly_chart(fig5, use_container_width=True)
    except ImportError:
        st.bar_chart(comp.set_index('Motorisation')[['Énergie', 'Taxe km']])


# ── TAB 3 : IMPACT FISCAL ───────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Balance fiscale — Taxe km vs perte TICPE</div>',
                unsafe_allow_html=True)

    # Courbe balance selon niveau de taxe
    taxes_range = np.arange(0, 10.5, 0.5)
    N_COND = 38_000_000
    KM_AN  = 24 * 21.5 * 12

    recettes = [N_COND * KM_AN * t / 100 / 1e9 for t in taxes_range]
    perte_f  = metr['perte_ticpe']

    df_fiscal = pd.DataFrame({
        'Taxe (c€/km)': taxes_range,
        'Recettes taxe km (Md€)':  recettes,
        'Perte TICPE (Md€)': [perte_f] * len(taxes_range),
    })

    try:
        fig6 = px.line(
            df_fiscal, x='Taxe (c€/km)',
            y=['Recettes taxe km (Md€)', 'Perte TICPE (Md€)'],
            color_discrete_map={
                'Recettes taxe km (Md€)': '#2d9e6b',
                'Perte TICPE (Md€)':      '#e05252'
            }
        )
        fig6.add_vline(x=taxe_km, line_dash='dash', line_color='#0077b6',
                       annotation_text=f"Taxe actuelle : {taxe_km} c€/km")
        fig6.add_hline(y=perte_f, line_dash='dot', line_color='#e05252',
                       annotation_text=f"Perte TICPE : {perte_f:.1f} Md€")
        fig6.update_layout(
            height=350, margin=dict(t=30, b=40, l=20, r=20),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(gridcolor='#eee', title='Milliards €/an'),
            xaxis=dict(gridcolor='#eee'),
            legend=dict(orientation='h', y=-0.25)
        )
        st.plotly_chart(fig6, use_container_width=True)
    except ImportError:
        st.line_chart(df_fiscal.set_index('Taxe (c€/km)'))

    col_f1, col_f2, col_f3 = st.columns(3)
    col_f1.metric("Recettes taxe km", f"{metr['recettes_taxe']:.2f} Md€/an")
    col_f2.metric("Perte TICPE", f"-{metr['perte_ticpe']:.2f} Md€/an")
    delta_val = metr['balance_fiscale']
    col_f3.metric(
        "Balance nette",
        f"{delta_val:+.2f} Md€/an",
        delta=f"{'Excédent' if delta_val >= 0 else 'Déficit'}",
        delta_color="normal" if delta_val >= 0 else "inverse"
    )

    st.markdown('<div class="section-header">Sensibilité par CSP</div>',
                unsafe_allow_html=True)
    df_sens = data['csp'][['occupation', 'sensibilite_moyenne',
                            'cout_moyen_eur', 'pct_bascule']].copy()
    df_sens.columns = ['CSP', 'Sensibilité prix (1-5)',
                       'Coût mobilité (€/mois)', 'Part en arbitrage']
    df_sens['Part en arbitrage'] = df_sens['Part en arbitrage'].map('{:.1%}'.format)
    df_sens['Sensibilité prix (1-5)'] = df_sens['Sensibilité prix (1-5)'].map('{:.1f}'.format)
    df_sens['Coût mobilité (€/mois)'] = df_sens['Coût mobilité (€/mois)'].map('{:.0f} €'.format)
    st.dataframe(df_sens, use_container_width=True, hide_index=True)


# ── TAB 4 : REPORTS MODAUX ──────────────────────────────────────────────────
with tab4:
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown('<div class="section-header">Répartition modale après taxe</div>',
                    unsafe_allow_html=True)

        modes_final = pd.DataFrame({
            'Mode': ['Thermique', 'VP électrique', 'L6/L7', 'Vélo', 'Véli', 'TC'],
            'Part': [
                metr['part_thermique'] * (1 - metr['pct_bascule']),
                metr['part_ve']        * (1 - metr['pct_bascule'] * 0.3),
                metr['part_l6l7']      * (1 - metr['pct_bascule'] * 0.2),
                metr['report_velo'],
                metr['report_veli'],
                metr['report_tc'],
            ]
        })
        # Normaliser
        total = modes_final['Part'].sum()
        modes_final['Part'] = modes_final['Part'] / total

        try:
            fig7 = px.bar(
                modes_final, x='Mode', y='Part',
                color='Mode',
                color_discrete_map={
                    'Thermique':    '#e05252',
                    'VP électrique':'#00b4d8',
                    'L6/L7':       '#0077b6',
                    'Vélo':        '#2d9e6b',
                    'Véli':        '#5cad8e',
                    'TC':          '#7c6fcd'
                },
                text_auto='.1%'
            )
            fig7.update_layout(
                height=320, showlegend=False,
                margin=dict(t=20, b=40, l=20, r=20),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(tickformat='.0%', gridcolor='#eee'),
                xaxis=dict(gridcolor='#eee')
            )
            st.plotly_chart(fig7, use_container_width=True)
        except ImportError:
            st.bar_chart(modes_final.set_index('Mode'))

    with col_b:
        st.markdown('<div class="section-header">Reports par type de zone</div>',
                    unsafe_allow_html=True)
        df_zone = data['zone'].copy()

        try:
            fig8 = px.bar(
                df_zone, x='type_zone', y='part', color='mode',
                barmode='stack', text_auto='.0%',
                color_discrete_map={
                    'vp_thermique':   '#e05252',
                    'vp_electrique':  '#00b4d8',
                    'l6l7_electrique':'#0077b6',
                    'velo':           '#2d9e6b',
                    'veli':           '#5cad8e',
                    'tc':             '#7c6fcd'
                },
                labels={'part': 'Part modale', 'type_zone': 'Zone',
                        'mode': 'Mode'}
            )
            fig8.update_layout(
                height=320, margin=dict(t=20, b=40, l=20, r=20),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(tickformat='.0%', gridcolor='#eee'),
                legend=dict(orientation='h', y=-0.3)
            )
            st.plotly_chart(fig8, use_container_width=True)
        except ImportError:
            st.bar_chart(df_zone.pivot(index='type_zone',
                                       columns='mode', values='part'))

    # Règles de report
    st.markdown('<div class="section-header">Règles de report modal appliquées</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    | Distance | Zone | Mode de report |
    |---|---|---|
    | < 5 km | Toutes | **Vélo** |
    | 5–30 km | Urbaine | **TC** (offre disponible) |
    | 5–30 km | Périurbaine / Rurale | **Véli** (L6/L7) |
    | > 30 km | Urbaine | **TC** |
    | > 30 km | Rurale | **Véli** (faute d'alternative) |

    *Déclenchement : coût mobilité > {12:.0f}% du revenu mensuel × sensibilité au prix (score 1–5)*
    """)

    st.info(
        f"💡 Avec une taxe de **{taxe_km} c€/km**, environ **{metr['pct_bascule']:.1%}** "
        f"des personas Nemotron sont en situation d'arbitrage modal. "
        f"Le report se décompose en **{metr['report_tc']:.1%} → TC**, "
        f"**{metr['report_veli']:.1%} → véli**, "
        f"**{metr['report_velo']:.1%} → vélo**."
    )


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Sources : Nemotron-Personas-France (NVIDIA, CC-BY 4.0) · "
    "EMP 2018-2019 (SDES/Ministère) · RTE Eco2mix · "
    "SDES Parc automobile · Calculs propres"
)
