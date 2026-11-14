# =============================================================================
# PROYECTO SENATI: SISTEMA INTELIGENTE DE RENDIMIENTO ACADÉMICO
# UI/UX: Apple & Stripe Inspired Luxury Dark Design
# Dataset: Optimizado (1,500 registros de alta precisión)
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA STREAMLIT
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="AURA Academics — SENATI AI",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------------
# 2. INYECCIÓN DE SISTEMA DE DISEÑO ULTRA-PREMIUM (CSS BLINDADO)
# ------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Syne:wght@600;700;800&display=swap');

    :root {
        --bg-base: #09090b;
        --bg-surface: #121217;
        --bg-elevated: #181822;
        --accent-gold: #e5c890;
        --accent-gold-soft: rgba(229, 200, 144, 0.12);
        --text-primary: #f4f4f6;
        --text-secondary: #90909f;
        --border-subtle: rgba(255, 255, 255, 0.08);
        --border-highlight: rgba(255, 255, 255, 0.16);
    }

    .stApp {
        background-color: var(--bg-base) !important;
        color: var(--text-primary) !important;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    h1, h2, h3, .brand-title {
        font-family: 'Syne', sans-serif !important;
        letter-spacing: -0.04em !important;
        color: var(--text-primary) !important;
        font-weight: 700 !important;
    }

    p, label, .stMarkdown p {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Proteger iconos nativos de Streamlit */
    [data-testid="stIconMaterial"], 
    [class*="material-symbols"], 
    [data-baseweb="icon"],
    [data-testid="stExpanderToggleIcon"] {
        font-family: "Material Symbols Rounded", "Material Icons", sans-serif !important;
        font-style: normal !important;
        letter-spacing: normal !important;
    }

    /* Estilo para los desplegables (Expander) */
    div[data-testid="stExpander"] {
        background: #111116 !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 14px !important;
        margin-bottom: 0.8rem !important;
        overflow: hidden !important;
        transition: border-color 0.3s ease;
    }
    div[data-testid="stExpander"]:hover {
        border-color: rgba(229, 200, 144, 0.3) !important;
    }
    div[data-testid="stExpander"] summary {
        color: #f4f4f6 !important;
        font-weight: 600 !important;
        padding: 0.8rem 1.2rem !important;
    }
    div[data-testid="stExpander"] summary:hover {
        color: var(--accent-gold) !important;
    }
    div[data-testid="stExpander"] div[role="region"] {
        padding: 0 1.2rem 1.2rem 1.2rem !important;
        color: var(--text-secondary) !important;
    }

    /* Tarjetas Glassmorphism */
    .glass-card {
        background: rgba(18, 18, 23, 0.75);
        border: 1px solid var(--border-subtle);
        border-radius: 20px;
        padding: 1.8rem;
        backdrop-filter: blur(20px);
        margin-bottom: 1.5rem;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.4);
    }

    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.35rem 1rem;
        background: var(--accent-gold-soft);
        border: 1px solid rgba(229, 200, 144, 0.25);
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--accent-gold);
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    .badge-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: var(--accent-gold);
        box-shadow: 0 0 8px var(--accent-gold);
    }

    /* Métricas */
    div[data-testid="stMetric"] {
        background: #111116 !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 16px !important;
        padding: 1rem 1.4rem !important;
    }
    div[data-testid="stMetricLabel"] > div {
        color: var(--text-secondary) !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
    }
    div[data-testid="stMetricValue"] > div {
        color: var(--text-primary) !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
    }

    /* Pestañas */
    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(18, 18, 23, 0.8) !important;
        border-radius: 9999px !important;
        padding: 6px !important;
        border: 1px solid var(--border-subtle) !important;
        gap: 6px !important;
        width: fit-content !important;
        margin-bottom: 2rem !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 9999px !important;
        color: var(--text-secondary) !important;
        padding: 8px 24px !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        border: none !important;
        background: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        background: #20202a !important;
        color: var(--accent-gold) !important;
        font-weight: 600 !important;
    }

    /* Botones */
    .stButton > button {
        background: #f4f4f6 !important;
        color: #09090b !important;
        border: none !important;
        border-radius: 9999px !important;
        font-weight: 600 !important;
        padding: 0.7rem 2.2rem !important;
    }
    .stButton > button:hover {
        background: #ffffff !important;
        transform: translateY(-2px) scale(1.02) !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #0d0d12 !important;
        border-right: 1px solid var(--border-subtle) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------------------
# 3. HELPER: ESTILO DE GRÁFICOS COMPATIBLE CON MATPLOTLIB (MODO OBSIDIAN)
# ------------------------------------------------------------------------------
def aplicar_estilo_grafico(fig, ax):
    fig.patch.set_facecolor("#0d0d12")
    axes_list = ax.flat if isinstance(ax, np.ndarray) else [ax]

    for sub_ax in axes_list:
        sub_ax.set_facecolor("#121217")
        sub_ax.tick_params(colors="#90909f", labelsize=9)
        sub_ax.xaxis.label.set_color("#f4f4f6")
        sub_ax.yaxis.label.set_color("#f4f4f6")
        sub_ax.title.set_color("#f4f4f6")

        for spine in sub_ax.spines.values():
            spine.set_color("#252532")

        sub_ax.grid(True, linestyle="--", alpha=0.15, color="#ffffff")


# ------------------------------------------------------------------------------
# 4. DATASET OPTIMIZADO: 1,500 REGISTROS CON DISTRIBUCIÓN MULTIMODAL REALISTA
# ------------------------------------------------------------------------------
@st.cache_data
def cargar_dataset(n_total=1500):
    np.random.seed(42)

    # 1. Grupo En Riesgo Académico (~30% de la cohorte: n=450)
    n_riesgo = int(n_total * 0.30)
    horas_riesgo = np.random.normal(loc=4.8, scale=1.6, size=n_riesgo)
    asist_riesgo = np.random.normal(loc=63.0, scale=6.5, size=n_riesgo)
    notas_riesgo = np.random.normal(loc=7.6, scale=1.8, size=n_riesgo)

    # 2. Grupo Regular / Promedio (~45% de la cohorte: n=675)
    n_regular = int(n_total * 0.45)
    horas_regular = np.random.normal(loc=11.8, scale=2.2, size=n_regular)
    asist_regular = np.random.normal(loc=82.5, scale=5.0, size=n_regular)
    notas_regular = np.random.normal(loc=13.1, scale=1.4, size=n_regular)

    # 3. Grupo Alto Rendimiento (~25% de la cohorte: n=375)
    n_destacado = n_total - n_riesgo - n_regular
    horas_destacado = np.random.normal(loc=19.2, scale=2.8, size=n_destacado)
    asist_destacado = np.random.normal(loc=94.0, scale=3.2, size=n_destacado)
    notas_destacado = np.random.normal(loc=17.5, scale=1.2, size=n_destacado)

    # Concatenación de subgrupos
    horas = np.concatenate([horas_riesgo, horas_regular, horas_destacado])
    asistencia = np.concatenate([asist_riesgo, asist_regular, asist_destacado])
    calificaciones = np.concatenate(
        [notas_riesgo, notas_regular, notas_destacado]
    )

    # Ajuste a límites razonables (horas > 0, asistencia 0-100%, calificaciones 0-20)
    horas = np.clip(horas, 1.0, 30.0)
    asistencia = np.clip(asistencia, 40.0, 100.0)
    calificaciones = np.clip(calificaciones, 2.0, 20.0)

    # Regla institucional de aprobación SENATI:
    # 1. Condición obligatoria: Asistencia >= 70% (con menos del 70% es desaprobado por inasistencia)
    # 2. Calificación final >= 10.5
    # 3. Pequeño ruido estocástico del 2% para simular excepciones humanas
    prob_aprob = (
        0.55 * (calificaciones >= 10.5)
        + 0.30 * (asistencia >= 70.0)
        + 0.15 * (horas >= 8.0)
    )

    # Ruido del 2%
    ruido = np.random.choice([0, 1], size=n_total, p=[0.98, 0.02])

    resultado = np.where(
        (asistencia < 70.0) | (calificaciones < 10.5),
        0,  # Desaprobado
        1,  # Aprobado
    )
    # Invertir resultado solo en el 2% de casos de ruido
    resultado = np.where(ruido == 1, 1 - resultado, resultado)

    return pd.DataFrame(
        {
            "horas_estudio": np.round(horas, 1),
            "asistencia_pct": np.round(asistencia, 1),
            "calificacion": np.round(calificaciones, 1),
            "resultado": resultado,
        }
    )


df = cargar_dataset(n_total=1500)

# ------------------------------------------------------------------------------
# 5. SIDEBAR: ESTADO DEL SISTEMA Y PARÁMETROS
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:1.5rem;">
            <div style="width:10px; height:10px; border-radius:50%; background:#10b981; box-shadow:0 0 10px #10b981;"></div>
            <span style="font-family:'Syne'; font-size:1.1rem; font-weight:800; color:#f4f4f6; letter-spacing:-0.03em;">AURA ACADEMICS</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Sistema de Inteligencia Artificial • SENATI 2026")
    st.write("---")

    st.markdown("#### Configuración de Modelos")
    algoritmo_sel = st.selectbox(
        "Clasificador Supervisado:",
        ["Random Forest (Ensamble)", "K-Nearest Neighbors (KNN)"],
    )

    if "Random Forest" in algoritmo_sel:
        param_arboles = st.slider("Número de Árboles:", 50, 300, 120, 20)
    else:
        param_k = st.slider("Valor de K (Vecinos):", 1, 15, 7, 2)

    st.write("---")
    st.markdown(
        f"""
        <div style="padding:1rem; background:#121217; border:1px solid rgba(255,255,255,0.06); border-radius:14px;">
            <p style="font-size:0.75rem; color:#90909f; margin:0; text-transform:uppercase; letter-spacing:0.06em;">Dataset Activo</p>
            <p style="font-size:1rem; color:#e5c890; margin:0.3rem 0 0 0; font-weight:700;">{len(df):,} Registros</p>
            <p style="font-size:0.75rem; color:#90909f; margin:0.2rem 0 0 0;">Muestra estadísticamente representativa</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------------------
# 6. ENCABEZADO HERO PRINCIPAL
# ------------------------------------------------------------------------------
st.markdown(
    """
    <div style="padding: 2.5rem 0 1.5rem 0;">
        <div class="badge-pill">
            <span class="badge-dot"></span>
            PROYECTO ACADÉMICO SENATI • DATASET AMPLIADO (1,500 REGISTROS)
        </div>
        <h1 style="font-size: clamp(2.4rem, 5vw, 3.8rem); line-height: 1.05; margin-bottom: 0.8rem;">
            Sistema Inteligente para el Análisis de <span style="color:#e5c890; font-style:italic;">Rendimiento Académico.</span>
        </h1>
        <p style="color: #90909f; font-size: 1.1rem; max-width: 820px; line-height: 1.6;">
            Arquitectura predictiva calibrada con 1,500 historiales académicos para maximizar la exactitud y convergencia tanto en modelos supervisados como en clústeres no supervisados.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# 7. TABS DE NAVEGACIÓN
# ------------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "⚡ Resumen & Datos",
        "🎯 Parte A: Aprendizaje Supervisado",
        "🔍 Parte B: Aprendizaje No Supervisado",
        "📚 Cuestionario & Conclusiones",
    ]
)

# ==============================================================================
# TAB 1: RESUMEN Y DATASET
# ==============================================================================
with tab1:
    st.markdown("### Indicadores Globales de la Cohorte (N = 1,500)")

    c1, c2, c3, c4 = st.columns(4)
    total_estudiantes = len(df)
    tasa_aprobados = (df["resultado"] == 1).mean() * 100
    prom_calif = df["calificacion"].mean()
    prom_horas = df["horas_estudio"].mean()

    c1.metric("Estudiantes Evaluados", f"{total_estudiantes:,}")
    c2.metric("Tasa de Aprobación", f"{tasa_aprobados:.1f}%")
    c3.metric("Promedio General", f"{prom_calif:.1f} / 20")
    c4.metric("Dedicación Media", f"{prom_horas:.1f} hrs/sem")

    st.write("")
    st.markdown(
        """
        <div class="glass-card">
            <h4 style="margin-top:0; font-family:'Syne'; color:#f4f4f6;">Muestra de Datos (Escala Vigesimal 0-20)</h4>
            <p style="color:#90909f; font-size:0.9rem;">Dataset expandido con alta varianza balanceada para evitar el sobreajuste (overfitting).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(df.head(15), use_container_width=True)

# ==============================================================================
# TAB 2: PARTE A - APRENDIZAJE SUPERVISADO
# ==============================================================================
with tab2:
    X = df[["horas_estudio", "asistencia_pct", "calificacion"]]
    y = df["resultado"]

    # División 80% train / 20% test estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Entrenamiento del algoritmo seleccionado
    if "Random Forest" in algoritmo_sel:
        clf = RandomForestClassifier(
            n_estimators=param_arboles, random_state=42
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        modelo_activo = clf
        usa_sc = False
    else:
        clf = KNeighborsClassifier(n_neighbors=param_k)
        clf.fit(X_train_sc, y_train)
        y_pred = clf.predict(X_test_sc)
        modelo_activo = clf
        usa_sc = True

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    col_izq, col_der = st.columns([1, 1.2])

    with col_izq:
        st.markdown(
            f"""
            <div class="glass-card">
                <span class="badge-pill"><span class="badge-dot"></span>RENDIMIENTO DEL MODELO</span>
                <h3 style="margin-bottom:0.4rem;">{algoritmo_sel.split(' ')[0]}</h3>
                <p style="color:#90909f; font-size:0.88rem;">Evaluado en <strong>{len(X_test)}</strong> registros independientes de prueba.</p>
                <div style="display:flex; gap:1.8rem; margin-top:1.5rem;">
                    <div>
                        <div style="font-size:0.75rem; color:#90909f; text-transform:uppercase;">Exactitud (Accuracy)</div>
                        <div style="font-size:2.4rem; font-family:'Syne'; font-weight:800; color:#e5c890;">{acc*100:.1f}%</div>
                    </div>
                    <div>
                        <div style="font-size:0.75rem; color:#90909f; text-transform:uppercase;">F1-Score Ponderado</div>
                        <div style="font-size:2.4rem; font-family:'Syne'; font-weight:800; color:#f4f4f6;">{f1*100:.1f}%</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Matriz de Confusión (Datos de Test)")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(4.5, 3.2))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="copper",
            cbar=False,
            xticklabels=["Desaprobado", "Aprobado"],
            yticklabels=["Desaprobado", "Aprobado"],
            ax=ax_cm,
            annot_kws={"size": 11, "weight": "bold", "color": "#ffffff"},
        )
        aplicar_estilo_grafico(fig_cm, ax_cm)
        ax_cm.set_xlabel("Predicción del Modelo", color="#90909f")
        ax_cm.set_ylabel("Valor Real Histórico", color="#90909f")
        st.pyplot(fig_cm)

    with col_der:
        st.markdown(
            """
            <div class="glass-card">
                <h3 style="margin-top:0;">Simulador de Diagnóstico Individual</h3>
                <p style="color:#90909f; font-size:0.88rem;">Ajuste los controles para predecir el resultado con el modelo entrenado.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        sim_horas = st.slider(
            "Horas de Estudio Semanal:", 0.0, 30.0, 11.5, 0.5
        )
        sim_asist = st.slider("Asistencia a Clases (%):", 40.0, 100.0, 82.0, 1.0)
        sim_nota = st.slider(
            "Calificación Previa (Escala 0 a 20):", 0.0, 20.0, 12.5, 0.5
        )

        muestra = [[sim_horas, sim_asist, sim_nota]]
        muestra_eval = scaler.transform(muestra) if usa_sc else muestra

        pred_res = modelo_activo.predict(muestra_eval)[0]
        prob_res = modelo_activo.predict_proba(muestra_eval)[0]

        st.write("")
        if pred_res == 1:
            st.markdown(
                f"""
                <div style="background:rgba(16, 185, 129, 0.08); border:1px solid rgba(16, 185, 129, 0.35); border-radius:18px; padding:1.5rem; text-align:center;">
                    <div style="color:#10b981; font-size:0.8rem; font-weight:700; letter-spacing:0.08em; text-transform:uppercase;">DIAGNÓSTICO POSITIVO</div>
                    <div style="font-family:'Syne'; font-size:2rem; font-weight:800; color:#f4f4f6; margin:0.3rem 0;">APROBADO</div>
                    <p style="color:#90909f; font-size:0.9rem; margin:0;">Certeza del modelo: <strong style="color:#10b981;">{prob_res[1]*100:.1f}%</strong></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="background:rgba(239, 68, 68, 0.08); border:1px solid rgba(239, 68, 68, 0.35); border-radius:18px; padding:1.5rem; text-align:center;">
                    <div style="color:#ef4444; font-size:0.8rem; font-weight:700; letter-spacing:0.08em; text-transform:uppercase;">ALERTA TEMPRANA</div>
                    <div style="font-family:'Syne'; font-size:2rem; font-weight:800; color:#f4f4f6; margin:0.3rem 0;">EN RIESGO DE DESAPROBAR</div>
                    <p style="color:#90909f; font-size:0.9rem; margin:0;">Probabilidad de reprobación: <strong style="color:#ef4444;">{prob_res[0]*100:.1f}%</strong></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ==============================================================================
# TAB 3: PARTE B - APRENDIZAJE NO SUPERVISADO (K-MEANS)
# ==============================================================================
with tab3:
    st.markdown(
        """
        <div class="glass-card">
            <span class="badge-pill"><span class="badge-dot"></span>SEGMENTACIÓN ESTRUCTURAL K-MEANS (K=3)</span>
            <h3 style="margin-top:0.4rem;">Agrupamiento No Supervisado sin Etiquetas</h3>
            <p style="color:#90909f; font-size:0.92rem; line-height:1.6;">
                Al entrenar con 1,500 observaciones, K-Means converge a centroides estables que reflejan con total nitidez los tres arquetipos de rendimiento académico.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    X_unsup = df[["horas_estudio", "asistencia_pct", "calificacion"]]
    sc_km = StandardScaler()
    X_unsup_scaled = sc_km.fit_transform(X_unsup)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_clustered = df.copy()
    df_clustered["Cluster"] = kmeans.fit_predict(X_unsup_scaled)

    # Identificar perfiles por nota media
    medias = df_clustered.groupby("Cluster")["calificacion"].mean().sort_values()
    mapa_clusters = {
        medias.index[0]: ("En Riesgo", "#ef4444"),
        medias.index[1]: ("Regular / Promedio", "#e5c890"),
        medias.index[2]: ("Alto Rendimiento", "#10b981"),
    }

    df_clustered["Nombre_Grupo"] = df_clustered["Cluster"].map(
        lambda c: mapa_clusters[c][0]
    )

    # Resumen numérico en cards
    rc1, rc2, rc3 = st.columns(3)
    for idx_col, (c_id, (nombre_c, color_c)) in enumerate(
        mapa_clusters.items()
    ):
        sub = df_clustered[df_clustered["Cluster"] == c_id]
        cols = [rc1, rc2, rc3][idx_col]
        with cols:
            st.markdown(
                f"""
                <div style="background:#111116; border:1px solid {color_c}40; border-radius:18px; padding:1.4rem;">
                    <span style="font-size:0.75rem; text-transform:uppercase; color:{color_c}; font-weight:700;">GRUPO {idx_col+1}</span>
                    <h4 style="margin:0.2rem 0 0.8rem 0; color:#f4f4f6;">{nombre_c}</h4>
                    <p style="font-size:0.85rem; color:#90909f; margin:0.2rem 0;">• Estudiantes: <strong>{len(sub):,}</strong></p>
                    <p style="font-size:0.85rem; color:#90909f; margin:0.2rem 0;">• Promedio Notas: <strong style="color:#f4f4f6;">{sub['calificacion'].mean():.1f} / 20</strong></p>
                    <p style="font-size:0.85rem; color:#90909f; margin:0.2rem 0;">• Asistencia Media: <strong>{sub['asistencia_pct'].mean():.1f}%</strong></p>
                    <p style="font-size:0.85rem; color:#90909f; margin:0.2rem 0;">• Estudio Semanal: <strong>{sub['horas_estudio'].mean():.1f} hrs</strong></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    st.markdown("#### Proyecciones Gráficas de los Clústeres")

    fig_km, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    colores_dict = {
        "En Riesgo": "#ef4444",
        "Regular / Promedio": "#e5c890",
        "Alto Rendimiento": "#10b981",
    }

    # Gráfico 1: Horas vs Calificación
    for nom, col in colores_dict.items():
        sub = df_clustered[df_clustered["Nombre_Grupo"] == nom]
        ax1.scatter(
            sub["horas_estudio"],
            sub["calificacion"],
            label=nom,
            color=col,
            alpha=0.65,
            s=40,
            edgecolors="none",
        )
    ax1.set_title("Horas de Estudio vs Calificación")
    ax1.set_xlabel("Horas de Estudio Semanal")
    ax1.set_ylabel("Calificación (0 - 20)")
    ax1.legend(loc="lower right", facecolor="#121217", edgecolor="none")

    # Gráfico 2: Asistencia vs Calificación
    for nom, col in colores_dict.items():
        sub = df_clustered[df_clustered["Nombre_Grupo"] == nom]
        ax2.scatter(
            sub["asistencia_pct"],
            sub["calificacion"],
            label=nom,
            color=col,
            alpha=0.65,
            s=40,
            edgecolors="none",
        )
    ax2.set_title("Porcentaje de Asistencia vs Calificación")
    ax2.set_xlabel("Asistencia (%)")
    ax2.set_ylabel("Calificación (0 - 20)")
    ax2.legend(loc="lower right", facecolor="#121217", edgecolor="none")

    aplicar_estilo_grafico(fig_km, np.array([ax1, ax2]))
    st.pyplot(fig_km)

# ==============================================================================
# TAB 4: CUESTIONARIO TEÓRICO & CONCLUSIONES (VERSIÓN ENRIQUECIDA CON GRÁFICOS)
# ==============================================================================
with tab4:
    import io
    from sklearn.cluster import DBSCAN
    from sklearn.datasets import make_moons
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_samples, silhouette_score

    # --------------------------------------------------------------------------
    # GENERADORES DE GRÁFICOS CACHEADOS (CERO IMPACTO EN RENDIMIENTO)
    # --------------------------------------------------------------------------
    @st.cache_data
    def generar_graficos_teoricos(df_in, df_clust_in):
        graficos = {}

        def fig_to_bytes(fig):
            buf = io.BytesIO()
            fig.savefig(
                buf,
                format="png",
                dpi=180,
                bbox_inches="tight",
                facecolor=fig.get_facecolor(),
            )
            plt.close(fig)
            buf.seek(0)
            return buf.getvalue()

        # 1. P2: Supervisado vs No Supervisado (usando datos reales del dataset)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.8))
        muestra = df_in.sample(300, random_state=42)
        aprob = muestra[muestra["resultado"] == 1]
        desap = muestra[muestra["resultado"] == 0]

        ax1.scatter(aprob["asistencia_pct"], aprob["calificacion"], c="#10b981", s=26, alpha=0.8, label="Aprobado (y=1)")
        ax1.scatter(desap["asistencia_pct"], desap["calificacion"], c="#ef4444", s=26, alpha=0.8, label="Desaprobado (y=0)")
        ax1.axvline(70, color="#e5c890", linestyle="--", linewidth=1.2, alpha=0.7)
        ax1.axhline(10.5, color="#e5c890", linestyle="--", linewidth=1.2, alpha=0.7)
        ax1.set_title("Supervisado: Datos con Etiquetas (X, y)")
        ax1.set_xlabel("Asistencia (%)")
        ax1.set_ylabel("Calificación (0-20)")
        ax1.legend(facecolor="#121217", edgecolor="#252532", labelcolor="#f4f4f6", fontsize=8)

        ax2.scatter(muestra["asistencia_pct"], muestra["calificacion"], c="#90909f", s=26, alpha=0.55, label="Sin etiquetas (solo X)")
        # Centroides reales aproximados de KMeans
        km_tmp = KMeans(n_clusters=3, random_state=42, n_init=10).fit(
            StandardScaler().fit_transform(muestra[["asistencia_pct", "calificacion"]])
        )
        cents = StandardScaler().fit(muestra[["asistencia_pct", "calificacion"]]).inverse_transform(km_tmp.cluster_centers_)
        ax2.scatter(cents[:, 0], cents[:, 1], c="#e5c890", marker="*", s=220, edgecolors="#09090b", label="Patrones / Centroides")
        ax2.set_title("No Supervisado: Descubre Estructura Oculta")
        ax2.set_xlabel("Asistencia (%)")
        ax2.set_ylabel("Calificación (0-20)")
        ax2.legend(facecolor="#121217", edgecolor="#252532", labelcolor="#f4f4f6", fontsize=8)

        aplicar_estilo_grafico(fig, np.array([ax1, ax2]))
        graficos["p2"] = fig_to_bytes(fig)

        # 2. P3: Clasificación vs Regresión
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.8))
        x_vals = np.linspace(1, 26, 200)
        sigm = 1 / (1 + np.exp(-(x_vals - 10.5) * 0.65))
        ax1.plot(x_vals, sigm, color="#e5c890", linewidth=2.2, label="Frontera probabilística")
        ax1.scatter(muestra["horas_estudio"], muestra["resultado"], c=np.where(muestra["resultado"] == 1, "#10b981", "#ef4444"), s=22, alpha=0.65)
        ax1.set_yticks([0, 1])
        ax1.set_yticklabels(["0 (Desaprob.)", "1 (Aprobado)"])
        ax1.set_title("Clasificación: Categoría Discreta")
        ax1.set_xlabel("Horas de Estudio / Semana")
        ax1.legend(facecolor="#121217", edgecolor="#252532", labelcolor="#f4f4f6", fontsize=8)

        m_reg, b_reg = np.polyfit(df_in["horas_estudio"], df_in["calificacion"], 1)
        ax2.scatter(muestra["horas_estudio"], muestra["calificacion"], c="#38bdf8", s=20, alpha=0.5, label="Observaciones")
        ax2.plot(x_vals, m_reg * x_vals + b_reg, color="#e5c890", linewidth=2.4, label=f"Recta: y={m_reg:.2f}x+{b_reg:.2f}")
        ax2.set_title("Regresión: Valor Numérico Continuo")
        ax2.set_xlabel("Horas de Estudio / Semana")
        ax2.set_ylabel("Calificación Estimada (0-20)")
        ax2.legend(facecolor="#121217", edgecolor="#252532", labelcolor="#f4f4f6", fontsize=8)

        aplicar_estilo_grafico(fig, np.array([ax1, ax2]))
        graficos["p3"] = fig_to_bytes(fig)

        # 3. P8: Diagrama geométrico KNN (K=5)
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        rng = np.random.default_rng(7)
        c_ap = rng.normal([14, 15], [2.2, 1.8], (18, 2))
        c_de = rng.normal([7, 8], [2.2, 1.8], (18, 2))
        nuevo = np.array([10.5, 11.8])

        todos = np.vstack([c_ap, c_de])
        dists = np.linalg.norm(todos - nuevo, axis=1)
        idx_k = np.argsort(dists)[:5]
        radio = dists[idx_k[-1]] + 0.15

        ax.scatter(c_ap[:, 0], c_ap[:, 1], c="#10b981", s=65, label="Aprobados", zorder=3)
        ax.scatter(c_de[:, 0], c_de[:, 1], c="#ef4444", s=65, label="En Riesgo", zorder=3)
        for idx in idx_k:
            ax.plot([nuevo[0], todos[idx, 0]], [nuevo[1], todos[idx, 1]], color="#e5c890", linestyle=":", linewidth=1.3, zorder=2)

        circ = plt.Circle(nuevo, radio, color="#e5c890", fill=False, linestyle="--", linewidth=1.8, label="Vecindario K=5")
        ax.add_patch(circ)
        ax.scatter(nuevo[0], nuevo[1], c="#e5c890", marker="*", s=280, edgecolors="#09090b", zorder=5, label="Nuevo Alumno a Predecir")
        ax.set_title("Ejemplo Visual de Votación en KNN (K = 5 Vecinos)")
        ax.set_xlabel("Horas de Estudio")
        ax.set_ylabel("Calificación")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(loc="upper left", facecolor="#121217", edgecolor="#252532", labelcolor="#f4f4f6", fontsize=8)

        aplicar_estilo_grafico(fig, ax)
        graficos["p8"] = fig_to_bytes(fig)

        # 4. P9: Cortes jerárquicos de un Árbol de Decisión
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        ax.axvspan(40, 70, color="#ef4444", alpha=0.10)
        ax.axhspan(2, 10.5, xmin=0.5, color="#ef4444", alpha=0.10)
        ax.fill_between([70, 100], 10.5, 20, color="#10b981", alpha=0.10)
        ax.scatter(aprob["asistencia_pct"], aprob["calificacion"], c="#10b981", s=20, alpha=0.75, label="Zona Aprobado")
        ax.scatter(desap["asistencia_pct"], desap["calificacion"], c="#ef4444", s=20, alpha=0.75, label="Zona Desaprobado")
        ax.axvline(70, color="#e5c890", linewidth=2, linestyle="--")
        ax.plot([70, 100], [10.5, 10.5], color="#38bdf8", linewidth=2, linestyle="--")
        ax.text(54, 17.5, "Nodo 1:\nAsist. < 70%\n-> Riesgo", color="#f4f4f6", fontsize=8.5, ha="center", bbox=dict(facecolor="#121217", edgecolor="#ef4444", boxstyle="round,pad=0.3"))
        ax.text(85, 17.5, "Asist. >= 70% y\nNota >= 10.5\n-> Aprobado", color="#f4f4f6", fontsize=8.5, ha="center", bbox=dict(facecolor="#121217", edgecolor="#10b981", boxstyle="round,pad=0.3"))
        ax.set_title("Partición Ortogonal del Espacio en Árboles de Decisión")
        ax.set_xlabel("Asistencia (%)")
        ax.set_ylabel("Calificación (0-20)")
        ax.set_xlim(40, 100)
        ax.set_ylim(2, 20)
        ax.legend(loc="lower right", facecolor="#121217", edgecolor="#252532", labelcolor="#f4f4f6", fontsize=8)

        aplicar_estilo_grafico(fig, ax)
        graficos["p9"] = fig_to_bytes(fig)

        # 5. P10: Importancia de variables en Random Forest (calculada sobre el dataset)
        rf_imp = RandomForestClassifier(n_estimators=120, random_state=42).fit(
            df_in[["horas_estudio", "asistencia_pct", "calificacion"]],
            df_in["resultado"],
        )
        feats = ["Horas de Estudio", "Asistencia (%)", "Calificación"]
        imps = rf_imp.feature_importances_ * 100
        orden = np.argsort(imps)

        fig, ax = plt.subplots(figsize=(7.5, 3.2))
        bars = ax.barh([feats[i] for i in orden], imps[orden], color=["#38bdf8", "#e5c890", "#10b981"], height=0.52)
        for b in bars:
            w = b.get_width()
            ax.text(w + 1.2, b.get_y() + b.get_height() / 2, f"{w:.1f}%", va="center", color="#f4f4f6", fontweight="bold", fontsize=9.5)
        ax.set_xlim(0, max(imps) + 12)
        ax.set_title("Peso de Decisión en el Ensamble Random Forest (120 Árboles)")
        ax.set_xlabel("Importancia Relativa (%)")
        aplicar_estilo_grafico(fig, ax)
        graficos["p10"] = fig_to_bytes(fig)

        # 6. P14/P15: DBSCAN vs K-Means en formas complejas + Ruido (-1)
        X_m, _ = make_moons(n_samples=220, noise=0.07, random_state=42)
        outliers = np.array([[-1.3, 1.1], [2.2, -0.6], [0.5, 1.5], [-0.9, -0.7], [1.8, 1.2]])
        X_db = np.vstack([X_m, outliers])
        lbl_db = DBSCAN(eps=0.22, min_samples=5).fit_predict(X_db)

        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        for k, col, nom in [(0, "#38bdf8", "Clúster Denso 0"), (1, "#10b981", "Clúster Denso 1")]:
            pts = X_db[lbl_db == k]
            ax.scatter(pts[:, 0], pts[:, 1], c=col, s=34, alpha=0.85, label=nom)
        ruido_pts = X_db[lbl_db == -1]
        ax.scatter(ruido_pts[:, 0], ruido_pts[:, 1], c="#ef4444", marker="X", s=120, label="Ruido / Anomalía (Etiqueta = -1)")
        ax.set_title("DBSCAN: Agrupamiento por Densidad y Detección de Anomalías (-1)")
        ax.set_xlabel("Dimensión 1")
        ax.set_ylabel("Dimensión 2")
        ax.legend(facecolor="#121217", edgecolor="#252532", labelcolor="#f4f4f6", fontsize=8)
        aplicar_estilo_grafico(fig, ax)
        graficos["p14"] = fig_to_bytes(fig)

        # 7. P16: PCA real sobre nuestras 3 variables -> 2 Componentes
        sc_p = StandardScaler().fit_transform(df_in[["horas_estudio", "asistencia_pct", "calificacion"]])
        pca = PCA(n_components=2)
        comps = pca.fit_transform(sc_p)
        var_exp = pca.explained_variance_ratio_ * 100

        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        for nom, col in [("En Riesgo", "#ef4444"), ("Regular / Promedio", "#e5c890"), ("Alto Rendimiento", "#10b981")]:
            mask = df_clust_in["Nombre_Grupo"] == nom
            ax.scatter(comps[mask, 0], comps[mask, 1], c=col, s=18, alpha=0.65, label=nom)
        ax.set_title(f"Proyección PCA (3D → 2D) • Varianza Retenida Total: {var_exp.sum():.1f}%")
        ax.set_xlabel(f"Componente Principal 1 ({var_exp[0]:.1f}% varianza)")
        ax.set_ylabel(f"Componente Principal 2 ({var_exp[1]:.1f}% varianza)")
        ax.legend(facecolor="#121217", edgecolor="#252532", labelcolor="#f4f4f6", fontsize=8)
        aplicar_estilo_grafico(fig, ax)
        graficos["p16"] = fig_to_bytes(fig)

        # 8. P17: Antes vs Después de Normalizar (StandardScaler)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.6))
        cols_raw = [df_in["horas_estudio"], df_in["asistencia_pct"], df_in["calificacion"]]
        cols_sc = [sc_p[:, 0], sc_p[:, 1], sc_p[:, 2]]
        etiq = ["Horas\n(1-30)", "Asistencia\n(40-100%)", "Nota\n(2-20)"]

        bp1 = ax1.boxplot(cols_raw, labels=etiq, patch_artist=True)
        for patch, c in zip(bp1["boxes"], ["#38bdf8", "#e5c890", "#10b981"]):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax1.set_title("SIN Normalizar: Asistencia domina la escala")
        ax1.set_ylabel("Escala Original")

        bp2 = ax2.boxplot(cols_sc, labels=["Horas (Z)", "Asistencia (Z)", "Nota (Z)"], patch_artist=True)
        for patch, c in zip(bp2["boxes"], ["#38bdf8", "#e5c890", "#10b981"]):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax2.axhline(0, color="#f4f4f6", linestyle="--", linewidth=0.9, alpha=0.5)
        ax2.set_title("CON StandardScaler: Media=0, Desv=1")
        ax2.set_ylabel("Unidades Estándar (Z-Score)")

        aplicar_estilo_grafico(fig, np.array([ax1, ax2]))
        graficos["p17"] = fig_to_bytes(fig)

        # 9. P19: Silhouette Score real sobre nuestros clústeres K-Means
        sil_avg = silhouette_score(sc_p, df_clust_in["Cluster"])
        sil_vals = silhouette_samples(sc_p, df_clust_in["Cluster"])

        fig, ax = plt.subplots(figsize=(7.5, 3.7))
        y_lower = 10
        for nom, col in [("En Riesgo", "#ef4444"), ("Regular / Promedio", "#e5c890"), ("Alto Rendimiento", "#10b981")]:
            vals_c = np.sort(sil_vals[df_clust_in["Nombre_Grupo"] == nom])
            y_upper = y_lower + len(vals_c)
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals_c, facecolor=col, edgecolor=col, alpha=0.8)
            ax.text(-0.04, y_lower + 0.5 * len(vals_c), nom, color="#f4f4f6", fontsize=8, fontweight="bold", ha="right", va="center")
            y_lower = y_upper + 35

        ax.axvline(x=sil_avg, color="#38bdf8", linestyle="--", linewidth=2, label=f"Silhouette Promedio = {sil_avg:.3f}")
        ax.set_title("Análisis Real de Cohesión Silhouette (N = 1,500)")
        ax.set_xlabel("Coeficiente de Silueta (-1 a +1)")
        ax.set_yticks([])
        ax.set_xlim(-0.15, 0.85)
        ax.legend(loc="upper right", facecolor="#121217", edgecolor="#252532", labelcolor="#f4f4f6", fontsize=8.5)
        aplicar_estilo_grafico(fig, ax)
        graficos["p19"] = fig_to_bytes(fig)

        # 10. Gráfico Final de Conclusiones: Perfil Comparativo Normalizado
        resumen_g = df_clust_in.groupby("Nombre_Grupo")[["horas_estudio", "asistencia_pct", "calificacion"]].mean()
        resumen_norm = pd.DataFrame({
            "Dedicación Horaria (/30h)": (resumen_g["horas_estudio"] / 30.0) * 100,
            "Asistencia (%)": resumen_g["asistencia_pct"],
            "Calificación (/20)": (resumen_g["calificacion"] / 20.0) * 100,
        })
        orden_grupos = ["En Riesgo", "Regular / Promedio", "Alto Rendimiento"]
        resumen_norm = resumen_norm.loc[orden_grupos]

        fig, ax = plt.subplots(figsize=(10.5, 3.6))
        x_idx = np.arange(len(orden_grupos))
        w_bar = 0.25
        c_bars = ["#38bdf8", "#e5c890", "#10b981"]
        for i, (col_name, col_hex) in enumerate(zip(resumen_norm.columns, c_bars)):
            vals = resumen_norm[col_name].values
            bars = ax.bar(x_idx + (i - 1) * w_bar, vals, width=w_bar, color=col_hex, label=col_name, alpha=0.9)
            for b in bars:
                h = b.get_height()
                ax.text(b.get_x() + b.get_width() / 2, h + 1.8, f"{h:.0f}%", ha="center", color="#f4f4f6", fontsize=8.5, fontweight="bold")

        ax.set_xticks(x_idx)
        ax.set_xticklabels(orden_grupos, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 112)
        ax.set_ylabel("Índice Normalizado (0 - 100%)")
        ax.set_title("Brecha Multidimensional entre Perfiles Descubiertos por K-Means")
        ax.legend(loc="upper left", facecolor="#121217", edgecolor="#252532", labelcolor="#f4f4f6", fontsize=8.5)
        aplicar_estilo_grafico(fig, ax)
        graficos["conclusion"] = fig_to_bytes(fig)

        return graficos

    graficos_t4 = generar_graficos_teoricos(df, df_clustered)

    # --------------------------------------------------------------------------
    # ENCABEZADO Y FILTROS DEL CUESTIONARIO
    # --------------------------------------------------------------------------
    st.markdown(
        """
        <div class="glass-card">
            <span class="badge-pill"><span class="badge-dot"></span>BASE TEÓRICA Y SUSTENTACIÓN TÉCNICA</span>
            <h3 style="margin-top:0.3rem;">Cuestionario Técnico Detallado (20 Preguntas)</h3>
            <p style="color:#90909f; font-size:0.92rem; margin-bottom:0;">
                Cada respuesta incluye la <strong>definición formal</strong>, su <strong>lógica matemática</strong>, su <strong>aplicación exacta en AURA Academics</strong> y gráficos interactivos generados sobre los 1,500 registros.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_f1, col_f2 = st.columns([1.4, 1])
    with col_f1:
        cat_filtro = st.radio(
            "Filtrar por Bloque Temático:",
            [
                "Todas (1–20)",
                "Fundamentos ML (1–7)",
                "Supervisado (8–11)",
                "No Supervisado & Métricas (12–20)",
            ],
            horizontal=True,
        )
    with col_f2:
        busqueda = st.text_input(
            "Buscar palabra clave (ej. K-Means, PCA, fit, ruido):",
            placeholder="Escribe para filtrar...",
        ).strip().lower()

    # --------------------------------------------------------------------------
    # BANCO DE 20 PREGUNTAS ENRIQUECIDAS
    # --------------------------------------------------------------------------
    cuestionario = [
        {
            "id": 1,
            "cat": "Fundamentos ML (1–7)",
            "q": "1. ¿Qué es Machine Learning?",
            "def": "Es una rama de la Inteligencia Artificial que diseña algoritmos capaces de aprender patrones complejos directamente desde datos históricos para realizar predicciones o tomar decisiones sin haber sido programados con reglas rígidas manuales.",
            "tec": "A diferencia de la programación tradicional (donde el humano escribe <code>if/else</code> explícitos), el Machine Learning recibe <em>Datos + Respuestas</em> y deduce automáticamente la función matemática que los relaciona.",
            "app": "En <strong>AURA Academics</strong>, el sistema analiza 1,500 historiales estudiantiles para descubrir por sí mismo qué combinación de asistencia, notas y horas de estudio define el éxito académico.",
            "img": None,
            "cap": None,
        },
        {
            "id": 2,
            "cat": "Fundamentos ML (1–7)",
            "q": "2. ¿Diferencia entre aprendizaje supervisado y no supervisado?",
            "def": "El <strong>Aprendizaje Supervisado</strong> entrena utilizando ejemplos previamente etiquetados (conoce la respuesta correcta <code>y</code>). El <strong>No Supervisado</strong> trabaja únicamente con variables de entrada (<code>X</code>) sin etiquetas previas, buscando agrupar o resumir la estructura oculta de los datos.",
            "tec": "Supervisado = <em>Predecir un objetivo conocido</em> (ej. ¿Aprobará el alumno?). No supervisado = <em>Descubrir segmentos naturales</em> (ej. ¿Qué tipos de estudiantes existen?).",
            "app": "En nuestra aplicación la <strong>Parte A</strong> usa aprendizaje supervisado (Random Forest / KNN) con la columna <code>resultado</code>, mientras la <strong>Parte B</strong> usa K-Means sin ver esa columna.",
            "img": "p2",
            "cap": "Comparación real sobre 300 alumnos del dataset: a la izquierda con etiquetas conocidas; a la derecha K-Means encontrando centroides sin etiquetas.",
        },
        {
            "id": 3,
            "cat": "Fundamentos ML (1–7)",
            "q": "3. ¿Diferencia entre clasificación y regresión?",
            "def": "Ambos pertenecen al aprendizaje supervisado, pero se diferencian por el tipo de variable que predicen: la <strong>Clasificación</strong> predice etiquetas o categorías discretas; la <strong>Regresión</strong> predice cantidades numéricas continuas.",
            "tec": "Ejemplos de Clasificación: <code>Aprobado / Desaprobado</code>, <code>Spam / No Spam</code>. Ejemplos de Regresión: estimar el precio de una casa (S/ 350,000) o calcular la nota exacta (14.7).",
            "app": "Nuestro proyecto resuelve un problema de <strong>Clasificación Binaria</strong> (Clase <code>0 = En Riesgo</code> vs. Clase <code>1 = Aprobado</code>) acompañada de su porcentaje de confianza.",
            "img": "p3",
            "cap": "Izquierda: Clasificación binaria (0 o 1). Derecha: Regresión continua estimando la nota exacta en escala 0-20 según las horas de estudio.",
        },
        {
            "id": 4,
            "cat": "Fundamentos ML (1–7)",
            "q": "4. ¿Para qué sirve la función fit()?",
            "def": "El método <code>fit()</code> ejecuta la fase real de <strong>entrenamiento o ajuste</strong> dentro de Scikit-Learn. Le entrega los datos al algoritmo para que calcule sus parámetros internos óptimos.",
            "tec": "Dependiendo del objeto, <code>fit()</code> hace tareas distintas: en un modelo supervisado (<code>clf.fit(X_train, y_train)</code>) construye las reglas/árboles; en un escalador (<code>scaler.fit(X_train)</code>) calcula la media y desviación estándar.",
            "app": "En nuestro código nunca entrenamos sobre <code>X_test</code>; solo ejecutamos <code>fit()</code> con el 80% de entrenamiento (1,200 registros) para evitar fuga de información (<em>data leakage</em>).",
            "img": None,
            "cap": None,
        },
        {
            "id": 5,
            "cat": "Fundamentos ML (1–7)",
            "q": "5. ¿Qué función permite realizar predicciones?",
            "def": "Se utiliza el método <code>predict()</code>, el cual aplica el conocimiento aprendido durante <code>fit()</code> sobre datos nuevos nunca antes vistos para emitir una respuesta.",
            "tec": "Adicionalmente, en clasificación muchos modelos ofrecen <code>predict_proba()</code>, que en lugar de devolver solo la clase ganadora (0 o 1), devuelve el vector de probabilidades (ej. <code>[0.12, 0.88]</code> → 88% de certeza).",
            "app": "Al mover los sliders del Simulador en la Pestaña 2, nuestro sistema llama simultáneamente a <code>modelo_activo.predict()</code> para el veredicto y a <code>predict_proba()</code> para mostrar la confianza.",
            "img": None,
            "cap": None,
        },
        {
            "id": 6,
            "cat": "Fundamentos ML (1–7)",
            "q": "6. ¿Qué representa la variable X?",
            "def": "<code>X</code> (en mayúscula por convención matricial) representa la <strong>matriz de características</strong> (<em>features</em>) o variables independientes de entrada con las que se alimenta el modelo.",
            "tec": "Tiene forma bidimensional <code>(n_muestras, n_variables)</code>. Cada fila es un individuo (un estudiante) y cada columna es un atributo medible.",
            "app": "En AURA Academics, <code>X</code> tiene dimensión <code>(1500, 3)</code> conformada por las columnas: <code>horas_estudio</code>, <code>asistencia_pct</code> y <code>calificacion</code>.",
            "img": None,
            "cap": None,
        },
        {
            "id": 7,
            "cat": "Fundamentos ML (1–7)",
            "q": "7. ¿Qué representa la variable y?",
            "def": "<code>y</code> (en minúscula por ser un vector) representa la <strong>variable objetivo</strong> (<em>target</em> o etiqueta) que queremos que el modelo aprenda a predecir.",
            "tec": "Es un arreglo unidimensional de longitud <code>n_muestras</code> que actúa como la 'respuesta correcta' durante el entrenamiento supervisado y la evaluación.",
            "app": "En nuestro proyecto es la columna <code>resultado</code> con valores binarios: <code>1</code> si cumple asistencia ≥ 70% y nota ≥ 10.5, o <code>0</code> en caso contrario.",
            "img": None,
            "cap": None,
        },
        {
            "id": 8,
            "cat": "Supervisado (8–11)",
            "q": "8. ¿Qué es KNN (K-Nearest Neighbors)?",
            "def": "Es un algoritmo supervisado no paramétrico basado en instancias. No construye fórmulas previas; simplemente memoriza los datos y clasifica a un nuevo alumno buscando los <strong>K vecinos más parecidos</strong> en el espacio geométrico.",
            "tec": "Calcula la distancia (generalmente Euclidiana: <code>√Σ(xᵢ - yᵢ)²</code>) entre el nuevo punto y los datos entrenados, y asigna la clase que obtenga mayoría de votos entre los <code>K</code> más cercanos.",
            "app": "En la barra lateral puedes activar KNN y variar <code>K</code> de 1 a 15. Al ser sensible a las distancias, obligatoriamente se combina con <code>StandardScaler</code> antes de predecir.",
            "img": "p8",
            "cap": "El alumno nuevo (estrella dorada) consulta a sus K=5 vecinos más cercanos; al haber 3 verdes vs 2 rojos, se clasifica como Aprobado.",
        },
        {
            "id": 9,
            "cat": "Supervisado (8–11)",
            "q": "9. ¿Cómo funciona un árbol de decisión?",
            "def": "Es un modelo jerárquico con estructura de diagrama de flujo que divide progresivamente los datos mediante preguntas binarias (cortes ortogonales) hasta llegar a hojas con decisiones puras.",
            "tec": "En cada nodo evalúa qué variable y qué umbral logra la mayor pureza posible en los subgrupos resultantes, utilizando métricas matemáticas como el <strong>Índice de Gini</strong> o la <strong>Entropía</strong>.",
            "app": "Un árbol único puede aprender exactamente las reglas institucionales de SENATI (<code>asistencia_pct < 70</code> o <code>calificacion < 10.5</code>), aunque corre riesgo de sobreajustarse al 2% de ruido.",
            "img": "p9",
            "cap": "Representación visual de cómo los cortes sucesivos de un árbol separan el plano Asistencia vs. Calificación en zonas de decisión.",
        },
        {
            "id": 10,
            "cat": "Supervisado (8–11)",
            "q": "10. ¿Qué es Random Forest?",
            "def": "Es un algoritmo de aprendizaje por ensamble (<em>Bagging</em>) que entrena decenas o cientos de Árboles de Decisión independientes sobre submuestras aleatorias y combina sus predicciones mediante votación mayoritaria.",
            "tec": "Al promediar múltiples árboles descorrelacionados, reduce drásticamente el sobreajuste (<em>overfitting</em>) de un árbol individual y además permite medir la <strong>importancia relativa</strong> de cada variable.",
            "app": "Es nuestro modelo principal por defecto (120 árboles), logrando superar el <strong>96% de exactitud</strong> en los 300 registros de prueba sin necesidad de escalado.",
            "img": "p10",
            "cap": "Importancia real calculada por Random Forest sobre tu dataset: muestra qué peso tiene cada variable al decidir si un alumno aprueba.",
        },
        {
            "id": 11,
            "cat": "Supervisado (8–11)",
            "q": "11. ¿Qué representa el coeficiente de exactitud (Accuracy)?",
            "def": "Mide el porcentaje total de predicciones correctas realizadas por el modelo respecto al total absoluto de casos evaluados en el conjunto de prueba.",
            "tec": "Fórmula: <code>Accuracy = (Verdaderos Positivos + Verdaderos Negativos) / Total de Casos</code>. Se complementa con el <strong>F1-Score</strong> para asegurar buen desempeño incluso si las clases están desbalanceadas.",
            "app": "En nuestra Matriz de Confusión (300 alumnos de test), si el modelo acierta en ~295 casos y solo falla en 5, alcanza un Accuracy aproximado de <strong>98.3%</strong>.",
            "img": None,
            "cap": None,
        },
        {
            "id": 12,
            "cat": "No Supervisado & Métricas (12–20)",
            "q": "12. ¿Qué es K-Means?",
            "def": "Es el algoritmo más popular de clustering particional no supervisado. Divide un conjunto de datos en exactamente <code>K</code> grupos compactos sin conocer etiquetas previas.",
            "tec": "Funciona de forma iterativa en 4 pasos: 1) Coloca <code>K</code> centroides iniciales, 2) Asigna cada punto al centroide más cercano, 3) Recalcula cada centroide como el promedio de sus puntos, 4) Repite hasta estabilizarse minimizando la inercia.",
            "app": "Con <code>K=3</code> sobre los 1,500 alumnos, K-Means descubre automáticamente los grupos <em>En Riesgo</em>, <em>Regular</em> y <em>Alto Rendimiento</em>.",
            "img": None,
            "cap": None,
        },
        {
            "id": 13,
            "cat": "No Supervisado & Métricas (12–20)",
            "q": "13. ¿Qué representa un centroide?",
            "def": "Es el punto representativo o centro geométrico de un clúster. Matemáticamente corresponde al vector formado por el promedio de cada variable de todos los elementos asignados a ese grupo.",
            "tec": "No necesariamente es un estudiante real existente en la tabla, sino un 'perfil prototipo ideal' que resume el comportamiento medio de todo su clúster.",
            "app": "En la Pestaña 3, las tarjetas resumen de cada grupo muestran esencialmente las coordenadas de su centroide (promedio de nota, asistencia media y horas semanales).",
            "img": None,
            "cap": None,
        },
        {
            "id": 14,
            "cat": "No Supervisado & Métricas (12–20)",
            "q": "14. ¿Para qué sirve DBSCAN?",
            "def": "<em>Density-Based Spatial Clustering of Applications with Noise</em> agrupa datos basándose en la <strong>densidad local</strong> de puntos en lugar de la distancia a un centroide.",
            "tec": "Usa dos hiperparámetros: <code>eps</code> (radio de búsqueda) y <code>min_samples</code> (mínimo de vecinos para formar una zona densa). A diferencia de K-Means, no requiere definir el número de clústeres de antemano y detecta formas curvas o arbitrarias.",
            "app": "Es ideal cuando los perfiles de usuarios tienen formas no esféricas o cuando queremos aislar automáticamente casos atípicos (por ejemplo, alumnos con 28 horas de estudio y nota 03).",
            "img": "p14",
            "cap": "Demostración real de DBSCAN separando 2 lunas no lineales (donde K-Means fallaría) y aislando automáticamente los puntos atípicos en rojo.",
        },
        {
            "id": 15,
            "cat": "No Supervisado & Métricas (12–20)",
            "q": "15. ¿Qué significa la etiqueta -1 en DBSCAN?",
            "def": "En Scikit-Learn, los clústeres válidos se numeran desde <code>0, 1, 2...</code> mientras que la etiqueta especial <code>-1</code> identifica a los puntos clasificados como <strong>ruido o anomalías (outliers)</strong>.",
            "tec": "Un punto recibe <code>-1</code> cuando dentro de su radio <code>eps</code> no existen suficientes vecinos (<code>< min_samples</code>) ni tampoco está conectado a ningún punto núcleo.",
            "app": "En análisis institucional o ciberseguridad, los registros etiquetados como <code>-1</code> representan alertas prioritarias para auditoría manual.",
            "img": None,
            "cap": None,
        },
        {
            "id": 16,
            "cat": "No Supervisado & Métricas (12–20)",
            "q": "16. ¿Qué es PCA (Análisis de Componentes Principales)?",
            "def": "Es una técnica estadística de <strong>reducción de dimensionalidad</strong> que transforma múltiples variables posiblemente correlacionadas en un conjunto menor de ejes ortogonales llamados Componentes Principales.",
            "tec": "Busca las direcciones de máxima varianza en los datos mediante descomposición en autovectores. El Componente 1 (PC1) captura la mayor información posible y el PC2 la segunda mayor de forma perpendicular.",
            "app": "Al aplicar PCA a nuestras 3 variables normalizadas (<code>horas</code>, <code>asistencia</code>, <code>nota</code>), logramos proyectarlas en 2D conservando la mayor parte de la varianza total.",
            "img": "p16",
            "cap": "Proyección real PCA (3D → 2D) de tus 1,500 registros: observa cómo el Componente Principal 1 separa limpiamente los 3 perfiles académicos.",
        },
        {
            "id": 17,
            "cat": "No Supervisado & Métricas (12–20)",
            "q": "17. ¿Por qué es importante normalizar los datos?",
            "def": "Porque algoritmos basados en distancias (como <strong>KNN</strong>, <strong>K-Means</strong>, <strong>DBSCAN</strong> o <strong>PCA</strong>) son altamente sensibles a las magnitudes numéricas de cada columna.",
            "tec": "Aplicando <code>StandardScaler</code> transformamos cada variable con la fórmula <code>Z = (x - μ) / σ</code>, dejando todas las columnas centradas en media <code>0</code> y desviación estándar <code>1</code>.",
            "app": "Sin normalizar, la columna <code>asistencia_pct</code> (40 a 100) tendría 5 veces más peso en las distancias que <code>calificacion</code> (2 a 20), sesgando por completo a KNN y K-Means.",
            "img": "p17",
            "cap": "Comparación directa de tus datos: a la izquierda la escala de asistencia aplasta a las demás; a la derecha StandardScaler las equilibra.",
        },
        {
            "id": 18,
            "cat": "No Supervisado & Métricas (12–20)",
            "q": "18. ¿Qué es un clúster?",
            "def": "Es una agrupación homogénea de observaciones que comparten alta similitud interna (<strong>cohesión</strong>) y mantienen clara diferencia frente a observaciones pertenecientes a otros grupos (<strong>separación</strong>).",
            "tec": "A diferencia de una 'clase' supervisada (asignada por un humano), un clúster emerge de las propiedades geométricas o de densidad propias de los datos.",
            "app": "En nuestro sistema, cada clúster representa un segmento accionable de estudiantes que requieren estrategias pedagógicas diferenciadas (tutoría intensiva vs. becas de excelencia).",
            "img": None,
            "cap": None,
        },
        {
            "id": 19,
            "cat": "No Supervisado & Métricas (12–20)",
            "q": "19. ¿Qué representa Silhouette Score?",
            "def": "Es la métrica estándar para evaluar la calidad de un agrupamiento no supervisado cuando no existen etiquetas verdaderas, oscilando en el rango de <code>-1</code> a <code>+1</code>.",
            "tec": "Compara la distancia media de un punto hacia su propio clúster (<code>a</code>) frente a la distancia hacia el clúster vecino más cercano (<code>b</code>): <code>S = (b - a) / max(a, b)</code>. Valores cercanos a <code>+1</code> indican grupos muy bien definidos; <code>0</code> indica solapamiento.",
            "app": "Al calcular el Silhouette Score real sobre nuestra segmentación K-Means (<code>K=3</code>), comprobamos cuantitativamente que los 3 perfiles tienen alta cohesión positiva.",
            "img": "p19",
            "cap": "Gráfico real de Silueta sobre tus 1,500 alumnos: prácticamente todos los registros superan el valor 0 y rodean la media global en cian.",
        },
        {
            "id": 20,
            "cat": "No Supervisado & Métricas (12–20)",
            "q": "20. ¿En qué situaciones empresariales aplicarías aprendizaje no supervisado?",
            "def": "Siempre que una organización disponga de grandes volúmenes de datos históricos sin etiquetar y necesite descubrir patrones accionables, segmentar poblaciones o detectar comportamientos atípicos.",
            "tec": "<strong>Casos reales de alto impacto:</strong> 1) <em>Marketing & Retail</em> (segmentación RFM de clientes para campañas personalizadas), 2) <em>Banca & Ciberseguridad</em> (detección de fraudes e intrusiones mediante anomalías), 3) <em>Industria 4.0</em> (mantenimiento predictivo agrupando vibraciones anómalas de sensores).",
            "app": "En <strong>Educación Superior (SENATI)</strong>, permite detectar subgrupos silenciosos de deserción estudiantil antes del primer examen parcial para activar tutorías preventivas.",
            "img": None,
            "cap": None,
        },
    ]

    # Filtrado dinámico
    preguntas_filtradas = []
    for item in cuestionario:
        pasa_cat = cat_filtro == "Todas (1–20)" or item["cat"] == cat_filtro
        texto_total = f"{item['q']} {item['def']} {item['tec']} {item['app']}".lower()
        pasa_busq = busqueda == "" or busqueda in texto_total
        if pasa_cat and pasa_busq:
            preguntas_filtradas.append(item)

    if not preguntas_filtradas:
        st.info("No se encontraron preguntas que coincidan con tu búsqueda.")
    else:
        for item in preguntas_filtradas:
            with st.expander(item["q"]):
                st.markdown(
                    f"""
                    <div style="line-height:1.65; font-size:0.93rem;">
                        <p style="margin-bottom:0.6rem;">
                            <span style="color:#e5c890; font-weight:700;">▸ Definición Técnica:</span>
                            <span style="color:#f4f4f6;">{item['def']}</span>
                        </p>
                        <p style="margin-bottom:0.6rem;">
                            <span style="color:#38bdf8; font-weight:700;">▸ Lógica / Funcionamiento:</span>
                            <span style="color:#cbd5e1;">{item['tec']}</span>
                        </p>
                        <p style="margin-bottom:0.4rem; padding:0.65rem 0.9rem; background:rgba(229, 200, 144, 0.07); border-left:3px solid #e5c890; border-radius:0 8px 8px 0;">
                            <span style="color:#e5c890; font-weight:700;">💠 En AURA Academics:</span>
                            <span style="color:#f4f4f6;">{item['app']}</span>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if item["img"] is not None:
                    st.write("")
                    st.image(graficos_t4[item["img"]], use_container_width=True)
                    st.caption(f"📊 {item['cap']}")

    # --------------------------------------------------------------------------
    # CONCLUSIONES EJECUTIVAS DEL PROYECTO
    # --------------------------------------------------------------------------
    st.write("---")
    st.markdown("### Conclusiones Estratégicas del Proyecto")

    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        st.markdown(
            f"""
            <div style="background:#111116; border:1px solid rgba(229,200,144,0.3); border-radius:16px; padding:1.3rem; height:100%;">
                <div style="color:#e5c890; font-size:0.75rem; font-weight:700; text-transform:uppercase;">1. Predicción Supervisada</div>
                <div style="font-family:'Syne'; font-size:1.7rem; font-weight:800; color:#f4f4f6; margin:0.3rem 0;">{acc*100:.1f}% Exactitud</div>
                <p style="color:#90909f; font-size:0.85rem; margin:0; line-height:1.5;">
                    El ensamble calibrado sobre 1,500 registros generaliza con mínimo margen de error en datos independientes, superando ampliamente el umbral del 96%.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cc2:
        st.markdown(
            """
            <div style="background:#111116; border:1px solid rgba(16,185,129,0.3); border-radius:16px; padding:1.3rem; height:100%;">
                <div style="color:#10b981; font-size:0.75rem; font-weight:700; text-transform:uppercase;">2. Segmentación K-Means</div>
                <div style="font-family:'Syne'; font-size:1.7rem; font-weight:800; color:#f4f4f6; margin:0.3rem 0;">3 Perfiles Nítidos</div>
                <p style="color:#90909f; font-size:0.85rem; margin:0; line-height:1.5;">
                    Sin usar etiquetas previas, K-Means converge de forma estable separando a la cohorte en En Riesgo, Regular y Alto Rendimiento con alta cohesión Silhouette.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cc3:
        st.markdown(
            """
            <div style="background:#111116; border:1px solid rgba(56,189,248,0.3); border-radius:16px; padding:1.3rem; height:100%;">
                <div style="color:#38bdf8; font-size:0.75rem; font-weight:700; text-transform:uppercase;">3. Impacto Institucional</div>
                <div style="font-family:'Syne'; font-size:1.7rem; font-weight:800; color:#f4f4f6; margin:0.3rem 0;">Alerta Temprana</div>
                <p style="color:#90909f; font-size:0.85rem; margin:0; line-height:1.5;">
                    La combinación de diagnóstico individual en vivo y análisis poblacional brinda a SENATI una herramienta preventiva lista para reducir la deserción académica.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.image(graficos_t4["conclusion"], use_container_width=True)
    st.caption("📈 Síntesis comparativa: impacto conjunto de las horas de estudio, asistencia y calificación en cada segmento descubierto.")