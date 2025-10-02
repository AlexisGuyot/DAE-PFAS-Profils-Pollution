import os
import json
import math
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely import wkt as shapely_wkt

import streamlit as st
from streamlit_folium import st_folium
import folium

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

import matplotlib.pyplot as plt

# --- Optional libs
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    import orjson as _fastjson
    def _loads_fast(x):
        if isinstance(x, (list, dict)): return x
        if isinstance(x, str):
            try: return _fastjson.loads(x)
            except Exception: return None
        return None
except Exception:
    import json as _fastjson
    def _loads_fast(x):
        if isinstance(x, (list, dict)): return x
        if isinstance(x, str):
            try: return _fastjson.loads(x)
            except Exception: return None
        return None

# ----------------- Config -----------------
DATA_URL = "https://pdh.cnrs.fr/download/full.csv"
CSV_CACHE = "pdh_full_cached.csv"

# Visu seulement (n'affecte PAS les calculs)
MAP_MAX_POINTS = 12000

# Mémoire/CPU
USE_FLOAT32 = True

DEBUG = True  # set False after you’re stable

try:
    st.set_page_config(page_title="PDH – Clustering & Carte", layout="wide")
except Exception as e:
    st.error("Une erreur est survenue.")
    st.exception(e)

# -------------- Utils couleur ----------------
PALETTE = [
    "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
    "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
    "#393b79","#637939","#8c6d31","#843c39","#7b4173",
    "#3182bd","#e6550d","#31a354","#756bb1","#636363"
]
def color_for_category(cat):
    if cat is None or (isinstance(cat, float) and math.isnan(cat)):
        return "#555555"
    return PALETTE[abs(hash(str(cat))) % len(PALETTE)]

def color_for_label(label):
    if label == -1:
        return "#000000"
    return PALETTE[int(label) % len(PALETTE)]

# -------------- Chargement & GDF --------------
@st.cache_data(show_spinner=True)
def load_pdh(nrows=None, use_cache=True):
    # Prefer local cache if present
    if use_cache and os.path.exists(CSV_CACHE):
        try:
            return pd.read_csv(CSV_CACHE, engine="pyarrow")
        except Exception:
            return pd.read_csv(CSV_CACHE)  # fallback to default engine

    # Download
    try:
        try:
            df = pd.read_csv(DATA_URL, engine="pyarrow")
        except Exception:
            df = pd.read_csv(DATA_URL)     # fallback if pyarrow missing/unavailable
        if use_cache:
            try:
                df.to_csv(CSV_CACHE, index=False)
            except Exception:
                pass
        return df
    except Exception as e:
        st.error(f"Échec chargement CSV: {e}")
        raise

def try_infer_geometry(df: pd.DataFrame) -> gpd.GeoDataFrame:
    cols = {c.lower(): c for c in df.columns}
    if 'wkt' in cols:
        try:
            geom = df[cols['wkt']].apply(shapely_wkt.loads)
            return gpd.GeoDataFrame(df.copy(), geometry=geom, crs="EPSG:4326")
        except Exception:
            pass
    lat_candidates = [c for c in ['lat','latitude','y','lat_dd'] if c in cols]
    lon_candidates = [c for c in ['lon','longitude','x','lon_dd'] if c in cols]
    if lat_candidates and lon_candidates:
        lat_c = cols[lat_candidates[0]]
        lon_c = cols[lon_candidates[0]]
        geom = [Point(float(lon), float(lat)) for lat,lon in zip(df[lat_c], df[lon_c])]
        return gpd.GeoDataFrame(df.copy(), geometry=geom, crs="EPSG:4326")
    raise ValueError("Impossible d'inférer la géométrie (lat/lon ou WKT requis).")

# -------------- PFAS feature space --------------
def build_pfas_feature_space(gdf, unit):
    """Liste ordonnée des clés substance|suffix ('total' puis isomers triés)."""
    substance_to_isomers = {}
    series = gdf['pfas_parsed'] if 'pfas_parsed' in gdf.columns else gdf['pfas_values']
    for v in series:
        arr = v if isinstance(v, list) else _loads_fast(v)
        arr = arr if isinstance(arr, list) else ([arr] if isinstance(arr, dict) else [])
        for item in arr:
            if item.get("unit") != unit:
                continue
            sub = item.get("substance")
            iso = item.get("isomer")
            if not sub:
                continue
            substance_to_isomers.setdefault(sub, set())
            substance_to_isomers[sub].add(iso if iso else "total")
    keys = []
    for sub in sorted(substance_to_isomers.keys()):
        iso = substance_to_isomers[sub]
        if "total" in iso:
            keys.append(f"{sub}|total")
        for i in sorted([x for x in iso if x != "total"]):
            keys.append(f"{sub}|{i}")
    return keys

def pfas_profile_vector(pfas_value, unit, feature_keys):
    """Uniquement 'value'; ignore 'less_than'. Si aucune 'value', renvoie tout-NaN."""
    vec = {k: np.nan for k in feature_keys}
    arr = pfas_value if isinstance(pfas_value, list) else _loads_fast(pfas_value)
    arr = arr if isinstance(arr, list) else ([arr] if isinstance(arr, dict) else [])
    for item in arr:
        if item.get("unit") != unit:
            continue
        sub = item.get("substance")
        iso = item.get("isomer")
        suffix = iso if iso else "total"
        key = f"{sub}|{suffix}"
        if key not in vec:
            continue
        val = item.get("value")
        if val is None or val == "":
            continue
        try:
            vec[key] = float(val)
        except Exception:
            pass
    return [vec[k] for k in feature_keys]

# -------------- Features pour clustering --------------
def prepare_features(df_num: pd.DataFrame, use_scaler=True, reduction='None', reduction_dim=2):
    X = df_num.values.astype(np.float32) if USE_FLOAT32 else df_num.values.astype(float)
    scaler = StandardScaler() if use_scaler else None
    if scaler is not None:
        X = scaler.fit_transform(X)
    reducer = None
    if reduction == 'PCA':
        reducer = PCA(n_components=reduction_dim, random_state=42)
        X = reducer.fit_transform(X)
    return X, scaler, reducer

def run_clustering(algo, X, k=6, db_eps=0.5, db_min_samples=5):
    if algo == 'K-Means':
        km = KMeans(
            n_clusters=k, n_init=10, random_state=42,
            algorithm='elkan', init='k-means++'
        )
        labels = km.fit_predict(X)
        centroids = {lab: km.cluster_centers_[lab] for lab in range(k)}
    elif algo == 'DBSCAN':
        db = DBSCAN(
            eps=db_eps, min_samples=db_min_samples,
            metric='euclidean', algorithm='ball_tree', leaf_size=40
        )
        labels = db.fit_predict(X)
        centroids = {}
        for lab in sorted(set(labels)):
            if lab == -1:
                continue
            m = (labels == lab)
            centroids[lab] = X[m].mean(axis=0)
    else:
        # fallback KMeans
        km = KMeans(n_clusters=k, n_init=10, random_state=42, algorithm='elkan', init='k-means++')
        labels = km.fit_predict(X)
        centroids = {lab: km.cluster_centers_[lab] for lab in range(k)}
    return labels, centroids

def pairwise_centroid_distances(centroids: dict):
    labs = sorted(centroids.keys())
    if not labs:
        return pd.DataFrame()
    M = np.zeros((len(labs), len(labs)), dtype=np.float32 if USE_FLOAT32 else float)
    for i, a in enumerate(labs):
        for j, b in enumerate(labs):
            M[i, j] = 0.0 if i==j else float(np.linalg.norm(centroids[a] - centroids[b]))
    return pd.DataFrame(M, index=labs, columns=labs)

# -------------------- UI --------------------
st.title("PDH – Carte interactive & clustering (appli web)")

with st.sidebar:
    st.subheader("1) Données")
    use_cache = st.checkbox("Utiliser le cache local", value=True)
    df = load_pdh(use_cache=use_cache)
    st.caption(f"CSV : {len(df):,} lignes, {len(df.columns)} colonnes")

    st.subheader("1.0) Carte avant nettoyage")
    map_color_by = st.selectbox("Couleur par (brut)", options=["None", "matrix"], index=1)

# Construire GDF brut (avant nettoyage)
@st.cache_data(show_spinner=True)
def make_gdf_raw(df_in: pd.DataFrame):
    gdf_raw = try_infer_geometry(df_in)
    return gdf_raw

try:
    gdf_raw = make_gdf_raw(df)
except Exception as e:
    st.error(f"Erreur GeoDataFrame (brut) : {e}")
    st.stop()

# Carte AVANT nettoyage
def render_map(gdf, colorby=None, title="Carte"):
    st.markdown(f"### {title}")
    gtmp = gdf[gdf.geometry.notna() & (gdf.geometry.geom_type == 'Point')].copy()
    if gtmp.empty:
        st.info("Aucun point à afficher.")
        return None

    center = [float(gtmp.geometry.y.median()), float(gtmp.geometry.x.median())]
    fmap = folium.Map(location=center, zoom_start=6, tiles="cartodbpositron")
    # limiter l'affichage pour la fluidité (n'affecte pas les calculs)
    shown = min(len(gtmp), MAP_MAX_POINTS)
    gshow = gtmp.iloc[:shown].copy()

    # couleur
    if colorby and colorby in gshow.columns:
        cats = gshow[colorby]
    else:
        cats = pd.Series([None]*len(gshow), index=gshow.index)

    for idx, row in gshow.iterrows():
        geom = row.geometry
        col = color_for_category(cats.loc[idx]) if colorby else "#3366cc"
        folium.CircleMarker(
            location=(geom.y, geom.x),
            radius=2.5,
            color=col,
            fill=True, fill_opacity=0.7,
            tooltip=f"id={row.get('id','?')}"
        ).add_to(fmap)

    st.caption(f"Affichés : {shown} / {len(gtmp)} points (carte bornée à {MAP_MAX_POINTS} pour la fluidité)")
    out = st_folium(fmap, width=None, height=600)
    return out

_ = render_map(gdf_raw, colorby=(None if map_color_by == "None" else map_color_by), title="Carte (AVANT nettoyage)")

# ---------------- Nettoyage ----------------
st.subheader("1.1) Nettoyage des données")
with st.expander("Étapes de nettoyage (appliquées à GDF_RAW)", expanded=False):
    st.markdown("- Suppression géométries manquantes ou non-Point\n- (Ajouter ici vos règles spécifiques si besoin)")

@st.cache_data(show_spinner=True)
def clean_gdf(gdf_in: gpd.GeoDataFrame):
    gdf = gdf_in.copy()
    gdf = gdf.dropna(subset=['geometry'])
    gdf = gdf[gdf.geometry.geom_type == 'Point'].copy()
    # parser pfas_values une fois
    if 'pfas_values' in gdf.columns:
        gdf['pfas_parsed'] = gdf['pfas_values'].map(_loads_fast)
    # colonnes catégorielles utiles
    for col in ['matrix','source','unit']:
        if col in gdf.columns:
            gdf[col] = gdf[col].astype('category')
    return gdf

gdf = clean_gdf(gdf_raw)
st.caption(f"GDF nettoyé : {len(gdf):,} points")

# Carte APRÈS nettoyage
render_map(gdf, colorby=(None if map_color_by == "None" else map_color_by), title="Carte (APRÈS nettoyage)")

# ---------------- PFAS Features ----------------
st.sidebar.subheader("2) Filtrage PFAS")
matrices = sorted([x for x in gdf['matrix'].dropna().unique().tolist()]) if 'matrix' in gdf.columns else []
units = []
if 'pfas_parsed' in gdf.columns:
    # extraire unités
    U = set()
    for v in gdf['pfas_parsed']:
        arr = v if isinstance(v, list) else []
        for it in arr:
            u = it.get("unit")
            if u: U.add(u)
    units = sorted(U)

sel_matrix = st.sidebar.selectbox("matrix", options=matrices) if matrices else None
sel_unit = st.sidebar.selectbox("unit", options=units) if units else None

st.subheader("2) Préparation du profil moléculaire PFAS")
if not sel_matrix or not sel_unit:
    st.info("Choisissez d'abord la matrix et l'unit dans la barre latérale.")
    st.stop()

gdf_f = gdf[gdf['matrix'] == sel_matrix].copy()
if gdf_f.empty:
    st.error(f"Aucune ligne pour matrix={sel_matrix}")
    st.stop()

feature_keys = build_pfas_feature_space(gdf_f, sel_unit)
if not feature_keys:
    st.error(f"Aucune clé de feature PFAS pour unit={sel_unit}")
    st.stop()

rows = []
idx_keep = []
for idx, row in gdf_f.iterrows():
    vec = pfas_profile_vector(row['pfas_parsed'] if 'pfas_parsed' in gdf_f.columns else row['pfas_values'], sel_unit, feature_keys)
    # garder la ligne si au moins une 'value'
    if all((v is None) or (isinstance(v, float) and np.isnan(v)) for v in vec):
        continue
    rows.append(vec)
    idx_keep.append(idx)

if not rows:
    st.error("Aucun point avec 'value' pour ces filtres.")
    st.stop()

PFAS = pd.DataFrame(rows, index=idx_keep, columns=feature_keys)
gdf_model = gdf_f.loc[idx_keep].copy()

st.caption(f"PFAS features: shape = {PFAS.shape}")
with st.expander("Aperçu PFAS (têtes)", expanded=False):
    st.dataframe(PFAS.head(10))

# ------------ Sélection des features PFAS ------------
st.subheader("3) Sélection des features PFAS (colonnes)")
nz = (PFAS != 0).sum(axis=0)
var = PFAS.var(axis=0)
stats = pd.DataFrame({"non_zero": nz, "variance": var}).sort_values("variance", ascending=False)

left, right = st.columns([2,1])
with left:
    default_sel = PFAS.columns[:min(30, PFAS.shape[1])].tolist()
    selected_feats = st.multiselect(
        "Colonnes utilisées pour caractériser les points (aucun échantillonnage de lignes)",
        options=PFAS.columns.tolist(),
        default=default_sel
    )
with right:
    k_var = st.number_input("Top-k variance", 1, int(PFAS.shape[1]), min(30, PFAS.shape[1]) if PFAS.shape[1]>=30 else PFAS.shape[1])
    if st.button("Sélectionner top-k variance"):
        selected_feats = stats.head(int(k_var)) .index.tolist()
    k_nz = st.number_input("Top-k non_zero", 1, int(PFAS.shape[1]), min(30, PFAS.shape[1]) if PFAS.shape[1]>=30 else PFAS.shape[1])
    if st.button("Sélectionner top-k non_zero"):
        selected_feats = nz.sort_values(ascending=False).head(int(k_nz)).index.tolist()

if not selected_feats:
    st.warning("Sélectionnez au moins une feature PFAS.")
    st.stop()

PFAS_sel = PFAS[selected_feats].astype(np.float32) if USE_FLOAT32 else PFAS[selected_feats].astype(float)

# ---------------- Clustering ----------------
st.subheader("4) Clustering")
algo = st.selectbox("Algorithme", options=["K-Means","DBSCAN","Autre"], index=0)
use_pca = st.checkbox("Réduction PCA", value=False)
n_comp = st.slider("Dimensions PCA", 2, min(30, max(2, len(selected_feats))), 10) if use_pca else 0
use_scaler = st.checkbox("StandardScaler", value=True)

k = st.slider("k (K-Means)", 2, 30, 6) if algo in ["K-Means","Autre"] else None
eps = st.number_input("eps (DBSCAN)", value=0.5, min_value=0.0001, step=0.1, format="%.4f") if algo=="DBSCAN" else None
min_samples = st.number_input("min_samples (DBSCAN)", value=5, min_value=1, step=1) if algo=="DBSCAN" else None

with st.spinner("Préparation des features…"):
    X, scaler, reducer = prepare_features(PFAS_sel, use_scaler=use_scaler, reduction=('PCA' if use_pca else 'None'), reduction_dim=n_comp if use_pca else 2)
with st.spinner("Clustering…"):
    labels, centroids = run_clustering(algo, X, k=k or 6, db_eps=eps or 0.5, db_min_samples=min_samples or 5)

st.caption(f"Clusters (valeurs): {dict(zip(*np.unique(np.array(labels), return_counts=True)))}")

# conserver dans la session (pour navigation & explicabilité)
if "current_cluster" not in st.session_state:
    st.session_state.current_cluster = None

# ---------------- Carte résultats & filtre ----------------
st.subheader("5) Carte – résultats du clustering")
lab_series = pd.Series(labels, index=PFAS_sel.index)
gdf_model["__label__"] = lab_series

# UI filtrage
all_labels = sorted(list(set(labels)))
sel_labs = st.multiselect("Clusters affichés", options=[str(l) for l in all_labels], default=[str(l) for l in all_labels])
sel_labs_int = set([int(x) for x in sel_labs]) if sel_labs else set(all_labels)

# Carte
def render_cluster_map(gdf_in: gpd.GeoDataFrame, selected_labels: set):
    gtmp = gdf_in[gdf_in["__label__"].isin(selected_labels)].copy()
    if gtmp.empty:
        st.info("Aucun point à afficher pour ce filtre.")
        return None
    center = [float(gtmp.geometry.y.median()), float(gtmp.geometry.x.median())]
    fmap = folium.Map(location=center, zoom_start=6, tiles="cartodbpositron")

    # borne d'affichage seulement
    shown = min(len(gtmp), MAP_MAX_POINTS)
    gshow = gtmp.iloc[:shown].copy()

    for idx, row in gshow.iterrows():
        lab = int(row["__label__"])
        col = color_for_label(lab)
        geom = row.geometry
        # popup/tooltip encode le label pour récupérer au clic
        folium.CircleMarker(
            location=(geom.y, geom.x),
            radius=3, color=col, fill=True, fill_opacity=0.8,
            tooltip=f"cluster={lab}"
        ).add_to(fmap)

    st.caption(f"Affichés : {shown} / {len(gtmp)} points (affichage borné, calculs intégraux)")
    out = st_folium(fmap, width=None, height=650)
    return out

map_out = render_cluster_map(gdf_model, sel_labs_int)

# Clic → sélection du cluster courant
if map_out and map_out.get("last_object_clicked_tooltip"):
    tip = map_out["last_object_clicked_tooltip"]
    if tip and "cluster=" in tip:
        try:
            selected_lab = int(tip.split("cluster=")[-1])
            st.session_state.current_cluster = selected_lab
            # mettre à jour la sélection UI (affichage du seul cluster)
            sel_labs_int = {selected_lab}
        except Exception:
            pass

st.write(f"**CURRENT_CLUSTER_LABEL** = {st.session_state.current_cluster}")

# ---------------- Infos cluster ----------------
st.subheader("6) Infos sur le cluster sélectionné")
cur = st.session_state.current_cluster
if cur is None or cur not in set(labels):
    st.info("Clique un point sur la carte ou sélectionne un seul cluster pour fixer la sélection.")
else:
    mask = (labels == cur)
    size = int(mask.sum())
    st.write(f"Cluster **{cur}** — taille = **{size}**")

    # centroïde courant
    if cur in centroids:
        c = centroids[cur]
        st.write("Centroïde (espace transformé) :")
        st.write(np.array(c))

    # distances intra au centroïde
    if cur in centroids:
        c = centroids[cur]
        D = np.linalg.norm(X[mask] - c.reshape(1, -1), axis=1)
        s = pd.Series(D)
        st.write("Statistiques des distances intra→centroïde :")
        st.write(s.describe())

    # distances aux autres centroïdes
    inter = pairwise_centroid_distances(centroids)
    if not inter.empty and cur in inter.index:
        st.write("Distances aux autres centroïdes :")
        st.dataframe(inter.loc[cur].drop(labels=[cur]).sort_values())

# ---------------- Explicabilité ----------------
st.subheader("6 bis) Explicabilité (impact des PFAS sur l'appartenance au cluster)")
if cur is None or cur not in set(labels):
    st.info("Sélectionne d'abord un cluster (clic sur la carte).")
else:
    # espace PFAS brut sur les features sélectionnées
    feat_cols = selected_feats
    X_full = PFAS_sel[feat_cols].values.astype(np.float32) if USE_FLOAT32 else PFAS_sel[feat_cols].values
    y_full = (labels == cur).astype(int)

    # Modèle surrogate arbre
    model = Pipeline([
        ('scaler', StandardScaler(with_mean=True, with_std=True)),
        ('clf', RandomForestClassifier(
            n_estimators=150, max_depth=None, random_state=42, n_jobs=-1,
            class_weight='balanced_subsample'
        ))
    ])
    with st.spinner("Entraînement du modèle surrogate…"):
        model.fit(X_full, y_full)

    if SHAP_AVAILABLE:
        try:
            # SHAP TreeExplainer sur la forêt
            from sklearn import set_config
            set_config(transform_output="default")
            X_back = model[:-1].transform(X_full)  # standardisé
            est = model[-1]
            explainer = shap.TreeExplainer(est, feature_perturbation="interventional")
            sv = explainer.shap_values(X_back)
            sv_pos = sv[1] if isinstance(sv, list) else sv
            mean_abs = np.abs(sv_pos).mean(axis=0)
            imp_df = pd.DataFrame({"feature": feat_cols, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)

            st.write("Top 25 (|SHAP| moyen) :")
            fig, ax = plt.subplots(figsize=(6, 10))
            topk = min(25, len(feat_cols))
            ax.barh(imp_df["feature"].head(topk)[::-1], imp_df["mean_abs_shap"].head(topk)[::-1])
            ax.set_title(f"Impact (|SHAP|) – cluster {cur}")
            ax.set_xlabel("|SHAP| moyen")
            st.pyplot(fig)
            st.dataframe(imp_df.head(50))
        except Exception as e:
            st.warning(f"SHAP indisponible/échoué ({e}). Fallback permutation_importance…")
            result = permutation_importance(model, X_full, y_full, n_repeats=10, random_state=42, n_jobs=-1)
            imp_df = pd.DataFrame({
                "feature": feat_cols,
                "perm_importance_mean": result.importances_mean,
                "perm_importance_std": result.importances_std
            }).sort_values("perm_importance_mean", ascending=False)
            fig, ax = plt.subplots(figsize=(6, 10))
            topk = min(25, len(feat_cols))
            ax.barh(imp_df["feature"].head(topk)[::-1], imp_df["perm_importance_mean"].head(topk)[::-1])
            ax.set_title(f"Impact (permutation) – cluster {cur}")
            ax.set_xlabel("Importance moyenne")
            st.pyplot(fig)
            st.dataframe(imp_df.head(50))
    else:
        st.info("SHAP non installé → permutation_importance")
        result = permutation_importance(model, X_full, y_full, n_repeats=10, random_state=42, n_jobs=-1)
        imp_df = pd.DataFrame({
            "feature": feat_cols,
            "perm_importance_mean": result.importances_mean,
            "perm_importance_std": result.importances_std
        }).sort_values("perm_importance_mean", ascending=False)
        fig, ax = plt.subplots(figsize=(6, 10))
        topk = min(25, len(feat_cols))
        ax.barh(imp_df["feature"].head(topk)[::-1], imp_df["perm_importance_mean"].head(topk)[::-1])
        ax.set_title(f"Impact (permutation) – cluster {cur}")
        ax.set_xlabel("Importance moyenne")
        st.pyplot(fig)
        st.dataframe(imp_df.head(50))
