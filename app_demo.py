from __future__ import annotations

import re
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
import zipfile
import hashlib

import joblib
import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

from run import (
    detect_events_for_audio,
    export_events_csv,
    extract_features,
    preprocess_audio,
    read_audio_row,
    load_species_dataset,
    window_audio,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = APP_DIR / "best_model.joblib"

DEFAULT_PART2_ANNOTATIONS_DIR = (
    APP_DIR / "Ressources du défi #1" / "Partie #2" / "Longues séquences" / "annotations"
)

UA = "Mozilla/5.0"


SPECIES_INFO = {
    "Beluga_WhiteWhale": {"nom": "Béluga"},
    "Fin_FinbackWhale": {"nom": "Rorqual commun"},
    "HumpbackWhale": {"nom": "Baleine à bosse"},
    "SpermWhale": {"nom": "Cachalot"},
    "White_sidedDolphin": {"nom": "Dauphin à flancs blancs"},
    "Noise": {"nom": "Bruit"},
}


@st.cache_resource
def charger_modele(path: str):
    return joblib.load(path)


def charger_annotation_officielle_sequence(nom_fichier, annotations_dir: Path):
    stem = Path(nom_fichier).stem
    m = re.match(r"sequence_(\d+)", stem)
    if not m:
        return None, None

    csv_path = annotations_dir / f"{stem}.csv"
    if not csv_path.exists():
        return None, csv_path

    try:
        df = pd.read_csv(csv_path)
        return df, csv_path
    except Exception:
        return None, csv_path


def predict_clip_probs(y, sr, bundle, window_sec):
    model = bundle["model"]
    labels = list(bundle["labels"])

    y_p, sr_p = preprocess_audio(y, sr, target_sr=16000)
    windows, _ = window_audio(y_p, sr_p, window_sec, window_sec)

    X = np.vstack([extract_features(w, sr_p, target_sr=sr_p) for w in windows])
    probs = model.predict_proba(X)

    return labels, probs.mean(axis=0)


def main():
    st.set_page_config(page_title="IA'Hack 2026", layout="wide")
    st.title("IA'Hack 2026 — Démo")

    # Sidebar
    with st.sidebar:
        model_path = st.text_input("Modèle", value=str(DEFAULT_MODEL))
        annotations_path = st.text_input(
            "Annotations Partie II",
            value=str(DEFAULT_PART2_ANNOTATIONS_DIR),
        )

    try:
        bundle = charger_modele(model_path)
    except Exception as e:
        st.error(f"Erreur modèle : {e}")
        return

    labels = bundle.get("labels", [])

    tab1, tab2 = st.tabs(["Analyse simple", "Partie II (CLI)"])

    # =========================
    # 🔹 PARTIE 1
    # =========================
    with tab1:
        st.markdown("## Partie 1 — évaluation (Accuracy / Precision / Recall / F1 / matrice)")
        with st.expander("Évaluer sur le dataset test (Base de données)", expanded=False):
            species_path = st.text_input(
                "Chemin Partie #1/Base de données (dossier ou .zip)",
                value=str(APP_DIR / "Base de données"),
                help="Doit contenir `train/` et `test/` comme dans le défi, ou être un zip.",
            )
            eval_window = st.slider(
                "Fenêtre pour la prédiction (s)",
                0.25,
                2.0,
                float(bundle.get("window_sec", 0.5)),
                0.25,
                help="Comme la Partie 1 : on fenêtrise puis on moyenne les probas.",
            )
            run_eval = st.button("Calculer les métriques", use_container_width=True)

        if run_eval:
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
            import zipfile

            with st.spinner("Chargement du dataset + prédictions…"):
                df = load_species_dataset(species_path)
                df_test = df[df["split"].eq("test")].reset_index(drop=True)
                if df_test.empty:
                    st.error("Aucun fichier trouvé dans `test/` (vérifie le chemin).")
                    return

                zf_species = zipfile.ZipFile(species_path) if str(species_path).lower().endswith(".zip") else None
                y_true: list[str] = []
                y_pred: list[str] = []

                for _, row in df_test.iterrows():
                    y_a, sr_a = read_audio_row(row, zf_species=zf_species, zf_noise=None)
                    labs, p = predict_clip_probs(y_a, sr_a, bundle, float(eval_window))
                    pred_lab = labs[int(np.argmax(p))]
                    # On évalue Partie 1 uniquement sur les espèces (pas Noise)
                    if pred_lab == "Noise":
                        # forcer un choix parmi espèces pour éviter d’écraser les scores
                        # (le modèle peut contenir Noise mais la vérité Partie 1 n’en a pas)
                        order = np.argsort(-p)
                        pred_lab = next((labs[int(i)] for i in order if labs[int(i)] != "Noise"), pred_lab)
                    y_true.append(str(row["label"]))
                    y_pred.append(str(pred_lab))

                if zf_species is not None:
                    zf_species.close()

            acc = float(accuracy_score(y_true, y_pred))
            st.metric("Accuracy", f"{acc:.3f}")

            report = classification_report(
                y_true,
                y_pred,
                labels=sorted(set(y_true)),
                output_dict=True,
                zero_division=0,
            )
            df_rep = pd.DataFrame(report).T
            # On met en avant macro avg (souvent attendu)
            if "macro avg" in df_rep.index:
                st.markdown(
                    f"**Macro avg** — Precision: **{df_rep.loc['macro avg','precision']:.3f}** · "
                    f"Recall: **{df_rep.loc['macro avg','recall']:.3f}** · "
                    f"F1: **{df_rep.loc['macro avg','f1-score']:.3f}**"
                )
            st.dataframe(df_rep, use_container_width=True)

            labels_order = sorted(set(y_true))
            cm = confusion_matrix(y_true, y_pred, labels=labels_order)
            st.markdown("**Matrice de confusion**")
            st.dataframe(
                pd.DataFrame(cm, index=labels_order, columns=labels_order),
                use_container_width=True,
            )

        st.divider()
        st.markdown("## Partie 1 — test rapide (un fichier)")
        uploaded = st.file_uploader("Audio", type=["wav", "mp3", "flac"])

        if not uploaded:
            return

        import soundfile as sf

        y, sr = sf.read(uploaded)

        if y.ndim > 1:
            y = y.mean(axis=1)

        st.audio(uploaded)

        labs, probs = predict_clip_probs(y, sr, bundle, 0.5)

        df = pd.DataFrame({
            "Classe": labs,
            "Probabilité": probs
        }).sort_values("Probabilité", ascending=False)

        st.bar_chart(df.set_index("Classe"))
        st.metric("Prédiction", df.iloc[0]["Classe"])

    # =========================
    # 🔹 PARTIE 2 (CORRIGÉE)
    # =========================
    with tab2:
        uploaded2 = st.file_uploader("Uploader sequence_XX.wav", type=["wav"])

        if not uploaded2:
            return

        import soundfile as sf

        y2, sr2 = sf.read(uploaded2)
        if y2.ndim > 1:
            y2 = y2.mean(axis=1)

        st.audio(uploaded2)

        # 📄 Annotation officielle
        df_ref, _ = charger_annotation_officielle_sequence(
            uploaded2.name,
            Path(annotations_path)
        )

        if df_ref is not None:
            st.write("Annotation officielle")
            st.dataframe(df_ref)

        st.markdown("## Prédiction CLI")

        with st.expander("Paramètres détection (comme le terminal)", expanded=False):
            window_sec = st.slider("window_sec", 0.25, 3.0, float(bundle.get("window_sec", 0.5)), 0.25)
            hop_sec = st.slider("hop_sec", 0.05, 2.0, float(bundle.get("hop_sec_detect", 0.25)), 0.05)
            prob_threshold = st.slider("prob_threshold", 0.05, 0.95, float(bundle.get("prob_threshold", 0.60)), 0.01)
            margin_vs_noise = st.slider("margin_vs_noise", 0.0, 0.5, float(bundle.get("margin_vs_noise", 0.10)), 0.01)
            min_event_sec = st.slider("min_event_sec", 0.0, 5.0, float(bundle.get("min_event_sec", 0.50)), 0.05)
            max_gap_sec = st.slider("max_gap_sec", 0.0, 2.0, float(bundle.get("max_gap_sec", 0.25)), 0.05)
            smooth_k = st.slider("smooth_k", 1, 25, 5, 1)

        col1, col2 = st.columns(2)
        with col1:
            out_dir = st.text_input("output_dir", "best_predictions")
        with col2:
            save_debug = st.checkbox("save_debug")

        run_btn = st.button("Lancer prédiction")

        st.code(
            f'py run.py predict --model_path "{model_path}" '
            f'--audio_path "{uploaded2.name}" '
            f'--output_dir "{out_dir}" '
            + ("--save_debug" if save_debug else "")
        )

        if run_btn:
            with st.spinner("Processing..."):

                events, debug = detect_events_for_audio(
                    y2,
                    sr2,
                    bundle,
                    window_sec=float(window_sec),
                    hop_sec=float(hop_sec),
                    prob_threshold=float(prob_threshold),
                    margin_vs_noise=float(margin_vs_noise),
                    min_event_sec=float(min_event_sec),
                    max_gap_sec=float(max_gap_sec),
                    smooth_k=int(smooth_k),
                    return_debug=save_debug,
                )

            df_pred = pd.DataFrame(events)

            if df_pred.empty:
                df_pred = pd.DataFrame(columns=[
                    "animal", "start_sec", "end_sec", "duration_sec"
                ])

            out_dir_path = Path(out_dir)
            out_dir_path.mkdir(parents=True, exist_ok=True)

            stem = Path(uploaded2.name).stem

            export_events_csv(events, out_dir_path / f"{stem}.csv")

            if save_debug and debug is not None:
                debug.to_csv(out_dir_path / f"{stem}_debug_windows.csv", index=False)

            st.success("Fichiers générés")

            st.download_button(
                "Télécharger CSV",
                df_pred.to_csv(index=False),
                file_name=f"{stem}.csv"
            )

            st.dataframe(df_pred)


if __name__ == "__main__":
    main()