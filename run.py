import argparse
import io
import json
import math
import random
import warnings
import zipfile
from collections import Counter
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal, stats
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(42)
random.seed(42)
np.random.seed(42)

# =========================================================
# I/O
# =========================================================

def is_zip_path(path: str) -> bool:
    return str(path).lower().endswith('.zip')


def list_wavs_in_dir(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob('*') if p.is_file() and p.suffix.lower() == '.wav'])

def load_species_dataset_from_zip(zip_path: str) -> pd.DataFrame:
    zf = zipfile.ZipFile(zip_path)
    rows = []
    for name in zf.namelist():
        if not name.lower().endswith('.wav'):
            continue
        parts = name.split('/')
        if len(parts) < 4:
            continue
        split = parts[2]
        label = parts[3]
        if split not in {'train', 'test'}:
            continue
        rows.append({
            'source_zip': 'species',
            'path_in_zip': name,
            'split': split,
            'label': label,
            'group_id': name.split('::')[0],
        })
    return pd.DataFrame(rows).reset_index(drop=True)


def load_species_dataset_from_dir(base_dir: str) -> pd.DataFrame:
    root = Path(base_dir)
    rows = []
    for split in ['train', 'test']:
        split_dir = root / split
        if not split_dir.exists():
            continue
        for label_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
            for wav in list_wavs_in_dir(label_dir):
                rows.append({
                    'source_zip': 'species_file',
                    'path_in_zip': str(wav),
                    'split': split,
                    'label': label_dir.name,
                    'group_id': wav.stem.split('::')[0],
                })
    return pd.DataFrame(rows).reset_index(drop=True)


def load_species_dataset(species_path: str) -> pd.DataFrame:
    if is_zip_path(species_path):
        return load_species_dataset_from_zip(species_path)
    return load_species_dataset_from_dir(species_path)


def load_noise_dataset_from_zip(zip_path: str, test_ratio: float = 0.30) -> pd.DataFrame:
    zf = zipfile.ZipFile(zip_path)
    wavs = sorted([n for n in zf.namelist() if n.lower().endswith('.wav')])
    rows = []
    n_test = max(1, int(round(len(wavs) * test_ratio))) if wavs else 0
    n_train = max(0, len(wavs) - n_test)
    for i, name in enumerate(wavs):
        split = 'train' if i < n_train else 'test'
        rows.append({
            'source_zip': 'noise',
            'path_in_zip': name,
            'split': split,
            'label': 'Noise',
            'group_id': name.split('::')[0],
        })
    return pd.DataFrame(rows).reset_index(drop=True)


def load_noise_dataset_from_dir(base_dir: str, test_ratio: float = 0.30) -> pd.DataFrame:
    root = Path(base_dir)
    rows = []
    train_dir = root / 'train'
    test_dir = root / 'test'

    # Handles both:
    # - root/train/*.wav and root/test/*.wav
    # - root/train/noise/*.wav and root/test/noise/*.wav
    if train_dir.exists() or test_dir.exists():
        for split, split_dir in [('train', train_dir), ('test', test_dir)]:
            if not split_dir.exists():
                continue
            for wav in list_wavs_in_dir(split_dir):
                rows.append({
                    'source_zip': 'noise_file',
                    'path_in_zip': str(wav),
                    'split': split,
                    'label': 'Noise',
                    'group_id': wav.stem.split('::')[0],
                })
        return pd.DataFrame(rows).reset_index(drop=True)

    wavs = list_wavs_in_dir(root)
    n_test = max(1, int(round(len(wavs) * test_ratio))) if wavs else 0
    n_train = max(0, len(wavs) - n_test)
    for i, wav in enumerate(wavs):
        split = 'train' if i < n_train else 'test'
        rows.append({
            'source_zip': 'noise_file',
            'path_in_zip': str(wav),
            'split': split,
            'label': 'Noise',
            'group_id': wav.stem.split('::')[0],
        })
    return pd.DataFrame(rows).reset_index(drop=True)


def load_noise_dataset(noise_path: str, test_ratio: float = 0.30) -> pd.DataFrame:
    if is_zip_path(noise_path):
        return load_noise_dataset_from_zip(noise_path, test_ratio=test_ratio)
    return load_noise_dataset_from_dir(noise_path, test_ratio=test_ratio)


def read_wav_from_zip(zf: zipfile.ZipFile, path_in_zip: str):
    with zf.open(path_in_zip) as f:
        y, sr = sf.read(io.BytesIO(f.read()), always_2d=False)
    if np.ndim(y) > 1:
        y = np.mean(y, axis=1)
    return np.asarray(y, dtype=np.float32), int(sr)


def read_wav_file(path: str):
    y, sr = sf.read(path, always_2d=False)
    if np.ndim(y) > 1:
        y = np.mean(y, axis=1)
    return np.asarray(y, dtype=np.float32), int(sr)


def read_audio_row(row: pd.Series, zf_species: zipfile.ZipFile | None, zf_noise: zipfile.ZipFile | None):
    if row['source_zip'] == 'species':
        return read_wav_from_zip(zf_species, row['path_in_zip'])
    if row['source_zip'] == 'noise':
        return read_wav_from_zip(zf_noise, row['path_in_zip'])
    return read_wav_file(row['path_in_zip'])


# =========================================================
# Audio preprocessing / features
# =========================================================

def preprocess_audio(y: np.ndarray, sr: int, target_sr: int = 16000):
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if len(y) == 0:
        return np.zeros(1, dtype=np.float32), target_sr
    y = y - np.mean(y)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / (peak + 1e-9)
    if sr != target_sr:
        gcd = np.gcd(int(sr), int(target_sr))
        y = signal.resample_poly(y, target_sr // gcd, sr // gcd).astype(np.float32)
        sr = target_sr
    return y, sr


def pad_or_trim(y: np.ndarray, target_len: int) -> np.ndarray:
    if len(y) == target_len:
        return y.astype(np.float32)
    if len(y) > target_len:
        return y[:target_len].astype(np.float32)
    return np.pad(y, (0, target_len - len(y)), mode='constant').astype(np.float32)


def window_audio(y: np.ndarray, sr: int, window_sec: float, hop_sec: float):
    y, sr = preprocess_audio(y, sr)
    win = max(1, int(round(window_sec * sr)))
    hop = max(1, int(round(hop_sec * sr)))
    if len(y) <= win:
        return [pad_or_trim(y, win)], sr
    out = []
    starts = list(range(0, len(y) - win + 1, hop))
    if starts[-1] != len(y) - win:
        starts.append(len(y) - win)
    for s in starts:
        out.append(y[s:s + win].astype(np.float32))
    return out, sr


def extract_features(y: np.ndarray, sr: int, target_sr: int = 16000) -> np.ndarray:
    y, sr = preprocess_audio(y, sr, target_sr)
    features = []

    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features.extend([
        float(np.mean(rms)), float(np.std(rms)), float(np.mean(rms ** 2)),
        float(np.mean(zcr)), float(np.std(zcr)), float(len(y) / sr),
    ])

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    for mat in [mfcc, delta, delta2]:
        features.extend(np.mean(mat, axis=1).tolist())
        features.extend(np.std(mat, axis=1).tolist())

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=y)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for mat in [centroid, bandwidth, rolloff, flatness, contrast, chroma]:
        features.extend(np.mean(mat, axis=1).tolist())
        features.extend(np.std(mat, axis=1).tolist())

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel)
    features.extend(np.mean(mel_db, axis=1).tolist())
    features.extend(np.std(mel_db, axis=1).tolist())

    try:
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        features.extend(np.mean(tonnetz, axis=1).tolist())
        features.extend(np.std(tonnetz, axis=1).tolist())
    except Exception:
        features.extend([0.0] * 12)

    if len(rms) > 2:
        features.extend([float(stats.skew(rms)), float(stats.kurtosis(rms))])
    else:
        features.extend([0.0, 0.0])

    return np.asarray(features, dtype=np.float32)


# =========================================================
# Augmentation with real noise
# =========================================================

def fit_length(y: np.ndarray, target_len: int) -> np.ndarray:
    if len(y) == 0:
        return np.zeros(target_len, dtype=np.float32)
    if len(y) >= target_len:
        if len(y) == target_len:
            return y.astype(np.float32)
        start = RNG.integers(0, len(y) - target_len + 1)
        return y[start:start + target_len].astype(np.float32)
    reps = int(np.ceil(target_len / len(y)))
    tiled = np.tile(y, reps)
    return tiled[:target_len].astype(np.float32)


def mix_with_noise(y: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    noise = fit_length(noise, len(y))
    sig_power = float(np.mean(y ** 2) + 1e-9)
    noise_power = float(np.mean(noise ** 2) + 1e-9)
    desired_noise_power = sig_power / (10 ** (snr_db / 10.0))
    scale = math.sqrt(desired_noise_power / noise_power)
    mixed = y + scale * noise
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed = mixed / (peak + 1e-9)
    return mixed.astype(np.float32)


def augment_species_window(y: np.ndarray, noise_bank: list[np.ndarray]) -> list[np.ndarray]:
    out = [y]
    out.append(np.clip(y * 0.90, -1, 1).astype(np.float32))
    out.append(np.clip(y * 1.10, -1, 1).astype(np.float32))
    if noise_bank:
        n1 = noise_bank[RNG.integers(0, len(noise_bank))]
        n2 = noise_bank[RNG.integers(0, len(noise_bank))]
        out.append(mix_with_noise(y, n1, snr_db=20))
        out.append(mix_with_noise(y, n2, snr_db=12))
    try:
        out.append(pad_or_trim(librosa.effects.time_stretch(y, rate=0.95).astype(np.float32), len(y)))
    except Exception:
        pass
    return out


def augment_noise_window(y: np.ndarray) -> list[np.ndarray]:
    out = [y]
    out.append(np.clip(y * 0.90, -1, 1).astype(np.float32))
    out.append(np.clip(y * 1.10, -1, 1).astype(np.float32))
    out.append(np.roll(y, int(0.1 * len(y))).astype(np.float32))
    return out


# =========================================================
# Dataset building on fixed windows
# =========================================================

def build_training_dataset(species_zip: str,
                           noise_zip: str,
                           window_sec: float = 0.5,
                           hop_sec_train: float = 0.5,
                           target_sr: int = 16000,
                           use_augmentation: bool = True):
    species_df = load_species_dataset(species_zip)
    noise_df = load_noise_dataset(noise_zip)
    meta = pd.concat([species_df, noise_df], ignore_index=True)

    zf_species = zipfile.ZipFile(species_zip) if is_zip_path(species_zip) else None
    zf_noise = zipfile.ZipFile(noise_zip) if is_zip_path(noise_zip) else None

    # build noise bank from noise-train windows
    noise_bank = []
    noise_train = noise_df[noise_df['split'] == 'train']
    for _, row in noise_train.iterrows():
        y, sr = read_audio_row(row, zf_species, zf_noise)
        windows, _ = window_audio(y, sr, window_sec, hop_sec_train)
        noise_bank.extend(windows)

    X, rows, rejected = [], [], []

    for _, row in meta.iterrows():
        try:
            y, sr = read_audio_row(row, zf_species, zf_noise)
            windows, sr = window_audio(y, sr, window_sec, hop_sec_train)
            if not windows:
                rejected.append({**row.to_dict(), 'reason': 'no_windows'})
                continue

            for w_idx, w in enumerate(windows):
                variants = [w]
                if row['split'] == 'train' and use_augmentation:
                    if row['label'] == 'Noise':
                        variants = augment_noise_window(w)
                    else:
                        variants = augment_species_window(w, noise_bank)

                for a_idx, w_aug in enumerate(variants):
                    feat = extract_features(w_aug, target_sr, target_sr)
                    X.append(feat)
                    rows.append({
                        **row.to_dict(),
                        'window_idx': w_idx,
                        'is_augmented': int(a_idx > 0),
                        'aug_idx': a_idx,
                        'window_sec': window_sec,
                        'base_group_id': row['group_id'],
                        'group_id': row['group_id'],
                    })
        except Exception as e:
            rejected.append({**row.to_dict(), 'reason': str(e)})

    return (
        np.vstack(X).astype(np.float32),
        pd.DataFrame(rows),
        pd.DataFrame(rejected),
    )


# =========================================================
# Models / eval
# =========================================================

def get_models():
    return {
        'ExtraTrees': ExtraTreesClassifier(
            n_estimators=1200,
            max_depth=None,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=800,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        ),
        'HistGB': HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.05,
            max_depth=10,
            random_state=42,
        ),
        'SVC_RBF': Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(C=10, kernel='rbf', probability=True, class_weight='balanced')),
        ]),
    }


def grouped_cv_scores(X_train, y_train, groups):
    models = get_models()
    rows = []
    uniq_groups = len(set(groups))
    n_splits = max(3, min(5, uniq_groups))
    gkf = GroupKFold(n_splits=n_splits)
    for name, model in models.items():
        try:
            scores = cross_validate(
                model, X_train, y_train,
                cv=gkf.split(X_train, y_train, groups),
                scoring=['accuracy', 'f1_macro'],
                n_jobs=-1,
                error_score='raise'
            )
            rows.append({
                'model': name,
                'cv_accuracy_mean': float(np.mean(scores['test_accuracy'])),
                'cv_f1_macro_mean': float(np.mean(scores['test_f1_macro'])),
            })
        except Exception as e:
            rows.append({'model': name, 'cv_accuracy_mean': np.nan, 'cv_f1_macro_mean': np.nan, 'error': str(e)})
    df = pd.DataFrame(rows).sort_values('cv_f1_macro_mean', ascending=False, na_position='last').reset_index(drop=True)
    return df, models


def train_best_model(X_train, y_train, X_test, y_test, groups_train):
    results_df, models = grouped_cv_scores(X_train, y_train, groups_train)
    best_name = results_df.iloc[0]['model']
    best_model = clone(models[best_name])
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    return best_model, best_name, results_df, y_pred


# =========================================================
# Long-sequence prediction
# =========================================================

def moving_average(arr: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return arr
    pad = k // 2
    out = np.empty_like(arr)
    for c in range(arr.shape[1]):
        x = np.pad(arr[:, c], (pad, pad), mode='edge')
        out[:, c] = np.convolve(x, np.ones(k) / k, mode='valid')[:arr.shape[0]]
    return out


def detect_events_for_audio(audio: np.ndarray,
                            sr: int,
                            model_bundle: dict,
                            window_sec: float,
                            hop_sec: float,
                            prob_threshold: float,
                            margin_vs_noise: float,
                            min_event_sec: float,
                            max_gap_sec: float,
                            smooth_k: int = 5,
                            energy_quantile: float = 0.10,
                            return_debug: bool = True):
    model = model_bundle['model']
    labels = list(model_bundle['labels'])
    noise_label = 'Noise' if 'Noise' in labels else None
    noise_idx = labels.index(noise_label) if noise_label else None

    y, sr = preprocess_audio(audio, sr, model_bundle.get('target_sr', 16000))
    win = int(round(window_sec * sr))
    hop = int(round(hop_sec * sr))
    if len(y) < win:
        y = pad_or_trim(y, win)

    starts = list(range(0, max(1, len(y) - win + 1), hop))
    if starts and starts[-1] != len(y) - win and len(y) > win:
        starts.append(len(y) - win)
    elif not starts:
        starts = [0]

    feats = []
    rms_vals = []
    windows = []
    for s in starts:
        w = y[s:s+win]
        if len(w) < win:
            w = pad_or_trim(w, win)
        feats.append(extract_features(w, sr, sr))
        rms_vals.append(float(np.sqrt(np.mean(w ** 2) + 1e-9)))
        windows.append((s / sr, (s + win) / sr))

    X = np.vstack(feats).astype(np.float32)
    probs = model.predict_proba(X)
    probs = moving_average(probs, smooth_k)

    rms_vals = np.asarray(rms_vals)
    energy_thr = float(np.quantile(rms_vals, energy_quantile))

    raw = []
    for i, ((start_t, end_t), p, rms) in enumerate(zip(windows, probs, rms_vals)):
        best_idx = int(np.argmax(p))
        best_label = labels[best_idx]
        best_prob = float(p[best_idx])
        noise_prob = float(p[noise_idx]) if noise_idx is not None else 0.0
        if rms < energy_thr:
            continue
        if best_label == noise_label:
            continue
        if best_prob < prob_threshold:
            continue
        if (best_prob - noise_prob) < margin_vs_noise:
            continue
        raw.append({
            'start': start_t,
            'end': end_t,
            'label': best_label,
            'confidence': best_prob,
            'noise_prob': noise_prob,
            'rms': rms,
            'window_index': i,
        })

    merged = []
    for det in raw:
        if not merged:
            merged.append({**det, 'count': 1})
            continue
        last = merged[-1]
        if det['label'] == last['label'] and det['start'] - last['end'] <= max_gap_sec:
            last['end'] = det['end']
            last['confidence'] = max(last['confidence'], det['confidence'])
            last['noise_prob'] = min(last['noise_prob'], det['noise_prob'])
            last['count'] += 1
        else:
            merged.append({**det, 'count': 1})

    final_events = []
    for e in merged:
        dur = e['end'] - e['start']
        if dur >= min_event_sec:
            final_events.append({
                'animal': e['label'],
                'start_sec': round(e['start'], 3),
                'end_sec': round(e['end'], 3),
                'duration_sec': round(dur, 3),
                'confidence': round(float(e['confidence']), 4),
                'count': int(e['count']),
            })
    if return_debug:
        debug_df = pd.DataFrame([
            {
                'start_sec': round(w[0], 3),
                'end_sec': round(w[1], 3),
                'rms': round(float(rms_vals[i]), 6),
                **{f'proba_{lab}': round(float(probs[i, j]), 6) for j, lab in enumerate(labels)},
            }
            for i, w in enumerate(windows)
        ])
    else:
        debug_df = pd.DataFrame()
    return final_events, debug_df


def export_events_csv(events: list[dict], out_csv: Path):
    df = pd.DataFrame(events)
    if df.empty:
        df = pd.DataFrame(columns=['animal', 'start_sec', 'end_sec', 'duration_sec'])
    else:
        df = df[['animal', 'start_sec', 'end_sec', 'duration_sec']]
    df.to_csv(out_csv, index=False)


# =========================================================
# CLI
# =========================================================

def cmd_train(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('1) Construction du dataset fenêtré 6 classes...')
    X, meta, rejected = build_training_dataset(
        species_zip=args.species_zip,
        noise_zip=args.noise_zip,
        window_sec=args.window_sec,
        hop_sec_train=args.hop_sec_train,
        target_sr=args.target_sr,
        use_augmentation=args.use_augmentation,
    )

    train_mask = meta['split'].eq('train').to_numpy()
    X_train = X[train_mask]
    y_train = meta.loc[train_mask, 'label'].to_numpy()
    X_test = X[~train_mask]
    y_test = meta.loc[~train_mask, 'label'].to_numpy()
    groups_train = meta.loc[train_mask, 'base_group_id'].to_numpy()

    print('Distribution train:', Counter(y_train))
    print('Distribution test :', Counter(y_test))

    print('\n2) Benchmark groupé + entraînement...')
    model, model_name, results_df, y_pred = train_best_model(X_train, y_train, X_test, y_test, groups_train)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f'✅ Modèle retenu : {model_name}')
    print(f'🎯 Test accuracy  : {acc:.4f}')
    print(f'🎯 Test macro-F1  : {f1:.4f}')

    labels = sorted(meta['label'].unique())
    report = classification_report(y_test, y_pred, labels=labels, target_names=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    bundle = {
        'model': model,
        'model_name': model_name,
        'labels': labels,
        'target_sr': args.target_sr,
        'window_sec': args.window_sec,
        'hop_sec_detect': args.hop_sec_detect,
        'prob_threshold': args.prob_threshold,
        'margin_vs_noise': args.margin_vs_noise,
        'min_event_sec': args.min_event_sec,
        'max_gap_sec': args.max_gap_sec,
        'feature_names': [f'f{i}' for i in range(X.shape[1])],
    }
    joblib.dump(bundle, output_dir / 'best_marine_model.joblib')
    results_df.to_csv(output_dir / 'model_comparison_grouped_cv.csv', index=False)
    pd.DataFrame(report).to_json(output_dir / 'classification_report.json', orient='index', force_ascii=False, indent=2)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(output_dir / 'confusion_matrix.csv')
    meta.to_csv(output_dir / 'training_windows_meta.csv', index=False)
    rejected.to_csv(output_dir / 'rejected_rows.csv', index=False)

    summary = {
        'model_name': model_name,
        'test_accuracy': acc,
        'test_f1_macro': f1,
        'window_sec': args.window_sec,
        'hop_sec_train': args.hop_sec_train,
        'hop_sec_detect': args.hop_sec_detect,
        'prob_threshold': args.prob_threshold,
        'margin_vs_noise': args.margin_vs_noise,
        'min_event_sec': args.min_event_sec,
        'max_gap_sec': args.max_gap_sec,
    }
    with open(output_dir / 'training_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f'\n✅ Fichiers sauvegardés dans : {output_dir}')
    print('   - best_marine_model.joblib')
    print('   - model_comparison_grouped_cv.csv')
    print('   - classification_report.json')
    print('   - confusion_matrix.csv')
    print('   - training_windows_meta.csv')
    print('   - rejected_rows.csv')


def iter_wavs_in_zip(zip_path: str):
    zf = zipfile.ZipFile(zip_path)
    for name in sorted(zf.namelist()):
        if name.lower().endswith('.wav'):
            yield zf, name


def cmd_predict(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(args.model_path)
    window_sec = args.window_sec if args.window_sec is not None else bundle.get('window_sec', 0.5)
    hop_sec = args.hop_sec if args.hop_sec is not None else bundle.get('hop_sec_detect', 0.25)
    prob_threshold = args.prob_threshold if args.prob_threshold is not None else bundle.get('prob_threshold', 0.60)
    margin_vs_noise = args.margin_vs_noise if args.margin_vs_noise is not None else bundle.get('margin_vs_noise', 0.10)
    min_event_sec = args.min_event_sec if args.min_event_sec is not None else bundle.get('min_event_sec', 0.50)
    max_gap_sec = args.max_gap_sec if args.max_gap_sec is not None else bundle.get('max_gap_sec', 0.25)

    produced = []

    if args.audio_path:
        y, sr = read_wav_file(args.audio_path)
        events, debug_df = detect_events_for_audio(
            y, sr, bundle,
            window_sec=window_sec, hop_sec=hop_sec,
            prob_threshold=prob_threshold,
            margin_vs_noise=margin_vs_noise,
            min_event_sec=min_event_sec,
            max_gap_sec=max_gap_sec,
            smooth_k=args.smooth_k,
            return_debug=bool(args.save_debug),
        )
        stem = Path(args.audio_path).stem
        export_events_csv(events, output_dir / f'{stem}.csv')
        if args.save_debug:
            debug_df.to_csv(output_dir / f'{stem}_debug_windows.csv', index=False)
        produced.append(output_dir / f'{stem}.csv')
    elif args.input_zip:
        zf = zipfile.ZipFile(args.input_zip)
        for name in sorted(zf.namelist()):
            if not name.lower().endswith('.wav'):
                continue
            y, sr = read_wav_from_zip(zf, name)
            events, debug_df = detect_events_for_audio(
                y, sr, bundle,
                window_sec=window_sec, hop_sec=hop_sec,
                prob_threshold=prob_threshold,
                margin_vs_noise=margin_vs_noise,
                min_event_sec=min_event_sec,
                max_gap_sec=max_gap_sec,
                smooth_k=args.smooth_k,
                return_debug=bool(args.save_debug),
            )
            stem = Path(name).stem
            export_events_csv(events, output_dir / f'{stem}.csv')
            if args.save_debug:
                debug_df.to_csv(output_dir / f'{stem}_debug_windows.csv', index=False)
            produced.append(output_dir / f'{stem}.csv')
    else:
        raise ValueError('Donne --audio_path ou --input_zip')

    zip_out = output_dir / 'predictions.zip'
    with zipfile.ZipFile(zip_out, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for p in produced:
            zf.write(p, arcname=p.name)

    print(f' {len(produced)} fichier(s) CSV généré(s) dans {output_dir}')
    print(f' Zip final : {zip_out}')


def build_parser():
    parser = argparse.ArgumentParser(description='Best practical solution for marine audio classification + long-sequence detection')
    sub = parser.add_subparsers(dest='command', required=True)

    p_train = sub.add_parser('train')
    p_train.add_argument('--species_zip', required=True, help='Chemin zip ou dossier Partie #1/Base de données')
    p_train.add_argument('--noise_zip', required=True, help='Chemin zip ou dossier Partie #2/Complément de base de données')
    p_train.add_argument('--output_dir', default='best_output')
    p_train.add_argument('--target_sr', type=int, default=16000)
    p_train.add_argument('--window_sec', type=float, default=0.5)
    p_train.add_argument('--hop_sec_train', type=float, default=0.5)
    p_train.add_argument('--hop_sec_detect', type=float, default=0.25)
    p_train.add_argument('--prob_threshold', type=float, default=0.60)
    p_train.add_argument('--margin_vs_noise', type=float, default=0.10)
    p_train.add_argument('--min_event_sec', type=float, default=0.50)
    p_train.add_argument('--max_gap_sec', type=float, default=0.25)
    p_train.add_argument('--use_augmentation', action='store_true')
    p_train.set_defaults(func=cmd_train)

    p_pred = sub.add_parser('predict')
    p_pred.add_argument('--model_path', required=True)
    p_pred.add_argument('--audio_path')
    p_pred.add_argument('--input_zip')
    p_pred.add_argument('--output_dir', default='best_predictions')
    p_pred.add_argument('--window_sec', type=float)
    p_pred.add_argument('--hop_sec', type=float)
    p_pred.add_argument('--prob_threshold', type=float)
    p_pred.add_argument('--margin_vs_noise', type=float)
    p_pred.add_argument('--min_event_sec', type=float)
    p_pred.add_argument('--max_gap_sec', type=float)
    p_pred.add_argument('--smooth_k', type=int, default=5)
    p_pred.add_argument('--save_debug', action='store_true')
    p_pred.set_defaults(func=cmd_predict)
    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()
    args.func(args)    