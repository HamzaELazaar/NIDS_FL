import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# =============================
# CONFIG
# =============================
CSV_PATH = "data/cicids2017_cleaned.csv"
N_SPLITS = 5

FEATS_7 = [
    'Destination Port','FIN Flag Count','PSH Flag Count','ACK Flag Count',
    'Init_Win_bytes_forward','Subflow Fwd Bytes','Average Packet Size'
]

FEATS_11 = FEATS_7 + [
    'Total Fwd Packets','Flow Bytes/s','Flow Packets/s','Fwd Packets/s'
]

LABEL = 'Attack Type'

# =============================
# LOAD DATA
# =============================
print("📂 Loading dataset...")
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

# Nettoyage sécurité
df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(df[LABEL])

# =============================
# BENCH FUNCTION
# =============================
def run_cv(features):
    X = df[features].values
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    acc_scores = []
    f1_macro_scores = []
    f1_per_class = {cls: [] for cls in le.classes_}
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"   Fold {fold}/{N_SPLITS}...")
        
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        
        model = RandomForestClassifier(
            n_estimators=100,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced"   # 🔥 important
        )
        
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        
        # Metrics
        acc_scores.append(accuracy_score(y_te, y_pred))
        f1_macro_scores.append(f1_score(y_te, y_pred, average='macro'))
        
        f1_cls = f1_score(y_te, y_pred, average=None)
        for i, cls in enumerate(le.classes_):
            f1_per_class[cls].append(f1_cls[i])
    
    return acc_scores, f1_macro_scores, f1_per_class


# =============================
# RUN
# =============================
print("\n🔬 Benchmark 7 features")
acc7, f17, f1c7 = run_cv(FEATS_7)

print("\n🔬 Benchmark 11 features")
acc11, f111, f1c11 = run_cv(FEATS_11)


# =============================
# RESULTS
# =============================
def summarize(name, acc, f1):
    print(f"\n{name}")
    print(f"Accuracy   : {np.mean(acc)*100:.2f}% ± {np.std(acc)*100:.2f}")
    print(f"F1 Macro   : {np.mean(f1):.4f} ± {np.std(f1):.4f}")

summarize("7 FEATURES", acc7, f17)
summarize("11 FEATURES", acc11, f111)

print("\n📊 F1 per class (mean):")
print(f"{'Class':<20} {'F1-7':>8} {'F1-11':>8} {'Δ':>8}")
print("-"*45)

for cls in le.classes_:
    m7 = np.mean(f1c7[cls])
    m11 = np.mean(f1c11[cls])
    d = m11 - m7
    sign = "+" if d >= 0 else ""
    
    print(f"{cls:<20} {m7:>8.3f} {m11:>8.3f} {sign}{d:>8.3f}")