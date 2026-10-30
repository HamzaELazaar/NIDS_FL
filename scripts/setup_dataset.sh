#!/usr/bin/env bash
# =============================================================================
# setup_dataset.sh — Téléchargement automatique du dataset CICIDS2017
# =============================================================================
set -euo pipefail

# ─── Couleurs & emojis ───────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${CYAN}ℹ️  $*${NC}"; }
success() { echo -e "${GREEN}✅ $*${NC}"; }
warning() { echo -e "${YELLOW}⚠️  $*${NC}"; }
error()   { echo -e "${RED}❌ $*${NC}"; }
step()    { echo -e "\n${BOLD}${BLUE}═══ $* ═══${NC}"; }

# ─── Chemins ─────────────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${PROJECT_DIR}/data"
VENV_PYTHON="${PROJECT_DIR}/venv/bin/python"
VENV_PIP="${PROJECT_DIR}/venv/bin/pip"
KAGGLE_BIN="${PROJECT_DIR}/venv/bin/kaggle"
KAGGLE_DIR="${HOME}/.kaggle"
KAGGLE_JSON="${KAGGLE_DIR}/kaggle.json"
DATASET="ericanacletoribeiro/cicids2017-cleaned-and-preprocessed"
INFO_FILE="${DATA_DIR}/dataset_info.txt"

echo -e "${BOLD}${BLUE}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║      NIDS-FL — Setup Dataset CICIDS2017              ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"

mkdir -p "${DATA_DIR}" "${KAGGLE_DIR}"

# =============================================================================
# ÉTAPE 1 — Installer le package kaggle dans le venv
# =============================================================================
step "1/6 — Installation du package kaggle"

if [[ ! -f "${VENV_PYTHON}" ]]; then
    error "venv introuvable dans ${PROJECT_DIR}/venv"
    echo "  Crée le venv d'abord : python3.12 -m venv venv && source venv/bin/activate"
    exit 1
fi

if "${VENV_PYTHON}" -c "import kaggle" 2>/dev/null; then
    success "kaggle déjà installé ($(${KAGGLE_BIN} --version 2>&1 | head -1))"
else
    info "Installation de kaggle dans le venv..."
    "${VENV_PIP}" install kaggle -q
    success "kaggle installé"
fi

# =============================================================================
# ÉTAPE 2 — Détecter / créer kaggle.json
# =============================================================================
step "2/6 — Configuration des credentials Kaggle"

_setup_credentials() {
    # 2a. Déjà en place ?
    if [[ -f "${KAGGLE_JSON}" ]]; then
        success "kaggle.json trouvé dans ~/.kaggle/"
        chmod 600 "${KAGGLE_JSON}"
        return 0
    fi

    # 2b. Variables d'environnement ?
    if [[ -n "${KAGGLE_USERNAME:-}" && -n "${KAGGLE_KEY:-}" ]]; then
        info "Création de kaggle.json depuis les variables d'environnement..."
        echo "{\"username\":\"${KAGGLE_USERNAME}\",\"key\":\"${KAGGLE_KEY}\"}" > "${KAGGLE_JSON}"
        chmod 600 "${KAGGLE_JSON}"
        success "kaggle.json créé depuis KAGGLE_USERNAME / KAGGLE_KEY"
        return 0
    fi

    # 2c. Chercher dans les Downloads Windows
    info "Recherche de kaggle.json dans les dossiers Windows..."
    local found=""
    for pattern in \
        "/mnt/c/Users/*/Downloads/kaggle.json" \
        "/mnt/c/Users/*/kaggle.json" \
        "/mnt/c/Users/*/.kaggle/kaggle.json"; do
        for f in ${pattern}; do
            if [[ -f "$f" ]]; then
                found="$f"
                break 2
            fi
        done
    done

    if [[ -n "$found" ]]; then
        info "Trouvé : $found"
        cp "$found" "${KAGGLE_JSON}"
        chmod 600 "${KAGGLE_JSON}"
        success "kaggle.json copié dans ~/.kaggle/"
        return 0
    fi

    # 2d. Saisie manuelle
    warning "kaggle.json introuvable — saisie manuelle requise"
    echo ""
    echo "  👉 Va sur https://www.kaggle.com/settings → API → Create New Token"
    echo "     puis copie ton username et ta key ci-dessous."
    echo ""
    read -rp "  Kaggle username : " kg_user
    read -rp "  Kaggle API key  : " kg_key
    echo "{\"username\":\"${kg_user}\",\"key\":\"${kg_key}\"}" > "${KAGGLE_JSON}"
    chmod 600 "${KAGGLE_JSON}"
    success "kaggle.json créé"
}

_setup_credentials

info "Contenu kaggle.json : $(python3 -c "import json; d=json.load(open('${KAGGLE_JSON}')); print('username=' + d['username'])")"

# =============================================================================
# ÉTAPE 3 — Tester la connexion Kaggle
# =============================================================================
step "3/6 — Test de connexion Kaggle"

info "Recherche 'cicids2017' sur Kaggle..."
if "${KAGGLE_BIN}" datasets list --search cicids2017 2>&1 | head -5; then
    success "Connexion Kaggle OK"
else
    error "Connexion Kaggle échouée"
    echo ""
    echo "  Solutions possibles :"
    echo "  1. Vérifie ton username et ta key dans ~/.kaggle/kaggle.json"
    echo "  2. Accepte les conditions du dataset sur kaggle.com"
    echo "  3. Vérifie ta connexion internet"
    exit 1
fi

# =============================================================================
# ÉTAPE 4 — Téléchargement du dataset
# =============================================================================
step "4/6 — Téléchargement CICIDS2017"

CSV_COUNT=$(find "${DATA_DIR}" -name "*.csv" 2>/dev/null | wc -l)
if [[ "${CSV_COUNT}" -gt 0 ]]; then
    warning "${CSV_COUNT} CSV déjà présent(s) dans data/ — téléchargement ignoré"
    info "  Supprime data/*.csv pour re-télécharger"
else
    info "Téléchargement dans ${DATA_DIR}..."
    info "Dataset : ${DATASET}"
    echo ""
    cd "${DATA_DIR}"
    "${KAGGLE_BIN}" datasets download "${DATASET}" --unzip
    echo ""
    success "Téléchargement terminé"
fi

# =============================================================================
# ÉTAPE 5 — Inspection du dataset
# =============================================================================
step "5/6 — Inspection du dataset"

mapfile -t CSV_FILES < <(find "${DATA_DIR}" -name "*.csv" | sort)

if [[ ${#CSV_FILES[@]} -eq 0 ]]; then
    error "Aucun fichier CSV trouvé dans ${DATA_DIR}"
    exit 1
fi

echo ""
echo -e "${BOLD}📁 Fichiers CSV téléchargés :${NC}"
printf "%-60s %10s %12s\n" "Fichier" "Taille" "Lignes"
printf "%-60s %10s %12s\n" "──────────────────────────────────────────────────────────" "──────────" "────────────"
for f in "${CSV_FILES[@]}"; do
    size=$(du -sh "$f" | cut -f1)
    lines=$(wc -l < "$f")
    printf "%-60s %10s %12s\n" "$(basename "$f")" "$size" "$lines"
done

FIRST_CSV="${CSV_FILES[0]}"
echo ""
echo -e "${BOLD}📊 Colonnes du fichier : $(basename "${FIRST_CSV}")${NC}"
"${VENV_PYTHON}" -c "
import pandas as pd
df = pd.read_csv('${FIRST_CSV}', nrows=0)
cols = list(df.columns)
print(f'  Total : {len(cols)} colonnes\n')
for i, c in enumerate(cols, 1):
    print(f'  {i:>3}. {c}')
"

echo ""
echo -e "${BOLD}🏷️  Distribution des labels (colonne cible) :${NC}"
"${VENV_PYTHON}" -c "
import pandas as pd, os, sys

# Candidats dans l'ordre de priorité
LABEL_CANDIDATES = ['Label', ' Label', 'Attack Type', 'attack_type', 'label', 'class', 'Class']

files = sys.argv[1:]
frames = []
label_col = None

for f in files:
    df_head = pd.read_csv(f, nrows=0)
    df_head.columns = df_head.columns.str.strip()
    if label_col is None:
        for c in LABEL_CANDIDATES:
            if c.strip() in df_head.columns.str.strip().tolist():
                label_col = c.strip()
                break
    if label_col is None:
        print(f'  ❌ Colonne label introuvable. Colonnes disponibles: {list(df_head.columns)}')
        sys.exit(1)
    chunk = pd.read_csv(f, usecols=lambda c: c.strip() == label_col)
    chunk.columns = chunk.columns.str.strip()
    frames.append(chunk)

print(f'  Colonne détectée : \"{label_col}\"\n')
df = pd.concat(frames, ignore_index=True)
dist = df[label_col].value_counts()
total = len(df)
print(f'  Total : {total:,} paquets\n')
print(f'  {\"Classe\":<40} {\"Count\":>10}  {\"Pct\":>7}')
print(f'  {\"─\"*40} {\"─\"*10}  {\"─\"*7}')
for label, count in dist.items():
    pct = count / total * 100
    print(f'  {str(label):<40} {count:>10,}  {pct:>6.2f}%')
" "${CSV_FILES[@]}"

# =============================================================================
# ÉTAPE 6 — Créer data/dataset_info.txt
# =============================================================================
step "6/6 — Génération de dataset_info.txt"

{
    echo "═══════════════════════════════════════════════════════"
    echo "  CICIDS2017 — Résumé du dataset"
    echo "  Généré le : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "═══════════════════════════════════════════════════════"
    echo ""
    echo "Source Kaggle : ${DATASET}"
    echo ""
    echo "─── Fichiers CSV ───────────────────────────────────────"
    for f in "${CSV_FILES[@]}"; do
        size=$(du -sh "$f" | cut -f1)
        lines=$(wc -l < "$f")
        echo "  $(basename "$f")  |  $size  |  $lines lignes"
    done
    echo ""
    echo "─── Colonnes ───────────────────────────────────────────"
    "${VENV_PYTHON}" -c "
import pandas as pd
df = pd.read_csv('${FIRST_CSV}', nrows=0)
for i, c in enumerate(df.columns, 1):
    print(f'  {i:>3}. {c}')
"
    echo ""
    echo "─── Distribution des labels ────────────────────────────"
    "${VENV_PYTHON}" -c "
import pandas as pd, sys
LABEL_CANDIDATES = ['Label', ' Label', 'Attack Type', 'attack_type', 'label', 'class', 'Class']
files = [f for f in '''$(printf '%s\n' "${CSV_FILES[@]}")'''.strip().split('\n') if f]
df_head = pd.read_csv(files[0], nrows=0)
df_head.columns = df_head.columns.str.strip()
label_col = next((c.strip() for c in LABEL_CANDIDATES if c.strip() in df_head.columns.str.strip().tolist()), None)
if not label_col:
    print('  Colonne label introuvable')
    sys.exit(0)
frames = []
for f in files:
    chunk = pd.read_csv(f, usecols=lambda c: c.strip() == label_col)
    chunk.columns = chunk.columns.str.strip()
    frames.append(chunk)
df = pd.concat(frames, ignore_index=True)
dist = df[label_col].value_counts()
total = len(df)
print(f'  Total : {total:,} paquets')
for label, count in dist.items():
    print(f'  {str(label):<40} {count:>10,}  ({count/total*100:.2f}%)')
"
    echo ""
    echo "─── Features retenues pour NIDS-FL (18) ───────────────"
    echo "  timestamp, protocol, src_ip, dst_ip, src_port, dst_port"
    echo "  packet_length, ttl, payload_length"
    echo "  syn_flag, fin_flag, rst_flag, psh_flag, ack_flag, urg_flag"
    echo "  window_size, icmp_type, icmp_code"
    echo ""
    echo "═══════════════════════════════════════════════════════"
} > "${INFO_FILE}"

success "dataset_info.txt créé : ${INFO_FILE}"

# ─── Résumé final ────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║   🎉  Setup terminé avec succès !                    ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo -e "  📂 Données : ${DATA_DIR}"
echo -e "  📄 Résumé  : ${INFO_FILE}"
echo -e "  📦 CSV     : ${#CSV_FILES[@]} fichier(s)"
echo ""
