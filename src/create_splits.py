"""
Création des splits Train/Validation/Test pour GTZAN
Split stratifié : 70% train / 15% val / 15% test
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

# Configuration
np.random.seed(42)  # Reproductibilité

DATA_PATH = Path("data/raw/Data/genres_original")
SPLITS_PATH = Path("data/splits")
SPLITS_PATH.mkdir(parents=True, exist_ok=True)

GENRES = ["blues", "classical", "country", "disco", "hiphop", 
          "jazz", "metal", "pop", "reggae", "rock"]

print("=" * 70)
print("🔀 CRÉATION DES SPLITS TRAIN/VAL/TEST")
print("=" * 70)

# ============================================================================
# 1. CHARGER LES MÉTADONNÉES (exclure fichiers corrompus)
# ============================================================================
print("\n📂 1. Chargement des métadonnées...")

# Charger les fichiers valides
df = pd.read_csv('data/processed/file_metadata.csv')
print(f"   → {len(df)} fichiers valides chargés")

# Charger les fichiers corrompus (si existent)
corrupted_path = Path('data/processed/corrupted_files.csv')
if corrupted_path.exists():
    df_corrupted = pd.read_csv(corrupted_path)
    print(f"   ⚠️  {len(df_corrupted)} fichiers corrompus exclus:")
    for _, row in df_corrupted.iterrows():
        print(f"      - {row['genre']}/{row['filename']}")
else:
    print("   ✅ Aucun fichier corrompu détecté")

# Statistiques par genre
print("\n📊 Distribution par genre:")
genre_counts = df['genre'].value_counts().sort_index()
for genre, count in genre_counts.items():
    print(f"   {genre:12s}: {count:3d} fichiers")

# ============================================================================
# 2. CRÉER LES SPLITS STRATIFIÉS
# ============================================================================
print("\n" + "=" * 70)
print("✂️  2. Création des splits stratifiés (70/15/15)")
print("=" * 70)

# Ajouter le chemin complet
df['filepath'] = df.apply(lambda row: str(DATA_PATH / row['genre'] / row['filename']).replace('\\', '/'), axis=1)

# Split 1 : Train (70%) vs Temp (30%)
train_files, temp_files, train_labels, temp_labels = train_test_split(
    df['filepath'].values,
    df['genre'].values,
    test_size=0.30,
    stratify=df['genre'].values,
    random_state=42
)

# Split 2 : Temp → Val (50%) + Test (50%) = 15% + 15% du total
val_files, test_files, val_labels, test_labels = train_test_split(
    temp_files,
    temp_labels,
    test_size=0.50,
    stratify=temp_labels,
    random_state=42
)

print(f"\n✅ Splits créés:")
print(f"   📚 Train: {len(train_files):3d} fichiers ({len(train_files)/len(df)*100:.1f}%)")
print(f"   📖 Val:   {len(val_files):3d} fichiers ({len(val_files)/len(df)*100:.1f}%)")
print(f"   📝 Test:  {len(test_files):3d} fichiers ({len(test_files)/len(df)*100:.1f}%)")
print(f"   🎯 Total: {len(train_files)+len(val_files)+len(test_files):3d} fichiers")

# Vérifier la stratification
print("\n🎭 Vérification de la stratification par genre:")
print(f"\n{'Genre':<12} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8}")
print("-" * 50)

for genre in GENRES:
    train_count = sum(train_labels == genre)
    val_count = sum(val_labels == genre)
    test_count = sum(test_labels == genre)
    total = train_count + val_count + test_count
    
    print(f"{genre:<12} {train_count:<8} {val_count:<8} {test_count:<8} {total:<8}")

# ============================================================================
# 3. SAUVEGARDER LES SPLITS
# ============================================================================
print("\n" + "=" * 70)
print("💾 3. Sauvegarde des splits")
print("=" * 70)

# Créer DataFrames pour chaque split
train_df = pd.DataFrame({
    'filepath': train_files,
    'genre': train_labels,
    'filename': [Path(f).name for f in train_files]
})

val_df = pd.DataFrame({
    'filepath': val_files,
    'genre': val_labels,
    'filename': [Path(f).name for f in val_files]
})

test_df = pd.DataFrame({
    'filepath': test_files,
    'genre': test_labels,
    'filename': [Path(f).name for f in test_files]
})

# Sauvegarder en CSV
train_df.to_csv(SPLITS_PATH / 'train.csv', index=False)
val_df.to_csv(SPLITS_PATH / 'val.csv', index=False)
test_df.to_csv(SPLITS_PATH / 'test.csv', index=False)

print("✅ Fichiers CSV sauvegardés:")
print(f"   - {SPLITS_PATH / 'train.csv'}")
print(f"   - {SPLITS_PATH / 'val.csv'}")
print(f"   - {SPLITS_PATH / 'test.csv'}")

# Sauvegarder aussi en JSON (pour faciliter le chargement)
splits_dict = {
    'train': train_files.tolist(),
    'val': val_files.tolist(),
    'test': test_files.tolist(),
    'label_mapping': {i: genre for i, genre in enumerate(GENRES)},
    'num_classes': len(GENRES)
}

with open(SPLITS_PATH / 'splits.json', 'w') as f:
    json.dump(splits_dict, f, indent=2)

print(f"✅ Configuration JSON sauvegardée: {SPLITS_PATH / 'splits.json'}")

# ============================================================================
# 4. CRÉER UN RÉSUMÉ DÉTAILLÉ
# ============================================================================
print("\n" + "=" * 70)
print("📋 4. Génération du résumé")
print("=" * 70)

summary = {
    'dataset': 'GTZAN Music Genre Classification',
    'total_files': len(df),
    'corrupted_files': len(pd.read_csv(corrupted_path)) if corrupted_path.exists() else 0,
    'genres': GENRES,
    'num_classes': len(GENRES),
    'splits': {
        'train': {
            'count': len(train_files),
            'percentage': round(len(train_files)/len(df)*100, 2),
            'distribution': {genre: int(sum(train_labels == genre)) for genre in GENRES}
        },
        'val': {
            'count': len(val_files),
            'percentage': round(len(val_files)/len(df)*100, 2),
            'distribution': {genre: int(sum(val_labels == genre)) for genre in GENRES}
        },
        'test': {
            'count': len(test_files),
            'percentage': round(len(test_files)/len(df)*100, 2),
            'distribution': {genre: int(sum(test_labels == genre)) for genre in GENRES}
        }
    },
    'random_seed': 42,
    'stratified': True
}

with open(SPLITS_PATH / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"✅ Résumé sauvegardé: {SPLITS_PATH / 'summary.json'}")

# ============================================================================
# 5. VISUALISATION DES SPLITS
# ============================================================================
print("\n" + "=" * 70)
print("📊 5. Visualisation des splits")
print("=" * 70)

import matplotlib.pyplot as plt
import seaborn as sns

# Préparer les données pour le graphique
split_data = []
for genre in GENRES:
    split_data.append({
        'Genre': genre,
        'Split': 'Train',
        'Count': sum(train_labels == genre)
    })
    split_data.append({
        'Genre': genre,
        'Split': 'Val',
        'Count': sum(val_labels == genre)
    })
    split_data.append({
        'Genre': genre,
        'Split': 'Test',
        'Count': sum(test_labels == genre)
    })

plot_df = pd.DataFrame(split_data)

# Créer la figure
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Graphique 1 : Barplot groupé
sns.barplot(data=plot_df, x='Genre', y='Count', hue='Split', ax=axes[0], palette='Set2')
axes[0].set_title('Distribution des fichiers par genre et split', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Genre', fontsize=12)
axes[0].set_ylabel('Nombre de fichiers', fontsize=12)
axes[0].legend(title='Split')
axes[0].tick_params(axis='x', rotation=45)

# Graphique 2 : Pie chart des proportions globales
split_sizes = [len(train_files), len(val_files), len(test_files)]
split_labels = [f'Train\n({len(train_files)} fichiers)', 
                f'Val\n({len(val_files)} fichiers)', 
                f'Test\n({len(test_files)} fichiers)']
colors = sns.color_palette('Set2', 3)

axes[1].pie(split_sizes, labels=split_labels, autopct='%1.1f%%', 
            startangle=90, colors=colors, textprops={'fontsize': 11})
axes[1].set_title('Proportion globale des splits', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('results/figures/03_train_val_test_splits.png', dpi=300, bbox_inches='tight')
print("✅ Visualisation sauvegardée: results/figures/03_train_val_test_splits.png")
plt.show()

# ============================================================================
# 6. VÉRIFICATION FINALE
# ============================================================================
print("\n" + "=" * 70)
print("🔍 6. Vérification finale")
print("=" * 70)

# Vérifier qu'il n'y a pas de fuites entre les splits
train_set = set(train_files)
val_set = set(val_files)
test_set = set(test_files)

leakage_train_val = train_set & val_set
leakage_train_test = train_set & test_set
leakage_val_test = val_set & test_set

if not (leakage_train_val or leakage_train_test or leakage_val_test):
    print("✅ Aucune fuite de données détectée entre les splits")
else:
    print("❌ ATTENTION: Fuite de données détectée!")
    if leakage_train_val:
        print(f"   Train ∩ Val: {len(leakage_train_val)} fichiers")
    if leakage_train_test:
        print(f"   Train ∩ Test: {len(leakage_train_test)} fichiers")
    if leakage_val_test:
        print(f"   Val ∩ Test: {len(leakage_val_test)} fichiers")

# Vérifier que tous les fichiers sont présents
all_files = train_set | val_set | test_set
if len(all_files) == len(df):
    print(f"✅ Tous les {len(df)} fichiers sont présents dans les splits")
else:
    print(f"❌ Incohérence: {len(all_files)} fichiers dans splits vs {len(df)} dans dataset")

# ============================================================================
# RAPPORT FINAL
# ============================================================================
print("\n" + "=" * 70)
print("✨ RAPPORT FINAL")
print("=" * 70)

print(f"""
✅ Splits Train/Val/Test créés avec succès!

📊 Configuration:
   - Stratégie: Split stratifié par genre
   - Proportions: 70% / 15% / 15%
   - Random seed: 42 (reproductible)
   
📚 Statistiques:
   - Train: {len(train_files)} fichiers ({len(train_files)/len(df)*100:.1f}%)
   - Val:   {len(val_files)} fichiers ({len(val_files)/len(df)*100:.1f}%)
   - Test:  {len(test_files)} fichiers ({len(test_files)/len(df)*100:.1f}%)
   
💾 Fichiers générés:
   - data/splits/train.csv
   - data/splits/val.csv
   - data/splits/test.csv
   - data/splits/splits.json
   - data/splits/summary.json
   - results/figures/03_train_val_test_splits.png
   
🎯 Prochaines étapes:
   1. ✅ Exploration terminée
   2. ✅ Splits créés
   3. 🔄 Prochaine étape: Rechercher baseline Kaggle
   4. ⏳ Ensuite: Implémenter le baseline
   5. ⏳ Puis: Fine-tuning WAV2VEC
""")

print("=" * 70)
print("🎉 Préparation des données terminée!")
print("=" * 70)