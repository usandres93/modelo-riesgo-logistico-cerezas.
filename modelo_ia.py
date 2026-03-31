# =============================================================================
# PROYECTO DE TÍTULO: MODELO MCRL (Calificación de Riesgo Logístico)
# AUTOR: Andres Uribe - Estudiante de Ingeniería Civil Industrial
# INSTITUCIÓN: Universidad Mayor, Santiago de Chile
# FECHA: Marzo 2026
# -----------------------------------------------------------------------------
# DERECHOS DE AUTOR: Este código es propiedad intelectual de Andres Uribe.
# Desarrollado exclusivamente para fines académicos y defensa de título.
# Queda prohibida su reproducción total o parcial sin autorización expresa.
# =============================================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score, roc_auc_score)
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("  MCRL — Fase 4: Módulo ML (RF vs XGBoost) v2.0")
print("=" * 60)

# ── 1. CARGA Y PREPARACIÓN ───────────────────────────────────────
df = pd.read_excel('Dataset_MCRL_Completo.xlsx')

features = [
    'X1_Variedad', 'X3_Macrozona', 'X4_Firmeza_Cosecha', 'X13_Tecnologia',
    'X19_Servicio_Naviero', 'X20_Tiempo_Viaje', 'X21_Temp_Promedio',
    'X24_Duracion_Desviaciones_h', 'teq_firmeza', 'VUC_%',
    'H1_Esponja', 'H2_Hum_Acopio', 'H3_Trans_Cerrado', 'H4_Hum_Recepcion'
]

X = df[features].copy()
y = df['Y6_Riesgo_Logistico']

# Encoders separados para evitar conflictos
le_var = LabelEncoder()
le_nav = LabelEncoder()
le_y   = LabelEncoder()

X['X1_Variedad']          = le_var.fit_transform(X['X1_Variedad'])
X['X19_Servicio_Naviero'] = le_nav.fit_transform(X['X19_Servicio_Naviero'])
y_encoded = le_y.fit_transform(y)
clases    = le_y.classes_

print(f"\nClases objetivo : {list(clases)}")
print(f"Distribución    : {dict(zip(clases, np.bincount(y_encoded)))}")

# División estratificada 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

print(f"\nEntrenamiento : {len(X_train)} lotes")
print(f"Prueba        : {len(X_test)} lotes")

# ── 2. CONFIGURACIÓN DE MODELOS ──────────────────────────────────
rf_model = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    max_depth=None,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    eval_metric='mlogloss',
    random_state=42,
    verbosity=0
)

pesos_train = compute_sample_weight(class_weight='balanced', y=y_train)

# ── 3. VALIDACIÓN CRUZADA (5-Fold) ──────────────────────────────
print("\n" + "─" * 60)
print("[ Validación Cruzada Estratificada — 5 Folds ]")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores_rf_acc  = cross_val_score(rf_model,  X, y_encoded, cv=cv, scoring='accuracy')
scores_rf_f1   = cross_val_score(rf_model,  X, y_encoded, cv=cv, scoring='f1_macro')
scores_xgb_acc = cross_val_score(xgb_model, X, y_encoded, cv=cv, scoring='accuracy')
scores_xgb_f1  = cross_val_score(xgb_model, X, y_encoded, cv=cv, scoring='f1_macro')

print(f"\n{'Modelo':<20} {'Accuracy CV':>18} {'F1-Macro CV':>18}")
print("─" * 60)
print(f"{'Random Forest':<20} "
      f"{scores_rf_acc.mean():.2%} ±{scores_rf_acc.std()*2:.2%}   "
      f"{scores_rf_f1.mean():.4f} ±{scores_rf_f1.std()*2:.4f}")
print(f"{'XGBoost':<20} "
      f"{scores_xgb_acc.mean():.2%} ±{scores_xgb_acc.std()*2:.2%}   "
      f"{scores_xgb_f1.mean():.4f} ±{scores_xgb_f1.std()*2:.4f}")

# ── 4. ENTRENAMIENTO FINAL ───────────────────────────────────────
rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train, sample_weight=pesos_train)

y_pred_rf  = rf_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)
y_prob_rf  = rf_model.predict_proba(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)

# ── 5. MÉTRICAS COMPARATIVAS ─────────────────────────────────────
print("\n" + "─" * 60)
print("[ Desempeño en Test Set — RF vs XGBoost ]")

for nombre, y_pred, y_prob in [
    ("Random Forest", y_pred_rf,  y_prob_rf),
    ("XGBoost",       y_pred_xgb, y_prob_xgb)
]:
    acc    = accuracy_score(y_test, y_pred)
    f1_mac = f1_score(y_test, y_pred, average='macro')
    f1_wgt = f1_score(y_test, y_pred, average='weighted')
    try:
        auc = roc_auc_score(
            label_binarize(y_test, classes=range(4)),
            y_prob, multi_class='ovr', average='macro')
    except Exception:
        auc = float('nan')

    print(f"\n  [{nombre}]")
    print(f"  Accuracy    : {acc:.4f}  ({acc:.2%})")
    print(f"  F1 Macro    : {f1_mac:.4f}")
    print(f"  F1 Weighted : {f1_wgt:.4f}")
    print(f"  AUC-ROC     : {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=clases))

# ── 6. ANÁLISIS DE ERRORES CRÍTICOS ──────────────────────────────
print("─" * 60)
print("[ Análisis de Falsos Negativos Críticos — Random Forest ]")

df_test = X_test.copy()
df_test['real']     = le_y.inverse_transform(y_test)
df_test['predicho'] = le_y.inverse_transform(y_pred_rf)

fn_amarillo = df_test[
    (df_test['real'] == 'Amarillo') & (df_test['predicho'] == 'Verde')
]
print(f"\n  Lotes Amarillo clasificados como Verde : {len(fn_amarillo)}")
if len(fn_amarillo) > 0:
    print(f"  VUC% promedio de estos lotes          : {fn_amarillo['VUC_%'].mean():.1f}%")
    print(f"  VUC% máximo                           : {fn_amarillo['VUC_%'].max():.1f}%")
    print(f"  VUC% mínimo                           : {fn_amarillo['VUC_%'].min():.1f}%")
    print(f"  teq_firmeza promedio                  : {fn_amarillo['teq_firmeza'].mean():.1f} días-eq")

# ── 7. GRÁFICOS ──────────────────────────────────────────────────
sns.set_theme(style='whitegrid', context='paper', font_scale=1.1)
COLOR_RF  = '#008080'
COLOR_XGB = '#2c7bb6'

# GRÁFICO 1 — Matriz de Confusión RF
fig, ax = plt.subplots(figsize=(7, 5))
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=clases, yticklabels=clases,
            ax=ax, linewidths=0.5)
ax.set_title('Matriz de Confusión — Random Forest Balanceado',
             fontsize=13, pad=15)
ax.set_ylabel('Riesgo Real (Observado)', fontsize=11)
ax.set_xlabel('Riesgo Predicho (Modelo)', fontsize=11)
plt.tight_layout()
plt.savefig('IA_Matriz_Confusion_v2.png', dpi=300,
            bbox_inches='tight', pad_inches=0.3)
print("\n-> Guardado: IA_Matriz_Confusion_v2.png")

# GRÁFICO 2 — Importancia de Variables RF Balanceado
imp = pd.Series(rf_model.feature_importances_,
                index=features).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.bar(imp.index, imp.values, color=COLOR_RF,
              edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, imp.values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f'{val*100:.1f}%',
            ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.set_title('Importancia de Variables — Random Forest Balanceado (Gini)',
             fontsize=13, pad=15)
ax.set_ylabel('Nivel de Importancia (Reducción Impureza Gini)', fontsize=11)
ax.set_xlabel('Variable', fontsize=11)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.tight_layout()
plt.savefig('IA_Importancia_Variables_v2.png', dpi=300,
            bbox_inches='tight', pad_inches=0.3)
print("-> Guardado: IA_Importancia_Variables_v2.png")

# GRÁFICO 3 — Comparación CV: RF vs XGBoost
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

datos = [
    (axes[0], [scores_rf_acc.mean(), scores_xgb_acc.mean()],
               [scores_rf_acc.std()*2, scores_xgb_acc.std()*2],
               'Accuracy — Validación Cruzada (5-Fold)', 0.80, '{:.2%}'),
    (axes[1], [scores_rf_f1.mean(),  scores_xgb_f1.mean()],
               [scores_rf_f1.std()*2,  scores_xgb_f1.std()*2],
               'F1-Score Macro — Validación Cruzada (5-Fold)', 0.50, '{:.3f}'),
]

for ax, vals, errs, titulo, ymin, fmt in datos:
    bars = ax.bar(['Random Forest', 'XGBoost'], vals,
                  yerr=errs, capsize=7,
                  color=[COLOR_RF, COLOR_XGB],
                  edgecolor='black', linewidth=0.8, width=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + 0.012,
                fmt.format(val),
                ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    ax.set_ylim(ymin, 1.08)
    ax.set_title(titulo, fontsize=11, pad=10)
    ax.set_ylabel('Valor de la Métrica', fontsize=10)

plt.suptitle('Comparación RF vs XGBoost — Validación Cruzada Estratificada',
             fontsize=13, fontweight='bold', y=1.00)
plt.subplots_adjust(top=0.88)
plt.tight_layout()
plt.savefig('IA_Comparacion_Modelos.png', dpi=300,
            bbox_inches='tight', pad_inches=0.3)
print("-> Guardado: IA_Comparacion_Modelos.png")

print("\n" + "=" * 60)
print("  ¡Fase 4 v2.0 Completada! 3 gráficos generados.")
print("=" * 60)
