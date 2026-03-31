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
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print("Iniciando Análisis Exploratorio de Datos (EDA)...")

# 1. Cargar el dataset generado
try:
    df = pd.read_excel('Dataset_MCRL_Completo.xlsx')
    print("Dataset cargado correctamente.")
except FileNotFoundError:
    print("Error: No se encontró el archivo Excel. Asegúrate de haber ejecutado el generador primero.")
    exit()

# Configurar el estilo de los gráficos para que se vean académicos y limpios
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# Diccionario de colores oficiales para el semáforo
colores_riesgo = {'Verde': '#2ecc71', 'Amarillo': '#f1c40f', 'Naranja': '#e67e22', 'Rojo': '#e74c3c'}
orden_riesgo = ['Verde', 'Amarillo', 'Naranja', 'Rojo']

# =====================================================================
# GRÁFICO 1: Distribución del Nivel de Riesgo Logístico (Semáforo)
# =====================================================================
plt.figure(figsize=(8, 6))
ax = sns.countplot(data=df, x='Y6_Riesgo_Logistico', order=orden_riesgo, palette=colores_riesgo)
plt.title('Distribución del Nivel de Riesgo Logístico en la Temporada', fontsize=14, pad=15)
plt.xlabel('Categoría de Riesgo', fontsize=12)
plt.ylabel('Cantidad de Contenedores', fontsize=12)

# Añadir porcentajes arriba de cada barra
total = len(df)
for p in ax.patches:
    porcentaje = f'{100 * p.get_height() / total:.1f}%'
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(porcentaje, (x, y), ha='center', va='bottom', fontsize=11, fontweight='bold', xytext=(0, 5), textcoords='offset points')

plt.tight_layout()
plt.savefig('EDA_1_Distribucion_Riesgo.png', dpi=300)
print("-> Gráfico 1 guardado: EDA_1_Distribucion_Riesgo.png")

# =====================================================================
# GRÁFICO 2: Impacto del Tiempo de Viaje en la Vida Útil Consumida (VUC)
# =====================================================================
plt.figure(figsize=(9, 6))
sns.boxplot(data=df, x='X19_Servicio_Naviero', y='VUC_%', palette='Set2')
plt.axhline(100, color='red', linestyle='--', linewidth=2, label='Límite Crítico (100% Vida Útil)')
plt.title('Consumo de Vida Útil (VUC) según Tipo de Servicio Naviero', fontsize=14, pad=15)
plt.xlabel('Servicio Naviero', fontsize=12)
plt.ylabel('Vida Útil Consumida (%)', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig('EDA_2_Impacto_Navieras.png', dpi=300)
print("-> Gráfico 2 guardado: EDA_2_Impacto_Navieras.png")

# =====================================================================
# GRÁFICO 3: El "Daño Oculto" (Pérdida de Peso vs Calidad del Pedicelo)
# =====================================================================
plt.figure(figsize=(9, 6))
sns.scatterplot(data=df, x='Y2_Perdida_Peso_%', y='Y3_Pedicelo_Destino', hue='Y6_Riesgo_Logistico', 
                palette=colores_riesgo, hue_order=orden_riesgo, s=80, alpha=0.8, edgecolor='k')
plt.axhline(2.5, color='orange', linestyle='--', linewidth=1.5, label='Límite Pedicelo Pardo')
plt.title('Efecto de la Deshidratación en la Condición del Pedicelo a Destino', fontsize=14, pad=15)
plt.xlabel('Pérdida de Peso Total (%)', fontsize=12)
plt.ylabel('Condición del Pedicelo (1=Pardo, 5=Verde)', fontsize=12)
plt.legend(title='Riesgo Final')
plt.tight_layout()
plt.savefig('EDA_3_Dano_Oculto.png', dpi=300)
print("-> Gráfico 3 guardado: EDA_3_Dano_Oculto.png")

# =====================================================================
# GRÁFICO 4: Matriz de Correlación (Heatmap)
# =====================================================================
columnas_clave = [
    'X20_Tiempo_Viaje', 'X21_Temp_Promedio', 'X24_Duracion_Desviaciones_h', 
    'teq_firmeza', 'VUC_%', 'Y1_Firmeza_Final', 'Y2_Perdida_Peso_%', 'Y3_Pedicelo_Destino'
]

nombres_legibles = {
    'X20_Tiempo_Viaje': 'Días Tránsito',
    'X21_Temp_Promedio': 'Temp Promedio',
    'X24_Duracion_Desviaciones_h': 'Horas Quiebre Térmico',
    'teq_firmeza': 'Tiempo Equiv. Deterioro',
    'VUC_%': 'Vida Útil Consumida (%)',
    'Y1_Firmeza_Final': 'Firmeza Destino',
    'Y2_Perdida_Peso_%': 'Deshidratación (%)',
    'Y3_Pedicelo_Destino': 'Calidad Pedicelo'
}

df_corr = df[columnas_clave].rename(columns=nombres_legibles)
matriz_correlacion = df_corr.corr(method='spearman')

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(matriz_correlacion, dtype=bool))

sns.heatmap(matriz_correlacion, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
            vmin=-1, vmax=1, square=True, linewidths=.5, cbar_kws={"shrink": .7})
plt.title('Matriz de Correlación: Variables Logísticas vs Calidad Final', fontsize=14, pad=15)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('EDA_4_Matriz_Correlacion.png', dpi=300)
print("-> Gráfico 4 guardado: EDA_4_Matriz_Correlacion.png")

# =====================================================================
# GRÁFICO 5: Impacto Agronómico (Variedad vs Nivel de Riesgo)
# =====================================================================
plt.figure(figsize=(9, 6))
# Crear tabla cruzada de porcentajes (Variedad vs Semáforo)
variedad_riesgo = pd.crosstab(df['X1_Variedad'], df['Y6_Riesgo_Logistico'], normalize='index') * 100
variedad_riesgo = variedad_riesgo.reindex(columns=orden_riesgo) # Ordenar Verde a Rojo

colores_stack = [colores_riesgo[c] for c in variedad_riesgo.columns]

variedad_riesgo.plot(kind='bar', stacked=True, color=colores_stack, figsize=(9, 6), edgecolor='k')
plt.title('Distribución del Riesgo Logístico por Variedad Genética', fontsize=14, pad=15)
plt.xlabel('Variedad de Cereza', fontsize=12)
plt.ylabel('Porcentaje de Contenedores (%)', fontsize=12)
plt.legend(title='Riesgo Final', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('EDA_5_Variedad_vs_Riesgo.png', dpi=300)
print("-> Gráfico 5 guardado: EDA_5_Variedad_vs_Riesgo.png")

# =====================================================================
# GRÁFICO 6: Distribución de Firmeza a Destino (Campanas de Densidad)
# =====================================================================
plt.figure(figsize=(9, 6))
sns.kdeplot(data=df, x='Y1_Firmeza_Final', hue='Y6_Riesgo_Logistico', 
            palette=colores_riesgo, hue_order=orden_riesgo, 
            fill=True, common_norm=False, alpha=0.5, linewidth=2)
plt.axvline(7.0, color='red', linestyle='--', linewidth=1.5, label='Límite de Firmeza Comercial (7.0)')
plt.title('Distribución de la Firmeza a Destino según Nivel de Riesgo', fontsize=14, pad=15)
plt.xlabel('Firmeza a Destino (kg/cm2)', fontsize=12)
plt.ylabel('Densidad (Frecuencia)', fontsize=12)
plt.legend(title='Riesgo Final')
plt.tight_layout()
plt.savefig('EDA_6_Distribucion_Firmeza.png', dpi=300)
print("-> Gráfico 6 guardado: EDA_6_Distribucion_Firmeza.png")


print("\n¡EDA Completado! Revisa tu carpeta, tienes 6 imágenes nuevas listas para tu Tesis.")