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
import random

# Semilla para reproducibilidad
np.random.seed(42)
random.seed(42)

n_lotes = 500

# Bloque 1: Variables Agronómicas
variedades = np.random.choice(['Lapins', 'Santina', 'Regina'], size=n_lotes, p=[0.5, 0.3, 0.2])
macrozonas = np.random.choice([0, 1], size=n_lotes, p=[0.6, 0.4])

firmeza_base = np.random.normal(loc=9.0, scale=1.2, size=n_lotes)
firmeza_cosecha = np.where(macrozonas == 1, firmeza_base + 1.5, firmeza_base)
firmeza_cosecha = np.clip(firmeza_cosecha, 6.0, 12.0)

brix_cosecha = np.random.normal(loc=18.0, scale=2.0, size=n_lotes)
brix_cosecha = np.clip(brix_cosecha, 14.0, 24.0)

dias_embarque = np.random.randint(3, 15, size=n_lotes)

# Bloque 2: Tecnología de Empaque
tecnologia_empaque = np.random.choice([0, 1], size=n_lotes, p=[0.7, 0.3])
protocolo_ca = np.random.choice([0, 1], size=n_lotes, p=[0.1, 0.9])

vida_util_max = np.where(tecnologia_empaque == 0, 35, 
                         np.where((tecnologia_empaque == 1) & (protocolo_ca == 1), 42, 35))

co2 = np.where(tecnologia_empaque == 0, 
               np.random.uniform(2.0, 4.0, size=n_lotes), 
               np.where(protocolo_ca == 1, 
                        np.random.uniform(8.0, 10.0, size=n_lotes), 
                        np.random.uniform(12.1, 15.0, size=n_lotes))) 

# Armado del DataFrame
df_mcrl = pd.DataFrame({
    'Lote_ID': [f'CONT-{i+1:04d}' for i in range(n_lotes)],
    'X1_Variedad': variedades,
    'X3_Macrozona': macrozonas,
    'X4_Firmeza_Cosecha': np.round(firmeza_cosecha, 2),
    'X5_Brix': np.round(brix_cosecha, 1),
    'X9_Dias_Embarque': dias_embarque,
    'X13_Tecnologia': tecnologia_empaque,
    'X14_Protocolo_CA': protocolo_ca,
    'X15_CO2': np.round(co2, 2),
    'X_Vida_Maxima': vida_util_max
})

# ==========================================
# SPRINT 2: LOGÍSTICA Y PERFILES TÉRMICOS
# ==========================================

# Bloque H: Tiempos operativos previos al viaje (en días o fracción)
t_cos = np.random.uniform(1/24, 2/24, size=n_lotes)  # 1-2 horas cosecha
t_ac = np.random.uniform(3/24, 6/24, size=n_lotes)   # 3-6 horas acopio
t_tr = np.random.uniform(1/24, 3/24, size=n_lotes)   # 1-3 horas transporte
t_rec = np.random.uniform(1/24, 3/24, size=n_lotes)  # 1-3 horas recepción
t_CMP = np.random.uniform(1, 4, size=n_lotes)        # 1-4 días cámara MP

# X19: Tipo de servicio naviero
tipo_servicio = np.random.choice(['Express', 'Estándar', 'Contingencia'], size=n_lotes, p=[0.70, 0.28, 0.02])

# X20: Tiempo total de tránsito real (depende del servicio)
rutas_express = [21, 22, 23, 24] # Destinos: HK (21), Nansha (22), Shanghái (23), +1 día congestión
prob_express = [0.25, 0.35, 0.30, 0.10] # Más barcos van a Nansha y Shanghái

rutas_estandar = [28, 30, 32, 35] # Escalas en Callao (28), Busan (30-32) o transbordos (35)
prob_estandar = [0.40, 0.30, 0.20, 0.10]

rutas_contingencia = [40, 45, 52] # Retrasos graves por clima o pérdida de conexión
prob_contingencia = [0.60, 0.30, 0.10]

t_viaje = np.where(tipo_servicio == 'Express', 
                   np.random.choice(rutas_express, size=n_lotes, p=prob_express),
                   np.where(tipo_servicio == 'Estándar', 
                            np.random.choice(rutas_estandar, size=n_lotes, p=prob_estandar),
                            np.random.choice(rutas_contingencia, size=n_lotes, p=prob_contingencia)))

# Temperaturas en Tránsito (Simulación del Datalogger)
# X21: Temp promedio del contenedor en tránsito (-0.5 a 2.5 °C)
temp_promedio = np.random.normal(loc=0.0, scale=0.8, size=n_lotes)
temp_promedio = np.clip(temp_promedio, -0.5, 2.5)

# X23: N° eventos de desviación térmica (Peaks)
eventos_desviacion = np.random.poisson(lam=0.3, size=n_lotes)
eventos_desviacion = np.clip(eventos_desviacion, 0, 15)

# X24: Duración total de desviaciones térmicas (cada evento suma entre 2 y 24h)
duracion_desviaciones = eventos_desviacion * np.random.uniform(2, 24, size=n_lotes) 

# X22: Temp máxima registrada
temp_maxima = temp_promedio + np.where(eventos_desviacion > 0, np.random.uniform(1.5, 8.0, size=n_lotes), np.random.uniform(0.1, 0.5, size=n_lotes))
temp_maxima = np.clip(temp_maxima, -0.5, 8.0)

# Añadimos todo al DataFrame Maestro
df_mcrl['t_CMP_dias'] = np.round(t_CMP, 1)
df_mcrl['X19_Servicio_Naviero'] = tipo_servicio
df_mcrl['X20_Tiempo_Viaje'] = t_viaje
df_mcrl['X21_Temp_Promedio'] = np.round(temp_promedio, 2)
df_mcrl['X22_Temp_Max'] = np.round(temp_maxima, 2)
df_mcrl['X23_Eventos_Termicos'] = eventos_desviacion
df_mcrl['X24_Duracion_Desviaciones_h'] = np.round(duracion_desviaciones, 1)

# ==========================================
# SPRINT 3: VARIABLES RESTANTES Y BLOQUE H (HUMEDAD)
# ==========================================

# Variables agronómicas y de postcosecha restantes (Tablas 4.2 y 4.3)
df_mcrl['X2_Horas_Frio'] = np.random.randint(400, 1201, size=n_lotes)
df_mcrl['X6_Color'] = np.random.randint(1, 6, size=n_lotes) # Escala 1-5
df_mcrl['X7_Calibre'] = np.random.choice(['XL', 'J', '2J', '3J', '4J'], size=n_lotes, p=[0.1, 0.3, 0.4, 0.15, 0.05])
df_mcrl['X8_Pitting_Cosecha'] = np.round(np.random.uniform(0, 15, size=n_lotes), 1)

df_mcrl['X10_Tiempo_Hidro_h'] = np.round(np.random.uniform(1, 8, size=n_lotes), 1)
df_mcrl['X11_Temp_Hidro'] = np.round(np.random.uniform(0, 4, size=n_lotes), 1)
df_mcrl['X12_Temp_Camara'] = np.round(np.random.uniform(-0.5, 1.5, size=n_lotes), 1)
df_mcrl['X16_Calcio'] = np.random.choice([0, 1], size=n_lotes)
df_mcrl['X17_Elicitores'] = np.random.choice([0, 1], size=n_lotes)
# Asumimos que el packing exporta SOLO fruta visualmente excelente (Pedicelos Verdes 4.5 o 5.0)
df_mcrl['X18_Pedicelo_Embalaje'] = np.random.choice([4.5, 5.0], size=n_lotes, p=[0.2, 0.8])

# Bloque H: Cadena de Humedad (Tabla 4.4)
df_mcrl['H1_Esponja'] = np.random.choice([0, 1], size=n_lotes, p=[0.15, 0.85])
df_mcrl['H2_Hum_Acopio'] = np.random.choice([0, 1], size=n_lotes, p=[0.1, 0.9])
df_mcrl['H3_Trans_Cerrado'] = np.random.choice([0, 1], size=n_lotes, p=[0.05, 0.95])
df_mcrl['H4_Hum_Recepcion'] = np.random.choice([0, 1], size=n_lotes, p=[0.1, 0.9])

# H5 Condicional: Solo aplica si t_CMP > 3 días AND H2=1 AND H4=1
condicion_h5 = (df_mcrl['t_CMP_dias'] > 3) & (df_mcrl['H2_Hum_Acopio'] == 1) & (df_mcrl['H4_Hum_Recepcion'] == 1)
df_mcrl['H5_Hum_CMP'] = np.where(condicion_h5, np.random.choice([0, 1], size=n_lotes), 0)

# Resto Bloque 3: Variables de Destino
df_mcrl['X26_Demora_Destino_dias'] = np.random.randint(0, 16, size=n_lotes)
df_mcrl['X27_Datalogger_Valido'] = np.random.choice([0, 1], size=n_lotes, p=[0.05, 0.95])
df_mcrl['X28_Temp_Destino'] = np.round(np.random.uniform(0, 8, size=n_lotes), 1)

# ==========================================
# SPRINT 4: MÓDULO CINÉTICO Y ARRHENIUS (Capa 2)
# ==========================================

T_ref = 273.15 # Temperatura de referencia 0°C en Kelvin

# Función de Arrhenius según tu Tabla 4.1
def calcular_k(coef_temp, constante, temp_celsius):
    T_kelvin = temp_celsius + 273.15
    return np.exp(coef_temp / T_kelvin + constante)

# Constantes de referencia (k) a 0°C (Tabla 4.1)
k_ref_firmeza = calcular_k(-4844.4, 15.294, 0)
k_ref_pedicelo = calcular_k(-5973.9, 18.209, 0)
k_ref_mda = calcular_k(-8011.0, 26.709, 0)

# Separamos el tiempo de viaje en "tiempo normal" y "tiempo con quiebres térmicos"
t_peak_dias = df_mcrl['X24_Duracion_Desviaciones_h'] / 24.0
t_base_dias = df_mcrl['X20_Tiempo_Viaje'] - t_peak_dias

# 1. Cálculo de t_eq (Tiempo equivalente de deterioro) integrando los peaks
def calcular_teq(coef_temp, constante, k_ref):
    k_base = calcular_k(coef_temp, constante, df_mcrl['X21_Temp_Promedio'])
    k_peak = calcular_k(coef_temp, constante, df_mcrl['X22_Temp_Max'])
    return ((k_base * t_base_dias) + (k_peak * t_peak_dias)) / k_ref

df_mcrl['teq_firmeza'] = np.round(calcular_teq(-4844.4, 15.294, k_ref_firmeza), 2)
df_mcrl['teq_pedicelo'] = np.round(calcular_teq(-5973.9, 18.209, k_ref_pedicelo), 2)
df_mcrl['teq_mda'] = np.round(calcular_teq(-8011.0, 26.709, k_ref_mda), 2)

# 2. Cálculo de Vida Útil Consumida (VUC) en porcentaje
df_mcrl['VUC_%'] = np.round((df_mcrl['teq_firmeza'] / df_mcrl['X_Vida_Maxima']) * 100, 1)

# 3. Cálculo Determinista de Y2: Pérdida de Peso (Tasas Sebastián Johnson)
# Tasas por hora (%/h)
k0_cos = np.where(df_mcrl['H1_Esponja'] == 1, 0.08, 0.19)
k0_ac  = np.where(df_mcrl['H2_Hum_Acopio'] == 1, 0.02, 0.31)
k0_tr  = np.where(df_mcrl['H3_Trans_Cerrado'] == 1, 0.09, 0.57)
k0_rec = np.where(df_mcrl['H4_Hum_Recepcion'] == 1, 0.02, 0.31)

# Tasas por día (%/día)
k0_cmp = np.where(df_mcrl['H5_Hum_CMP'] == 1, 0.02, 0.09)

# ÍNDICE DE EFICIENCIA DE HUMEDAD PREVIA (0 a 1)
# Pesos basados en la tabla de ahorro: Acopio 38%, Recepción 29%, Transp 16%, CMP 10%, Cosecha 7%
eficiencia_humedad = (
    (df_mcrl['H2_Hum_Acopio'] * 0.38) +
    (df_mcrl['H4_Hum_Recepcion'] * 0.29) +
    (df_mcrl['H3_Trans_Cerrado'] * 0.16) +
    (df_mcrl['H5_Hum_CMP'] * 0.10) +
    (df_mcrl['H1_Esponja'] * 0.07)
)

# Tasa de viaje dinámica: se mueve linealmente entre 0.008 (sin protección previa) y 0.004 (protección total)
k0_viaje = 0.008 - (0.004 * eficiencia_humedad)

# Multiplicamos el tiempo (en horas o días) por la tasa de deshidratación
Y2_cos = k0_cos * (t_cos * 24)
Y2_ac  = k0_ac * (t_ac * 24)
Y2_tr  = k0_tr * (t_tr * 24)
Y2_rec = k0_rec * (t_rec * 24)
Y2_cmp = k0_cmp * t_CMP
Y2_viaje = k0_viaje * df_mcrl['X20_Tiempo_Viaje']

# Sumatoria de Y2 (Total Pérdida de Peso)
df_mcrl['Y2_Perdida_Peso_%'] = np.round(Y2_cos + Y2_ac + Y2_tr + Y2_rec + Y2_cmp + Y2_viaje, 2)

# ==========================================
# SPRINT 5: VARIABLES DE SALIDA (TARGETS) Y SEMÁFORO
# ==========================================

# Y1: Firmeza a destino (kg/cm2) -> AC retiene la firmeza (Tasas reales ajustadas)
tasa_ablandamiento = np.where(df_mcrl['X14_Protocolo_CA'] == 1, 0.012, 0.035) 
df_mcrl['Y1_Firmeza_Final'] = np.round(df_mcrl['X4_Firmeza_Cosecha'] - (df_mcrl['teq_firmeza'] * tasa_ablandamiento), 2)
df_mcrl['Y1_Firmeza_Final'] = np.clip(df_mcrl['Y1_Firmeza_Final'], 3.0, 12.0)

# Y3: Condición de Pedicelo (Escala 1-5)
# Efecto de "Estrés Hídrico Latente": Si hubo mala humedad en campo (eficiencia baja), el tejido muere silenciosamente y se pardea en el viaje.
dano_oculto_campo = (1 - eficiencia_humedad) * 1.5 # Castiga hasta 1.5 puntos extra si no se usó tecnología de humedad

df_mcrl['Y3_Pedicelo_Destino'] = np.round(
    df_mcrl['X18_Pedicelo_Embalaje'] 
    - (df_mcrl['teq_pedicelo'] * 0.01) 
    - (df_mcrl['Y2_Perdida_Peso_%'] * 0.4) 
    - dano_oculto_campo, 1
)
df_mcrl['Y3_Pedicelo_Destino'] = np.clip(df_mcrl['Y3_Pedicelo_Destino'], 1.0, 5.0)

# Y4: Pardeamiento Interno (%) y Y5: Pitting (%)
df_mcrl['Y4_Pardeamiento_%'] = np.round(df_mcrl['teq_mda'] * np.random.uniform(0.1, 0.3, size=n_lotes), 1)
df_mcrl['Y5_Pitting_%'] = np.round(df_mcrl['X8_Pitting_Cosecha'] + (df_mcrl['teq_firmeza'] * np.random.uniform(0.05, 0.2, size=n_lotes)), 1)

# ==========================================
# Y6: VARIABLE OBJETIVO PRINCIPAL - NIVEL DE RIESGO LOGÍSTICO (Tabla 4.7)
# 0 = Verde (Bajo), 1 = Amarillo (Medio), 2 = Naranja (Alto), 3 = Rojo (Crítico)
# ==========================================

condiciones = [
    # ROJO (Siniestro/Rechazo): Pérdida total de valor comercial o reclamo a seguro
    (df_mcrl['VUC_%'] >= 120) | (df_mcrl['Y1_Firmeza_Final'] < 5.0) | (df_mcrl['Y2_Perdida_Peso_%'] >= 6.0) | (df_mcrl['Y3_Pedicelo_Destino'] <= 1.5),
    
    # NARANJA (Remate/Castigo Severo): Fruta fatigada, se vende a pérdida rápida
    (df_mcrl['VUC_%'] >= 100) | (df_mcrl['Y1_Firmeza_Final'] < 6.0) | (df_mcrl['Y2_Perdida_Peso_%'] >= 4.0) | (df_mcrl['Y3_Pedicelo_Destino'] <= 2.5),
    
    # AMARILLO (Desgaste natural): Fruta aceptable, pero con menor vida útil residual (requiere venta rápida)
    (df_mcrl['VUC_%'] >= 85) | (df_mcrl['Y1_Firmeza_Final'] < 7.0) | (df_mcrl['Y2_Perdida_Peso_%'] >= 2.0) | (df_mcrl['Y3_Pedicelo_Destino'] <= 3.5)
]

# Si no cae en ninguna (ej: VUC < 85%, Firmeza >= 7.0, Pérdida < 2.0%), es VERDE (Premium)
valores_riesgo = ['Rojo', 'Naranja', 'Amarillo']
df_mcrl['Y6_Riesgo_Logistico'] = np.select(condiciones, valores_riesgo, default='Verde')

print("\n--- ¡Target Calculado: Semáforo de Riesgo! ---")
print(df_mcrl[['Lote_ID', 'Y1_Firmeza_Final', 'Y2_Perdida_Peso_%', 'Y3_Pedicelo_Destino', 'VUC_%', 'Y6_Riesgo_Logistico']].head(15))

# Exportar el Dataset completo a Excel
df_mcrl.to_excel('Dataset_MCRL_Completo.xlsx', index=False)
print("\n¡ÉXITO TOTAL! El archivo 'Dataset_MCRL_Completo.xlsx' ha sido guardado en tu carpeta.")
