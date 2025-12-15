import pandas as pd
import os

# Ruta de la carpeta
carpeta = "data_game"
os.makedirs(carpeta, exist_ok=True)

# Archivos de entrada y salida
archivo_entrada = os.path.join(carpeta, "input_online.csv")
archivo_salida = os.path.join(carpeta, "partidas_online.csv")

# Función para determinar el ganador
def determinar_resultado(j1, j2):
    if j1 == j2:
        return "empate"
    elif (j1 == "piedra" and j2 == "tijera") or \
         (j1 == "tijera" and j2 == "papel") or \
         (j1 == "papel" and j2 == "piedra"):
        return "gana_j1"
    else:
        return "gana_j2"

# Leer CSV de entrada
df = pd.read_csv(archivo_entrada)

# Calcular resultado
df['resultado'] = df.apply(lambda row: determinar_resultado(row['jugada_j1'], row['jugada_j2']), axis=1)

# Calcular cambia_j1 y cambia_j2 comparando con la jugada anterior
df['cambia_j1'] = (df['jugada_j1'] != df['jugada_j1'].shift(1)).fillna(False)
df['cambia_j2'] = (df['jugada_j2'] != df['jugada_j2'].shift(1)).fillna(False)

# Guardar CSV actualizado
df.to_csv(archivo_salida, index=False)

print(f"Archivo '{archivo_salida}' generado correctamente.")
print(df.head(10))
