import pandas as pd
import os

# Carpetas / archivos
carpeta = "data_game"
archivo_csv1 = os.path.join(carpeta, "partidas_046.csv")
archivo_csv2 = os.path.join(carpeta, "partida_ppt_Ana_2.csv")
archivo_salida = os.path.join(carpeta, "partidas_test.csv")

# Leer los CSV
df1 = pd.read_csv(archivo_csv1)
df2 = pd.read_csv(archivo_csv2)

# Concatenar
df_comb = pd.concat([df1, df2], ignore_index=True)

# Reasignar el número de ronda consecutivamente
df_comb['numero_ronda'] = range(1, len(df_comb) + 1)

# Guardar el CSV combinado
df_comb.to_csv(archivo_salida, index=False)

print(f"Archivo combinado guardado como '{archivo_salida}'.")
print(df_comb.head(10))

