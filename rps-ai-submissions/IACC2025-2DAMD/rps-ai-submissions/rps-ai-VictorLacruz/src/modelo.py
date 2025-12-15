import os
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import warnings

JUGADA_A_NUM = {"piedra": 0, "papel": 1, "tijera": 2}
NUM_A_JUGADA = {0: "piedra", 1: "papel", 2: "tijera"}
GANA_A = {"piedra": "tijera", "papel": "piedra", "tijera": "papel"}
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# ===========================================================
# CARGA Y PREPARACION DE DATOS
# ===========================================================
def cargar_datos(ruta_csv="../data/partidas.csv"):
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"No se encontró el CSV en: {ruta_csv}")
    df = pd.read_csv(ruta_csv)
    return df


def preparar_datos(df):
    df.columns = df.columns.str.strip()
    df["victor"] = df["victor"].str.strip().str.lower()
    df["prof"] = df["prof"].str.strip().str.lower().map({"si": 1, "no": 0})

    le = LabelEncoder()
    df["enemigo_enc"] = le.fit_transform(df["enemigo"])
    df["victor_enc"] = le.fit_transform(df["victor"])

    X = df[["enemigo_enc"]]
    y = df["victor_enc"]
    pesos = df["prof"].apply(lambda x: 3 if x == 1 else 1)

    return X, y, pesos, le


# ===========================================================
# ENTRENAMIENTO DEL MODELO
# ===========================================================
def entrenar_modelo(X, y, pesos):
    X_train, X_test, y_train, y_test, pesos_train, pesos_test = train_test_split(
        X, y, pesos, test_size=0.2, random_state=42
    )
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train, sample_weight=pesos_train)

    score = modelo.score(X_test, y_test)
    print(f"Exactitud del modelo: {score * 100:.2f}%")
    return modelo


# ===========================================================
# CLASE JUGADOR IA
# ===========================================================
class JugadorIA:
    def __init__(self, modelo_path="models/rps_modelo_historial.pkl", le_path="models/le_enemigo.pkl"):
        self.modelo = None
        self.historial = []

        # Cargar modelo entrenado
        if os.path.exists(modelo_path):
            self.modelo = joblib.load(modelo_path)
        else:
            print(f"[!] No se encontró el modelo en {modelo_path}")

        # Cargar LabelEncoder
        if os.path.exists(le_path):
            self.le_enemigo = joblib.load(le_path)
        else:
            self.le_enemigo = LabelEncoder()
            self.le_enemigo.classes_ = np.array(["piedra", "papel", "tijera"])
            print(f"[!] No se encontró LabelEncoder en {le_path}, usando default")

    def decidir_jugada(self):
        # 20% de aleatoriedad para evitar estancamiento
        if not self.historial or not self.modelo or np.random.rand() < 0.2:
            return np.random.choice(["piedra", "papel", "tijera"])

        # Usar las últimas 3 jugadas del humano si existen
        ultimos_enemigos = [h[0] for h in self.historial[-3:]]  # max 3 últimas
        predicciones = []

        for enemigo in ultimos_enemigos:
            try:
                enemigo_enc = self.le_enemigo.transform([enemigo])
                pred_enc = self.modelo.predict([enemigo_enc])[0]
                predicciones.append(NUM_A_JUGADA[pred_enc])
            except:
                predicciones.append(np.random.choice(["piedra", "papel", "tijera"]))

        # Elegir la predicción más común o al azar si hay empate
        jugada_modelo = max(set(predicciones), key=predicciones.count)

        # 30% de aleatoriedad para no quedarse fijo
        if np.random.rand() < 0.3:
            return np.random.choice(["piedra", "papel", "tijera"])
        else:
            return jugada_modelo

    def registrar_ronda(self, jugada_humano, jugada_ia):
        self.historial.append((jugada_humano, jugada_ia))


# ===========================================================
# SCRIPT PRINCIPAL DE ENTRENAMIENTO
# ===========================================================
def main():
    print("=" * 50)
    print("   ENTRENANDO MODELO RPS-AI CON HISTORIAL")
    print("=" * 50)

    df = cargar_datos()
    X, y, pesos, le_enemigo = preparar_datos(df)
    modelo = entrenar_modelo(X, y, pesos)

    os.makedirs("models", exist_ok=True)
    joblib.dump(modelo, "models/rps_modelo_historial.pkl")
    joblib.dump(le_enemigo, "models/le_enemigo.pkl")
    print("Modelo guardado en models/rps_modelo_historial.pkl")
    print("LabelEncoder guardado en models/le_enemigo.pkl")


if __name__ == "__main__":
    main()
