"""
Piedra, Papel o Tijera con Webcam
Detecta gestos de la mano usando OpenCV
"""

import cv2
import numpy as np
import time
import os
from collections import Counter

# Importar la IA
from rps_ai import RPSAIPlayer, BEATS, MOVE_MAP


class HandGestureDetector:
    """Detector de gestos de mano para RPS."""

    def __init__(self):
        # Rango de color de piel en HSV - MAS AMPLIO
        # Rango 1: tonos rosados/claros
        self.lower_skin1 = np.array([0, 20, 60], dtype=np.uint8)
        self.upper_skin1 = np.array([20, 180, 255], dtype=np.uint8)
        # Rango 2: tonos mas oscuros
        self.lower_skin2 = np.array([160, 20, 60], dtype=np.uint8)
        self.upper_skin2 = np.array([180, 180, 255], dtype=np.uint8)

    def detect_hand(self, frame):
        """Detecta la mano y cuenta los dedos."""
        # Suavizar imagen para reducir ruido
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)

        # Convertir a HSV
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Mascara de piel con dos rangos
        mask1 = cv2.inRange(hsv, self.lower_skin1, self.upper_skin1)
        mask2 = cv2.inRange(hsv, self.lower_skin2, self.upper_skin2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Limpieza morfologica
        kernel = np.ones((5, 5), np.uint8)

        # Eliminar ruido pequeno
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Cerrar huecos
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Blur final
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Encontrar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, 0, mask

        # Contorno mas grande (la mano)
        max_contour = max(contours, key=cv2.contourArea)

        # Filtrar por area minima MAS GRANDE
        area = cv2.contourArea(max_contour)
        if area < 8000:
            return None, 0, mask

        # Filtrar por solidez (area / area_convex_hull)
        hull_area = cv2.contourArea(cv2.convexHull(max_contour))
        if hull_area > 0:
            solidity = area / hull_area
            if solidity < 0.4:  # Muy irregular, probablemente no es mano
                return None, 0, mask

        # Convex hull
        hull = cv2.convexHull(max_contour, returnPoints=False)

        # Contar dedos usando convexity defects
        if len(hull) > 3:
            defects = cv2.convexityDefects(max_contour, hull)

            if defects is not None:
                finger_count = 0
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(max_contour[s][0])
                    end = tuple(max_contour[e][0])
                    far = tuple(max_contour[f][0])

                    # Calcular angulo entre dedos
                    a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                    b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                    c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

                    angle = np.arccos((b**2 + c**2 - a**2) / (2*b*c + 1e-6))

                    # Si el angulo es menor a 90 grados, es un dedo
                    if angle <= np.pi / 2 and d > 10000:
                        finger_count += 1
                        cv2.circle(frame, far, 5, (0, 0, 255), -1)

                # Dedos = defectos + 1 (aproximado)
                finger_count = min(finger_count + 1, 5)

                return max_contour, finger_count, mask

        return max_contour, 1, mask

    def classify_gesture(self, finger_count):
        """Clasifica el gesto basado en el numero de dedos."""
        if finger_count <= 1:
            return 'piedra'  # Puno cerrado
        elif finger_count == 2:
            return 'tijera'  # Dos dedos (indice y medio)
        elif finger_count >= 4:
            return 'papel'   # Mano abierta
        else:
            return 'papel'   # 3 dedos, asumimos papel


def draw_game_ui(frame, round_num, ai_stats, countdown=None, result=None,
                 player_move=None, ai_move=None, gesture=None, fingers=None,
                 calibrating=False, hand_detected=False):
    """Dibuja la interfaz del juego en el frame."""
    h, w = frame.shape[:2]

    # Fondo semi-transparente para texto
    overlay = frame.copy()

    # Panel superior
    cv2.rectangle(overlay, (0, 0), (w, 70), (40, 40, 40), -1)

    # Panel inferior
    cv2.rectangle(overlay, (0, h-80), (w, h), (40, 40, 40), -1)

    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    # Titulo
    cv2.putText(frame, "PIEDRA PAPEL TIJERA vs IA", (w//2 - 180, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Ronda y Stats
    cv2.putText(frame, f"Ronda: {round_num}", (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, ai_stats, (w - 300, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Area de deteccion
    margin = 30
    box_color = (0, 255, 0) if hand_detected else (100, 100, 100)
    box_thickness = 3 if hand_detected else 2
    cv2.rectangle(frame, (margin, 80), (w//2 - 20, h - 90), box_color, box_thickness)

    if not hand_detected and not calibrating and countdown is None:
        cv2.putText(frame, "PON TU MANO AQUI", (margin + 50, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)

    # Calibracion
    if calibrating:
        cv2.putText(frame, "CALIBRANDO...", (w//2 - 100, h//2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, "No pongas la mano", (w//2 - 100, h//2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Countdown
    if countdown is not None:
        if countdown > 0:
            # Circulo con numero
            cv2.circle(frame, (w//2, h//2), 80, (0, 200, 200), -1)
            cv2.putText(frame, str(countdown), (w//2 - 25, h//2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 5)
        else:
            cv2.circle(frame, (w//2, h//2), 80, (0, 255, 0), -1)
            cv2.putText(frame, "YA!", (w//2 - 40, h//2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

    # Gesto detectado
    if gesture and fingers is not None and hand_detected:
        # Icono del gesto
        icons = {'piedra': '[*]', 'papel': '[=]', 'tijera': '[V]'}
        icon = icons.get(gesture, '?')
        cv2.putText(frame, f"{icon} {gesture.upper()}", (20, h - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Resultado
    if result:
        # Fondo del resultado
        result_bg = (0, 150, 0) if "GANASTE" in result else (0, 0, 150) if "IA" in result else (150, 150, 0)
        cv2.rectangle(frame, (w//2 - 120, h//2 - 30), (w//2 + 120, h//2 + 30), result_bg, -1)
        cv2.putText(frame, result, (w//2 - 100, h//2 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if player_move and ai_move:
        cv2.putText(frame, f"Tu: {player_move.upper()}  vs  IA: {ai_move.upper()}",
                    (w//2 - 140, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Instrucciones
    cv2.putText(frame, "[ESPACIO] Jugar  [R] Reset  [Q] Salir",
                (w//2 - 160, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    return frame


def play_webcam_game():
    """Juego principal con webcam."""
    # Inicializar
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se puede abrir la webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    detector = HandGestureDetector()

    # Cargar IA
    print("Cargando IA...")
    ai = RPSAIPlayer()
    base_path = os.path.dirname(os.path.abspath(__file__))
    ai.train(base_path)

    round_num = 0
    game_state = "waiting"  # waiting, countdown, capturing, result
    countdown_start = None
    result_text = None
    player_move = None
    ai_move = None
    last_gesture = None
    last_fingers = 0
    hand_detected = False

    print("\n" + "="*50)
    print("  PIEDRA PAPEL TIJERA - WEBCAM")
    print("="*50)
    print("  [ESPACIO] = Jugar ronda")
    print("  [R] = Reiniciar marcador")
    print("  [Q] = Salir")
    print("="*50)
    print("\n  Pon tu mano en el recuadro y pulsa ESPACIO!")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Voltear horizontalmente (espejo)
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Region de interes
        roi = frame[80:h-90, 30:w//2-20].copy()

        # Detectar mano
        contour, fingers, mask = detector.detect_hand(roi)

        hand_detected = contour is not None
        if hand_detected:
            # Dibujar contorno en verde
            cv2.drawContours(roi, [contour], -1, (0, 255, 0), 2)
            # Dibujar convex hull
            hull = cv2.convexHull(contour)
            cv2.drawContours(roi, [hull], -1, (255, 0, 0), 2)
            last_gesture = detector.classify_gesture(fingers)
            last_fingers = fingers

        # Maquina de estados
        if game_state == "waiting":
            result_text = None

        elif game_state == "countdown":
            elapsed = time.time() - countdown_start
            countdown_val = 3 - int(elapsed)

            if countdown_val < 0:
                game_state = "capturing"
                capture_time = time.time()

        elif game_state == "capturing":
            if time.time() - capture_time > 0.3:
                if hand_detected and last_gesture:
                    player_move = last_gesture
                    round_num += 1

                    ai.register_opponent_move(player_move)
                    ai.register_my_move(ai_move)

                    result = ai.get_result(ai_move, player_move)

                    if result == "empate":
                        result_text = "EMPATE!"
                    elif result == "gano_ia":
                        result_text = "GANA IA!"
                    else:
                        result_text = "GANASTE!"

                    game_state = "result"
                    result_time = time.time()
                else:
                    result_text = "SIN MANO!"
                    game_state = "result"
                    result_time = time.time()

        elif game_state == "result":
            if time.time() - result_time > 2.0:
                game_state = "waiting"
                player_move = None
                result_text = None

        # Countdown value
        countdown_val = None
        if game_state == "countdown":
            countdown_val = max(0, 3 - int(time.time() - countdown_start))

        # Dibujar UI
        frame = draw_game_ui(
            frame, round_num, ai.get_stats(),
            countdown=countdown_val,
            result=result_text,
            player_move=player_move,
            ai_move=ai_move if game_state == "result" else None,
            gesture=last_gesture if hand_detected else None,
            fingers=last_fingers if hand_detected else None,
            calibrating=False,
            hand_detected=hand_detected
        )

        # Mostrar mascara en esquina (mas pequena)
        mask_small = cv2.resize(mask, (120, 90))
        mask_color = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
        frame[75:165, w-130:w-10] = mask_color
        cv2.putText(frame, "Mascara", (w-120, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        cv2.imshow("Piedra Papel Tijera vs IA", frame)

        # Controles
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' ') and game_state == "waiting":
            game_state = "countdown"
            countdown_start = time.time()
            ai_move, _, _ = ai.choose_move()
        elif key == ord('r'):
            ai.wins = 0
            ai.losses = 0
            ai.ties = 0
            ai.history = []
            ai.my_history = []
            ai.results = []
            round_num = 0
            game_state = "waiting"

    # Resultado final
    print(f"\n{'='*50}")
    print("  RESULTADO FINAL")
    print(f"{'='*50}")
    print(f"  {ai.get_stats()}")
    print(f"{'='*50}\n")

    cap.release()
    cv2.destroyAllWindows()


def main():
    """Punto de entrada."""
    print("""
    ____  ____  ____     __        __  ______  ____  ___    __  ___
   / __ \\/ __ \\/ __ \\   / /       / / / / __ \\/ __ \\/   |  /  |/  /
  / /_/ / /_/ / / / /  / / _ /|_ / /_/ / / / / / / / /| | / /|_/ /
 / _, _/ ____/ /_/ /  / /_/ / // / __  / /_/ / /_/ / ___ |/ /  / /
/_/ |_/_/    \\____/   \\__/_/  /_/_/ /_/\\____/\\____/_/  |_/_/  /_/

  PIEDRA - PAPEL - TIJERA con WEBCAM
    """)

    play_webcam_game()


if __name__ == "__main__":
    main()
