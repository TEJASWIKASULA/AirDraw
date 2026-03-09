import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

canvas = None
prev_x, prev_y = 0, 0

# Colors
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
color_index = 0
draw_color = colors[color_index]

erase_mode = False
last_switch_time = 0

def fingers_up(hand_landmarks):
    tips = [8, 12, 16, 20]
    fingers = []

    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            finger_state = fingers_up(hand_landmarks)

            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)

            # DRAW MODE (only index finger up)
            if finger_state[0] == 1 and sum(finger_state) == 1:
                erase_mode = False
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y

                cv2.line(canvas, (prev_x, prev_y), (x, y), draw_color, 5)
                prev_x, prev_y = x, y

            # COLOR SWITCH MODE (index + middle up)
            elif finger_state[0] == 1 and finger_state[1] == 1:
                if time.time() - last_switch_time > 1:
                    color_index = (color_index + 1) % len(colors)
                    draw_color = colors[color_index]
                    last_switch_time = time.time()

                prev_x, prev_y = 0, 0

            # ERASE MODE (fist)
            elif sum(finger_state) == 0:
                erase_mode = True
                cv2.circle(canvas, (x, y), 20, (0, 0, 0), -1)
                prev_x, prev_y = 0, 0

            else:
                prev_x, prev_y = 0, 0

    frame = cv2.add(frame, canvas)

    # Display current mode
    mode_text = "ERASE MODE" if erase_mode else "DRAW MODE"
    cv2.putText(frame, f"Mode: {mode_text}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(frame, "Press 's' to Save | 'q' to Quit", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Air Draw - Research Demo", frame)

    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):
        break

    if key & 0xFF == ord('s'):
        filename = f"drawing_{int(time.time())}.png"
        cv2.imwrite(filename, canvas)
        print(f"Saved as {filename}")

cap.release()
cv2.destroyAllWindows()
