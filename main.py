
# hand_brightness.py
import cv2
import mediapipe as mp
from math import hypot
import screen_brightness_control as sbc
import numpy as np

def main():
    # --- Setup MediaPipe hand detector ---
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75,
        max_num_hands=1)   # set to 1 for simplicity
    draw_utils = mp.solutions.drawing_utils

    # --- Open webcam (change index if you have multiple cameras) ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera. Try changing the index (0 -> 1) or grant camera permission.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Frame not received from camera.")
                break

            frame = cv2.flip(frame, 1)  # mirror image so it's intuitive
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            lm_list = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # collect landmarks (id, x_px, y_px)
                    h, w, _ = frame.shape
                    for idx, lm in enumerate(hand_landmarks.landmark):
                        x_px, y_px = int(lm.x * w), int(lm.y * h)
                        lm_list.append([idx, x_px, y_px])
                    # draw the skeleton on the frame
                    draw_utils.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

            # if we detected landmarks, get thumb tip(4) and index tip(8)
            if lm_list:
                x1, y1 = lm_list[4][1], lm_list[4][2]   # thumb tip
                x2, y2 = lm_list[8][1], lm_list[8][2]   # index tip

                # draw markers and line
                cv2.circle(frame, (x1, y1), 7, (0, 255, 0), cv2.FILLED)
                cv2.circle(frame, (x2, y2), 7, (0, 255, 0), cv2.FILLED)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # distance between tips
                length = hypot(x2 - x1, y2 - y1)

                # map hand distance range to brightness 0-100
                # tweak [15, 220] if your camera/hand distance differs
                brightness = int(np.interp(length, [15, 220], [0, 100]))

                # try to set system brightness — may fail on some systems/monitors
                try:
                    sbc.set_brightness(brightness)
                except Exception as e:
                    # we don't crash — show a warning on the frame
                    cv2.putText(frame, "Brightness control unavailable", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                cv2.putText(frame, f"Brightness: {brightness}%", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # show frame and quit on 'q'
            cv2.imshow("Hand Brightness Control", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
