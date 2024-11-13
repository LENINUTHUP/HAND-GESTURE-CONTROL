import cv2
import mediapipe as mp
import time
import pyautogui
import screen_brightness_control as sbc
import numpy as np

# Define constants
THUMB_INDEX_DIST_THRESHOLD = 50
PINKY_WRIST_DIST_THRESHOLD = 110
PINKY_WRIST_DIST_BRIGHTNESS_RANGE = [10, 70]
BRIGHTNESS_RANGE = [0, 100]
CLICK_DIST_THRESHOLD = 20

# Initialize variables
x1 = y1 = x2 = y2 = x3 = y3 = x4 = y4 = 0
hand_detected = False
mouse_smoothing = 0.5  # adjust this value to control the smoothing
last_mouse_x, last_mouse_y = 0, 0

webcam = cv2.VideoCapture(0)
my_hands = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    _, image = webcam.read()
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output = my_hands.process(rgb_image)
    hands = output.multi_hand_landmarks

    if hands:
        hand_detected = True
        for hand in hands:
            drawing_utils.draw_landmarks(image, hand)
            landmark = hands[0].landmark
            for id, landmark in enumerate(landmark):
                h, w, c = image.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                if id == 8:  # Index
                    cv2.circle(image, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                    x1, y1 = cx, cy
                if id == 4:  # Thumb
                    cv2.circle(image, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                    x2, y2 = cx, cy
                if id == 20:  # Pinky
                    cv2.circle(image, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                    x3, y3 = cx, cy
                if id == 0:  # Wrist
                    cv2.circle(image, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                    x4, y4 = cx, cy

            # Calculate distances
            dist = np.linalg.norm(np.array([x2, y2]) - np.array([x1, y1]))
            dist_bright = np.linalg.norm(np.array([x4, y4]) - np.array([x3, y3]))
            dist_click = np.linalg.norm(np.array([x2, y2]) - np.array([x1, y1]))

            cv2.line(image, (x1, y1), (x2, y2), (189, 189, 189), 5)
            cv2.line(image, (x3, y3), (x4, y4), (189, 189, 189), 5)

            # Brightness control
            if dist_bright > PINKY_WRIST_DIST_THRESHOLD:
                b_level = np.interp(dist_bright - PINKY_WRIST_DIST_THRESHOLD, PINKY_WRIST_DIST_BRIGHTNESS_RANGE, BRIGHTNESS_RANGE)
                sbc.set_brightness(int(b_level))

            # Volume control
            if dist > THUMB_INDEX_DIST_THRESHOLD:
                pyautogui.press("volumeup")
            else:
                pyautogui.press("volumedown")

            # Click button
            if dist_click < CLICK_DIST_THRESHOLD:
                pyautogui.click()

            # Get screen resolution
            screen_width, screen_height = pyautogui.size()

            # Scale webcam coordinates to screen coordinates
            scaled_x = int(x1 / w * screen_width)
            scaled_y = int(y1 / h * screen_height)

            # Apply mouse smoothing
            mouse_x = int(last_mouse_x * mouse_smoothing + scaled_x * (1 - mouse_smoothing))
            mouse_y = int(last_mouse_y * mouse_smoothing + scaled_y * (1 - mouse_smoothing))
            last_mouse_x, last_mouse_y = mouse_x, mouse_y

            # Move mouse cursor to scaled coordinates
            pyautogui.moveTo(mouse_x, mouse_y)

    else:
        hand_detected = False

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", image)
    cv2.waitKey(1)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()