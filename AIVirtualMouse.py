
import pyautogui
import cv2
import numpy as np
import HandTrackingModule as htm
import time

# Mouse controller


# Parameters
wCam, hCam = 640, 480  # Camera resolution
speed = 4
frameR = 100  # Frame reduction for active region
smoothening = 5 # Smoothening factor for mouse movement

# Variables
pTime = 0  # Previous time for FPS calculation
plocX, plocY = 0, 0  # Previous mouse location
clocX, clocY = 0, 0  # Current mouse location

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(3, wCam)  # Set width
cap.set(4, hCam)  # Set height

# Hand detector
detector = htm.handDetector(maxHands=1)  # Detect 1 hand for simplicity
wScr, hScr = pyautogui.size()  # Screen width and height

# Main loop
while True:
    # 1. Find hand landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # 2. Get index and middle finger tip positions
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]  # Index finger tip
        x2, y2 = lmList[12][1:]  # Middle finger tip

        # 3. Check which fingers are up
        fingers = detector.fingersUp()

        # Draw frame boundary for the active region
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        # 4. Moving mode (only index finger up)
        if fingers[1] == 1 and fingers[2] == 0:
            # Convert coordinates
            x3 = np.interp(x1, (0, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (0, hCam - frameR), (0, hScr))

            # Smoothen values
            clocX = plocX + (x3 - plocX) / (smoothening/speed)
            clocY = plocY + (y3 - plocY) / (smoothening/speed)

            # Move mouse
            pyautogui.moveTo(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)

            # Update previous location
            plocX, plocY = clocX, clocY

        # 5. Clicking mode (both index and middle fingers up)
        if fingers[1] == 1 and fingers[2] == 1:
            # Find distance between fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)

            # If distance is short, perform a click
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                pyautogui.click()

    # 6. Frame rate calculation
    cTime = time.time()
    fps = 1 / max(cTime - pTime, 1e-5)  # Avoid division by zero
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # 7. Display
    cv2.imshow("Image", img)

    # Break loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
