import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0)


cap.set(cv2.CAP_PROP_FPS, 20)
print(f"Camera FPS: {cap.get(cv2.CAP_PROP_FPS)}")
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)

while True:
    start = time.time()
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    print(f"camera fps: {1./(time.time()-start)}")

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
