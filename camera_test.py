import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not accessible")
else:
    print("✅ Camera opened successfully")

ret, frame = cap.read()

if ret:
    print("✅ Frame captured")
    cv2.imshow("Test Frame", frame)
    cv2.waitKey(0)
else:
    print("❌ Failed to capture frame")

cap.release()
cv2.destroyAllWindows()
