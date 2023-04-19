import cv2

# Loading the face cascade
face_cascade = cv2.CascadeClassifier('C:/Users/PC/Desktop/cascade/haarcascade_frontalface_default.xml')

# Open the default camera
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Unable to access camera")
    exit()

# Initialize the frame counter and total number of frames
frame_counter = 0
total_frames = 0

# Loop through frames from the camera
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Check if frame was successfully read
    if not ret:
        print("Unable to read frame")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Loop through the detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Extract the region of interest (ROI) for the eyes
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes in the ROI for drowsiness detection
        eyes = cv2.CascadeClassifier('C:/Users/PC/Desktop/cascade/haarcascade_eye.xml').detectMultiScale(roi_gray)

        # Loop through the detected eyes
        for (ex, ey, ew, eh) in eyes:
            # Draw a rectangle around the eyes
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)


        # Increment the total number of frames
        total_frames += 1

    # Display the frame in a window6
    cv2.imshow("Camera", frame)

    # Wait for user to press 'q' key to exit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
