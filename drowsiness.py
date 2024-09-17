from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import cv2
import face_alignment

# Initialize the mixer
mixer.init()
mixer.music.load("drowniness/music.wav")


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


# Parameters
thresh = 0.24
frame_check = 15   ##to check for drowsiness. If the EAR is below the threshold for this many frames, an alert will be triggered.
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cpu')
cap = cv2.VideoCapture(0)
flag = 0     ##Initializes a flag to count consecutive frames where the EAR is below the threshold.

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect facial landmarks
    landmarks = fa.get_landmarks_from_image(rgb_frame)

    if landmarks:
        for landmark_set in landmarks:
            # Extract eye landmarks
            leftEye = landmark_set[42:48]  # Left eye landmarks
            rightEye = landmark_set[36:42]  # Right eye landmarks

            # Calculate Eye Aspect Ratio (EAR)
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            # Draw eye contours
            leftEyeHull = cv2.convexHull(leftEye.astype("int"))
            rightEyeHull = cv2.convexHull(rightEye.astype("int"))
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # Check if EAR is below the threshold
            if ear < thresh:
                flag += 1
                print(flag)
                if flag >= frame_check:
                    cv2.putText(frame, "****************ALERT!****************", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "****************ALERT!****************", (10, 325),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    mixer.music.play()
            else:
                flag = 0

    # Display the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
cap.release()