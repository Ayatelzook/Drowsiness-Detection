# Drowsiness Detection with Eye Aspect Ratio

This project detects drowsiness by monitoring the user's eye aspect ratio (EAR) in real-time using facial landmarks. If the EAR falls below a certain threshold for a specified number of frames, an alert is triggered along with an audio signal.

###  Eye Landmark Extraction

For each set of landmarks detected:

- The code extracts landmarks for the left and right eyes.
- The EAR is calculated for both eyes.
- 

###  Drawing Eye Contours

The contours of the eyes are drawn on the frame using `cv2.drawContours()` for visual feedback.

###  Drowsiness Detection

- If the EAR is below the threshold, a flag is incremented.
- If the flag exceeds `frame_check`, an alert message is displayed on the screen, and the audio alert is played.
- If the EAR is above the threshold, the flag is reset.
