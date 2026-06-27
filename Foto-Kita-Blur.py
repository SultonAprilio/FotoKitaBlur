import cv2
import numpy as np
import mediapipe as mp


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
  static_image_mode=False, 
  max_num_hands=2,
  min_detection_confidence=0.5,
  min_tracking_confidence=0.5
)

def finger_up(tip, pip, landmarks):
  return landmarks[tip].y < landmarks[pip].y

def is_peace(landmarks):
  index_up = finger_up(8, 6 , landmarks)
  middle_up = finger_up(12, 10, landmarks)
  ring_up = finger_up(16, 14, landmarks)
  pinky_up = finger_up(20, 18, landmarks)
  return (
    index_up
    and middle_up
    and not ring_up
    and not pinky_up
  )
  
cap = cv2.VideoCapture(0)

while True:
  succes, frame = cap.read()
  if not succes:
    break
  
  rgb = cv2.cvtColor(
    frame,
    cv2.COLOR_BGR2RGB
  )
  
  hand_result = hands.process(rgb)
  
  peace_detected = False
  
  if hand_result.multi_hand_landmarks:
    for hand_landmarks in hand_result.multi_hand_landmarks:
      mp_draw.draw_landmarks(
        frame,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
        mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
      )
      if is_peace(hand_landmarks.landmark):
        peace_detected = True
        
      
  if peace_detected:
    frame = cv2.GaussianBlur(
      frame,
      (61, 61),
      0
    )
  
  cv2.imshow("Peace Blur", frame)
  
  if cv2.waitKey(1) & 0xFF in (27, ord('q')):
    break
  
  
  
cap.release()
cv2.destroyAllWindows()