#IMPORTING THE NECESSARY LIBRARIES

import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
import itertools

#Importing the Image

sample_img = cv2.imread('F:\Computer_vision\Open cv\Project\WhatsApp Image 2023-03-23 at 10.37.33.jpeg')
plt.figure(figsize = [15, 15])
plt.title("REAL IMAGE");plt.axis('off');plt.imshow(sample_img[:,:,::-1]);plt.show()

#Drawing rectangle on the face

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
face_detection_results = face_detection.process(sample_img[:,:,::-1])

img_copy = sample_img[:,:,::-1].copy()

if face_detection_results.detections:
    for face_no, face in enumerate(face_detection_results.detections):        
        mp_drawing.draw_detection(image=img_copy, detection=face, keypoint_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))
fig = plt.figure(figsize = [15, 15]) 
plt.title("RECTANGLE ON FACE");plt.axis('off');plt.imshow(img_copy);plt.show()

#Drawing Facial Landmarks on the face

mp_face_mesh = mp.solutions.face_mesh
face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2,min_detection_confidence=0.5)
mp_drawing_styles = mp.solutions.drawing_styles
face_mesh_results = face_mesh_images.process(sample_img[:,:,::-1])

img_copy = sample_img[:,:,::-1].copy()

if face_mesh_results.multi_face_landmarks:
    for face_landmarks in face_mesh_results.multi_face_landmarks:
        mp_drawing.draw_landmarks(image=img_copy,landmark_list=face_landmarks,connections=mp_face_mesh.FACEMESH_TESSELATION,landmark_drawing_spec=None,connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(image=img_copy, landmark_list=face_landmarks,connections=mp_face_mesh.FACEMESH_CONTOURS,landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
fig = plt.figure(figsize = [15, 15])
plt.title("IMAGE WITH FACIAL LANDMARKS");plt.axis('off');plt.imshow(img_copy);plt.show()

#CREATE A FACE LANDMARK DETECTION FUNCTION

def detectFacialLandmarks(image, face_mesh, display = True):
    results = face_mesh.process(image[:,:,::-1])
    output_image = image[:,:,::-1].copy()
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(image=output_image, landmark_list=face_landmarks,connections=mp_face_mesh.FACEMESH_TESSELATION,landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(image=output_image, landmark_list=face_landmarks,connections=mp_face_mesh.FACEMESH_CONTOURS,landmark_drawing_spec=None,connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
    if display:
        plt.figure(figsize=[1280,960])
    else:
        return np.ascontiguousarray(output_image[:,:,::-1], dtype=np.uint8),results              

#FACE LANDMARK DETECTION ON WEB CAM

face_mesh_videos = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,  min_detection_confidence=0.5,min_tracking_confidence=0.3)

camera_video = cv2.VideoCapture(0)
camera_video.set(3,1280)
camera_video.set(4,960)
cv2.namedWindow('Face Landmarks Detection', cv2.WINDOW_NORMAL)

time1 = 0

while camera_video.isOpened():
    ok, frame = camera_video.read()
    if not ok:
        continue
    frame = cv2.flip(frame, 1)
    frame, _ = detectFacialLandmarks(frame, face_mesh_videos, display=False)
    from time import time
    time2 = time()
    if (time2 - time1) > 0:
        frames_per_second = 1.0 / (time2 - time1)
        cv2.putText(frame, 'FPS: {}'.format(int(frames_per_second)), (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
    time1 = time2
    cv2.imshow('Face Landmarks Detection', frame)
    k = cv2.waitKey(1) & 0xFF    
    if(k == 27):
        break              
camera_video.release()
cv2.destroyAllWindows()

#Assigning a function to get the size

def Size(image, face_landmarks, INDEXES):
    image_height, image_width, _ = image.shape
    INDEXES_LIST = list(itertools.chain(*INDEXES))
    landmarks = []
    for INDEX in INDEXES_LIST:
        landmarks.append([int(face_landmarks.landmark[INDEX].x * image_width),int(face_landmarks.landmark[INDEX].y * image_height)])
    _, _, width, height = cv2.boundingRect(np.array(landmarks))
    landmarks = np.array(landmarks)
    return width, height, landmarks

#Assigning a function to find whether the face part is open or not

def Open(image, face_mesh_results, face_part, threshold=5, display=True):
    image_height, image_width, _ = image.shape
    output_image = image.copy()
    status={}
    if face_part == 'MOUTH':
        INDEXES = mp_face_mesh.FACEMESH_LIPS
        loc = (10, image_height - image_height//40)
        increment=-30    
    elif face_part == 'LEFT EYE':
        INDEXES = mp_face_mesh.FACEMESH_LEFT_EYE
        loc = (10, 30)
        increment=30    
    elif face_part == 'RIGHT EYE':
        INDEXES = mp_face_mesh.FACEMESH_RIGHT_EYE 
        loc = (image_width-300, 30)
        increment=30
    else:
        return
    for face_no, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):
        _, height, _ = Size(image, face_landmarks, INDEXES)
        _, face_height, _ = Size(image, face_landmarks, mp_face_mesh.FACEMESH_FACE_OVAL)
        if (height/face_height)*100 > threshold:
            status[face_no] = 'OPEN'
        else:
            status[face_no] = 'CLOSE'
        return output_image, status

#Assigning a funtion to overlay the images on the face

def overlay(image, filter_img, face_landmarks, face_part, INDEXES, display=True):
    annotated_image = image.copy()
    try:
        filter_img_height, filter_img_width, _  = filter_img.shape
        _, face_part_height, landmarks = Size(image, face_landmarks, INDEXES)
        required_height = int(face_part_height*2.5)
        resized_filter_img = cv2.resize(filter_img, (int(filter_img_width*(required_height/filter_img_height)),required_height))
        filter_img_height, filter_img_width, _  = resized_filter_img.shape
        _, filter_img_mask = cv2.threshold(cv2.cvtColor(resized_filter_img, cv2.COLOR_BGR2GRAY),25, 255, cv2.THRESH_BINARY_INV)
        center = landmarks.mean(axis=0).astype("int")
        if face_part == 'MOUTH':
            location = (int(center[0] - filter_img_width / 3), int(center[1]))
        else:  
            location = (int(center[0]-filter_img_width/2), int(center[1]-filter_img_height/2))
        ROI = image[location[1]: location[1] + filter_img_height,location[0]: location[0] + filter_img_width]
        resultant_image = cv2.bitwise_and(ROI, ROI, mask=filter_img_mask)
        resultant_image = cv2.add(resultant_image, resized_filter_img)
        annotated_image[location[1]: location[1] + filter_img_height,location[0]: location[0] + filter_img_width] = resultant_image
    except Exception as e:
        pass
    return annotated_image
    
#FILTER
camera_video = cv2.VideoCapture(0)
camera_video.set(3,1280)
camera_video.set(4,960)

left_eye = cv2.imread('F:\Computer_vision\Open cv\Project\FACE_FILTER\left_eye.png')
right_eye = cv2.imread('F:\Computer_vision\Open cv\Project\FACE_FILTER\\right_eye.png')
fire_animation = cv2.VideoCapture('F:\Computer_vision\Open cv\Project\FACE_FILTER\smoke_animation.mp4')

cv2.namedWindow('Filter', cv2.WINDOW_NORMAL)

smoke_frame_counter = 0

while camera_video.isOpened():
    ok, frame = camera_video.read()
    if not ok:
        continue
    _, smoke_frame = fire_animation.read()
    smoke_frame_counter += 1
    if smoke_frame_counter == fire_animation.get(cv2.CAP_PROP_FRAME_COUNT):   
        fire_animation.set(cv2.CAP_PROP_POS_FRAMES, 0)
        smoke_frame_counter = 0
    frame = cv2.flip(frame, 1)
    _, face_mesh_results = detectFacialLandmarks(frame, face_mesh_videos, display=False)
    if face_mesh_results.multi_face_landmarks:
        _, mouth_status = Open(frame, face_mesh_results, 'MOUTH', threshold=15, display=False)
        _, left_eye_status = Open(frame, face_mesh_results, 'LEFT EYE', threshold=4.5 , display=False)
        _, right_eye_status = Open(frame, face_mesh_results, 'RIGHT EYE', threshold=4.5, display=False)
        for face_num, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):
            if left_eye_status[face_num] == 'OPEN':
                frame = overlay(frame, left_eye, face_landmarks,'LEFT EYE', mp_face_mesh.FACEMESH_LEFT_EYE, display=False)
            if right_eye_status[face_num] == 'OPEN':
                frame = overlay(frame, right_eye, face_landmarks,'RIGHT EYE', mp_face_mesh.FACEMESH_RIGHT_EYE, display=False)
            if mouth_status[face_num] == 'OPEN':
                frame = overlay(frame, smoke_frame, face_landmarks, 'MOUTH', mp_face_mesh.FACEMESH_LIPS, display=False)
    cv2.imshow('Filter', frame)
    k = cv2.waitKey(1) & 0xFF    
    if(k == 27):
        break                 
camera_video.release()
cv2.destroyAllWindows()