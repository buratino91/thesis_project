import numpy as np
import mediapipe as mp
import cv2
import time
from facelandmarker import get_top_expressions
from mediapipe.framework.formats import landmark_pb2
import face_recognition

latest_frame = None

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

mp_drawing = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

model_path = "/Users/glenchua/Documents/thesis_project/face_landmarker.task"

images = ["ben.jpeg", "shafah.jpeg", "mark.jpeg", "glen.jpeg", "ian.jpeg"]
known_face_encodings = []
known_face_names = [
        "Ben",
        "Shafah",
        "Mark",
        "Glen",
        "Ian"
]

def learn_to_recognize(images):
  '''Function to learn to recognize images and append to know_face_encodings

    :param images: list -> path of images to be loaded
    :return: face_encodings 
  '''
  for image in images:
    loaded_image = face_recognition.load_image_file(image)
    loaded_image_encoding = face_recognition.face_encodings(loaded_image)[0]
    known_face_encodings.append(loaded_image_encoding)



def face_detected(result, output_image, timestamp_ms):
    global latest_frame
    latest_frame = output_image.numpy_view().copy()
    image_height, image_width, _ = latest_frame.shape

    if not result.face_landmarks:
        return

    for idx, landmarks in enumerate(result.face_landmarks):
        # Convert normalized landmarks to pixel coordinates
        xs = [int(landmark.x * image_width) for landmark in landmarks]
        ys = [int(landmark.y * image_height) for landmark in landmarks]

        # Calculate bounding box
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # Draw bounding box
        cv2.rectangle(
            latest_frame,
            (x_min, y_min),
            (x_max, y_max),
            (255, 0, 0),  # Green box
            2
        )        
        expression = get_top_expressions(result)
        if expression:
            labels = [f"Expression: {expression}"]
            for i, text in enumerate(labels):
                y_offset = y_min - 20 + i * 20
                cv2.putText(
                    latest_frame,
                    text,
                    (x_min, y_offset),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.8,
                    (0, 255, 0), # Green text
                    2,
                    cv2.LINE_AA
                )
        # Convert to protobuf format for drawing mesh
        landmark_proto = landmark_pb2.NormalizedLandmarkList()
        landmark_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lmk.x, y=lmk.y, z=lmk.z)
            for lmk in landmarks
        ])
        # Draw face mesh (tessellation + contours + irises)
        mp_drawing.draw_landmarks(
            image=latest_frame,
            landmark_list=landmark_proto,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_style.get_default_face_mesh_tesselation_style()
        )
        mp_drawing.draw_landmarks(
            image=latest_frame,
            landmark_list=landmark_proto,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_style.get_default_face_mesh_contours_style()
        )
        mp_drawing.draw_landmarks(
            image=latest_frame,
            landmark_list=landmark_proto,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_style.get_default_face_mesh_iris_connections_style()
        )


if __name__ =='__main__':
    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    # Load facelandmarker options
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=face_detected,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True)
    
    # Load images and recognize them 
    learn_to_recognize(images)
  
   

    # Create facelandmarker from options
    with FaceLandmarker.create_from_options(options) as landmarker:
        start_time = time.time()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        
        while True:
            ret, frame = cap.read()
            color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert from cv to mediapipe
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=color)
            frame_timestamp_ms = int((time.time() - start_time) * 1000)

            # Detect faces
            landmarker.detect_async(mp_image, frame_timestamp_ms)
            if process_this_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                rgb_small_frame = small_frame[:, :, ::-1]
                rgb_small_frame = cv2.cvtColor(rgb_small_frame , cv2.COLOR_BGR2RGB)

                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"
                
                    # Compare distances between known face and new face
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                    face_names.append(name)

            process_this_frame = not process_this_frame

            # Display results
            if latest_frame is not None:
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    top *= 4
                    right *= 4
                    bottom *=4
                    left *= 4
                    
                    # Draw a label with a name below the face
                    label_top = max(top - 35, 0)
                    label_bottom = top
                    cv2.rectangle(latest_frame, (left, bottom), (right, bottom + 35), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(latest_frame, name, (left + 6, bottom + 25), font, 1.0, (255, 255, 255), 1)

                cv2.imshow("window", cv2.cvtColor(latest_frame, cv2.COLOR_RGB2BGR))
        
            if not ret:
                print("Can't receive from frame (stream end?). Exiting...")
                break

            if cv2.waitKey(1) == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()