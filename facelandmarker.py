from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
import cv2
import face_recognition

model_path = "/Users/glenchua/Documents/thesis_project/face_landmarker.task"

# Change input picture here
IMAGE_PATH = '4Head.jpg'

def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image

def plot_face_blendshapes_bar_graph(face_blendshapes):
  # Extract the face blendshapes category names and scores.
  face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
  face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
  # The blendshapes are ordered in decreasing score value.
  face_blendshapes_ranks = range(len(face_blendshapes_names))

  fig, ax = plt.subplots(figsize=(12, 12))
  bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
  ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
  ax.invert_yaxis()

  # Label each bar with values
  for score, patch in zip(face_blendshapes_scores, bar.patches):
    plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

  ax.set_xlabel('Score')
  ax.set_title("Face Blendshapes")
  plt.tight_layout()
  plt.show()

  
def get_top_expressions(result):
  ''' Retrieves the most dominant expressions from 
  the result and returns the emotion based on
  the expressions dict mapping

  :param result: result from face detection
  :return str -> expression
  '''
  expressions = {
  'happy': ['mouthSmileRight', 'mouthSmileLeft', 'mouthUpperUpRight',
            'mouthUpperUpLeft', 'eyeSquintLeft', 'eyeSquintRight'],
  'surprised': ['browInnerUp', 'jawOpen', 'browOuterUpRight',
                'browOuterUpLeft', 'eyeWideLeft','eyeWideRight',
                'mouthStretchLeft', 'mouthStretchRight'],
  'sad': ['mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthFrownRight', 
          'mouthFrownLeft', 'browDownRight', 'browDownLeft',
          'eyeLookDownLeft', 'eyeLookDownRight'],
  'angry': ['browDownLeft', 'browDownRight', 'noseSneerRight', 
            'noseSneerLeft', 'mouthPressLeft', 'mouthPressRight', 
            'jawForward', 'eyeSquintLeft','eyeSquintRight'],
  'disgust': ['noseSneerLeft', 'noseSneerRight', 'upperLipRaiseLeft', 
            'upperLipRaiseRight', 'cheekSquintLeft', 'cheekSquintRight']
  }
  if result:
    blendshapes = result.face_blendshapes[0]
    top_landmark = max(blendshapes, key=lambda x:x.score)
    if top_landmark.score > 0.5:
      for expression, landmarks in expressions.items():
        for landmark in landmarks:
          if landmark in top_landmark.category_name:
            return expression.title()
    else:
      return "Neutral"
    


BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

base_options = python.BaseOptions(model_asset_path=model_path)
options = FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=5)

if __name__ == "__main__":
  with FaceLandmarker.create_from_options(options) as landmarker:
    # The landmarker is initialized. Use it here.
    
    mp_image = mp.Image.create_from_file(IMAGE_PATH)
    bgr_image = cv2.cvtColor(mp_image.numpy_view(), cv2.COLOR_RGB2BGR) # numpy array in opencv format
    

    # Run the result
    face_landmarker_result = landmarker.detect(mp_image)
    top_expression = get_top_expressions(face_landmarker_result)
    print(top_expression)

    annotated_image = draw_landmarks_on_image(bgr_image, face_landmarker_result)
    cv2.imshow("window", annotated_image)

  plot_face_blendshapes_bar_graph(face_landmarker_result.face_blendshapes[0])
 


  


