from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

model_path = "/Users/glenchua/Documents/thesis_project/face_landmarker.task"

expressions = {
  'happy': ['mouthSmileRight', 'mouthSmileLeft'],
  'surprised': ['browInnerUp', 'jawOpen'],
  'sad': ['mouthLowerDownLeft', 'mouthLowerDownRight']
}

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

def get_expressions(face_blendshapes):
  if face_blendshapes[44].score + face_blendshapes[45].score > 0.6:
    return 'happy'
  elif face_blendshapes[34].score + face_blendshapes[35].score > 0.6:
    return 'sad'
  elif face_blendshapes[3].score + face_blendshapes[25].score > 0.6:
    return 'surprised'
  else:
    return 'Expression not detected'
  
def get_top_expressions(result):
  if result:
    blendshapes = result.face_blendshapes[0]
    top_expression = max(blendshapes, key=lambda x:x.score)
    print(f"Dominant expression: {top_expression.category_name} ({top_expression.score:.2f})")
    return top_expression.category_name


    

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

with FaceLandmarker.create_from_options(options) as landmarker:
  # The landmarker is initialized. Use it here.
  
  mp_image = mp.Image.create_from_file('/Users/glenchua/Pictures/DE3BB0B6-6BDC-4CF4-B91C-A3BAF48F7DC9_1_105_c.jpeg')
  bgr_image = cv2.cvtColor(mp_image.numpy_view(), cv2.COLOR_RGB2BGR) # numpy array in opencv format
  

  # Run the result
  face_landmarker_result = landmarker.detect(mp_image)
  

  annotated_image = draw_landmarks_on_image(bgr_image, face_landmarker_result)
  cv2.imshow("window", annotated_image)

plot_face_blendshapes_bar_graph(face_landmarker_result.face_blendshapes[0])
dominant_expression = get_top_expressions(face_landmarker_result)

for expression, landmarks in expressions.items():
  for landmark in landmarks:
    if landmark in dominant_expression:
      print(f"Expression detected: {expression.title()}")
# scores = face_landmarker_result.face_blendshapes[0]

# expressions = get_expressions(scores)
# print(f"Expressions detected: {expressions}")

