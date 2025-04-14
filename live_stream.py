import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2 as cv
import time
import numpy as np


model_path = "/Users/glenchua/Documents/thesis_project/blaze_face_short_range.tflite"

latest_frame = None

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
FaceDetectorResult = mp.tasks.vision.FaceDetectorResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a face detector instance with the live stream mode:
def print_result(result: FaceDetectorResult, output_image: mp.Image, timestamp_ms: int): # type: ignore
    print('face detector result: {}'.format(result))  

def on_face_detected(result, output_image, timestamp_ms):
    global latest_frame
    latest_frame = output_image.numpy_view().copy()
    for detection in result.detections:
        bbox = detection.bounding_box   
        x, y = int(bbox.origin_x), int(bbox.origin_y)
        w, h = int(bbox.width), int(bbox.height)

        cv.rectangle(latest_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=on_face_detected)



if __name__ == "__main__":
    with FaceDetector.create_from_options(options) as detector:
        start_time = time.time()
        cap = cv.VideoCapture(0)

        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        while True:
        # Capture frame-by-frame
            ret, frame = cap.read()
            color = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=color) # Convert the frame received from OpenCV to a MediaPipe’s Image object.
            frame_timestamp_ms = int((time.time() - start_time) * 1000)

            # Detect faces
            detector.detect_async(mp_image, frame_timestamp_ms)

            # Display the frame
            if latest_frame is not None:
                cv.imshow("window", cv.cvtColor(latest_frame, cv.COLOR_RGB2BGR))

    
        # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            if cv.waitKey(1) == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()