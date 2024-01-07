# ye h optimization of models on small code with no camera stuff
import sys 
import time
import json 
import cv2

from face_recognition import load_image_file, face_encodings, face_locations, compare_faces
def timing_decorator(func_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"[INFO] Execution Time for {func_name}: {execution_time:.4f} seconds")
            return result
        return wrapper
    return decorator
def load_model():
        print("[FACE]Loading Face Recognition Model...")
        from super_gradients.training import models  #Comment out when not using model
        from torch.cuda import is_available
        sys.stdout = sys.__stdout__  # Resolve the error of models import
        best_model = models.get('yolo_nas_m ', num_classes=len(['Face']), checkpoint_path=conf["face_detection_model_path"])
        best_model = best_model.to("cuda" if is_available() else "cpu")
        print("[FACE]Loading Completed")
        return best_model

@timing_decorator("process_frame") 
def process_frame( frame,useModel,best_model,known_face_encodings, known_face_names):
    if useModel:
        # Use your YOLO NAS model for face detection
        images_predictions = best_model.predict(frame)
        for image_prediction in images_predictions:
            bboxes = image_prediction.prediction.bboxes_xyxy
            face_locations = [(y1, x2, y2, x1) for x1, y1, x2, y2 in bboxes]
            print(f"[FACE]Face Location: {face_locations}")
    else:
        face_locations = detect_faces(frame)
        # print(f"Face Location: {face_locations}")
    recognized_names = recognize_faces(face_encodings(cv2.imread(frame), face_locations),known_face_encodings, known_face_names)
    return face_locations, recognized_names

def detect_faces(frame):
    return face_locations(frame)

def initialize_known_faces():
    known_face_encodings = []
    known_face_names = []
    for name, file_path in conf['known_faces'].items():
        image_of_known_person = load_image_file(file_path)
        known_face_encodings.append(face_encodings(image_of_known_person)[0])
        known_face_names.append(name)
    return known_face_encodings, known_face_names
    
def recognize_faces(face_encodings,known_face_encodings,known_face_names):
    recognized_names = []
    for face_encoding in face_encodings:
        matches = compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        recognized_names.append(name)
    return recognized_names
    
def load_configuration():
    with open(config_path) as config_file:
        return json.load(config_file)
if __name__ == "__main__":
    config_path = 'Project/conf.json'
    useModel = True
    images = ["FaceImage/MyPhoto.jpg","FaceImage/MyPhoto.jpg","FaceImage/MyPhoto.jpg","FaceImage/MyPhoto.jpg","FaceImage/MyPhoto.jpg","FaceImage/MyPhoto.jpg"]
    conf = load_configuration()
    known_face_encodings, known_face_names = initialize_known_faces()
    if useModel:
        best_model = load_model()
    for frame in images:
        face_locations, recognized_names = process_frame(frame,useModel,best_model,known_face_encodings, known_face_names)
        print(recognized_names)

