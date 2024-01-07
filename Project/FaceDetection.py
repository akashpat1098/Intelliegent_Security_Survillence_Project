print("Loading Modules...............Please Wait!")
import cv2
from face_recognition import load_image_file, face_encodings, face_locations, compare_faces
import sys
import time
import json 
import datetime
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
class FaceRecognitionSystem:
    def __init__(self,  config_path='conf.json', video_path=None):
        self.config_path = config_path
        self.conf = self.load_configuration()
        self.last_save_time = time.time()
        self.image_counter = self.conf["Face_image_counter"]
        self.video_path = video_path
        self.camera = self.initialize_camera()
        self.known_face_encodings, self.known_face_names = self.initialize_known_faces()

    def load_configuration(self):
        with open(self.config_path) as config_file:
            return json.load(config_file)

    def initialize_known_faces(self):
        known_face_encodings = []
        known_face_names = []
        for name, file_path in self.conf['known_faces'].items():
            image_of_known_person = load_image_file(file_path)
            known_face_encodings.append(face_encodings(image_of_known_person)[0])
            known_face_names.append(name)
        return known_face_encodings, known_face_names
    
    def load_model(self):
        print("[FACE]Loading Face Recognition Model...")
        from super_gradients.training import models  #Comment out when not using model
        from torch.cuda import is_available
        sys.stdout = sys.__stdout__  # Resolve the error of models import
        self.best_model = models.get('yolo_nas_m ', num_classes=len(['faces','face']), checkpoint_path=self.conf["face_detection_model_path"])
        self.best_model = self.best_model.to("cuda" if is_available() else "cpu")
        print("[FACE]Loading Completed")
    @timing_decorator("process_frame") 
    def process_frame(self, frame,useModel):
        if useModel:
            # Use your YOLO NAS model for face detection
            images_predictions = self.best_model.predict(frame,conf=self.conf['conf'])
            for image_prediction in images_predictions:
                bboxes = image_prediction.prediction.bboxes_xyxy
                face_locations = [(y1, x2, y2, x1) for x1, y1, x2, y2 in bboxes]
                print(f"[FACE]Face Location: {face_locations}") 
        else:
            face_locations = self.detect_faces(frame)
            # print(f"Face Location: {face_locations}")
        recognized_names = self.recognize_faces(face_encodings(frame, face_locations))
        return face_locations, recognized_names

    def display_info(self, frame, face_locations, recognized_names, fps):
        for bbox, name in zip(face_locations, recognized_names):
            top, right, bottom, left = bbox
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 1)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (int(left) + 6, int(bottom) - 6), font, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
    def detect_faces(self, frame):
        return face_locations(frame)
    
    def recognize_faces(self, face_encodings):
        recognized_names = []
        for face_encoding in face_encodings:
            matches = compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]
            recognized_names.append(name)
        return recognized_names

    def calculate_fps(self, prev_time):
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        return fps

    def initialize_camera(self):
        print("[FACE]Initializing Camera...")
        if self.video_path:
            camera = cv2.VideoCapture(self.video_path)
        else:
            try:
                camera = cv2.VideoCapture(self.conf["url"])
                if not camera.isOpened():
                    raise Exception("Error: Unable to open video source from URL.")
            except Exception as e:
                print(f"[FACE]Error: {e}")
                print("[FACE]Falling back to the default camera.")
                camera = cv2.VideoCapture(0)  # 0 represents the default camera (usually the built-in webcam)
        return camera

    def save_image_locally(self, frame):
        local_path = f"FaceImage/{self.image_counter}.jpg"
        cv2.imwrite(local_path, frame)
        print(f"[SAVED LOCALLY] [FACE] {datetime.datetime.now().strftime('%A %d %B %Y %I:%M:%S%p')}")
        self.image_counter+=1

    def willImageBeSaved(self,frame, recognized_names):
        if any(name != 'Unknown' for name in recognized_names):
            current_time = time.time()
            if current_time - self.last_save_time >= self.conf["save_interval_seconds"]:
                self.save_image_locally(frame)

                # Update the last save time
                self.last_save_time = current_time

    def run(self,useModel = False):
        if useModel:
            self.load_model()
        frame_count = 0
        prev_time = time.time()
        while self.camera.isOpened():
            ret, frame = self.camera.read()
            if not ret:
                self.conf["Face_image_counter"] = self.image_counter
                with open(self.config_path, 'w') as config_file:
                    json.dump(self.conf, config_file, indent=4)
                break
            frame_count += 1
            fps = self.calculate_fps(prev_time)
            if (frame_count % self.conf["skip_frames"] == 0):
                face_locations, recognized_names = self.process_frame(frame,useModel)
                self.display_info(frame, face_locations, recognized_names, fps)
                self.willImageBeSaved(frame, recognized_names)
            if self.conf["show_video"]:
                cv2.imshow('Face Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.conf["Face_image_counter"] = self.image_counter
                    with open(self.config_path, 'w') as config_file:
                        json.dump(self.conf, config_file, indent=4)
                    break
        self.camera.release()
        cv2.destroyAllWindows()

# Usage example
if __name__ == "__main__":
    # Specify the path to your JSON configuration file
    config_path = 'Project/conf.json'
    useModel = True
    # video_path = "static/vvvv.mp4"
    video_path = None
    face_recognition_system = FaceRecognitionSystem(config_path, video_path)
    face_recognition_system.run(useModel)


