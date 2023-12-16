print("Loading Modules...............Please Wait!")
import cv2
import sys
import time
import json 
import datetime
class ObjectDetectionSystem:
    def __init__(self,  config_path='conf.json', video_path=None,isCombined=False):
        self.config_path = config_path
        self.conf = self.load_configuration()
        self.isCombined = isCombined
        self.last_save_time = time.time()
        self.image_counter = self.conf["Object_image_counter"]
        if isCombined == False:
            self.video_path = video_path
            self.camera = self.initialize_camera()
        # self.known_face_encodings, self.known_face_names = self.initialize_known_faces()

    def load_configuration(self):
        with open(self.config_path) as config_file:
            return json.load(config_file)

    def load_model(self):
        print("[OBJECT]Loading Object Detection Model...")
        from super_gradients.training import models  #Comment out when not using model
        from torch.cuda import is_available
        sys.stdout = sys.__stdout__  # Resolve the error of models import
        self.best_model = models.get("yolo_nas_l", pretrained_weights="coco")
        # self.best_model = models.get('yolo_nas_l ', num_classes=len(['Face']), checkpoint_path=self.conf["face_detection_model_path"])
        self.best_model = self.best_model.to("cuda" if is_available() else "cpu")
        print("[OBJECT]Loading Completed")
        
    def process_frame(self, frame):
        # Use your YOLO NAS model for object detection
        images_predictions = self.best_model.predict(frame, conf=0.50)
        isDetection = False
        for image_prediction in images_predictions:
            bboxes = image_prediction.prediction.bboxes_xyxy
            class_ids=image_prediction.prediction.labels.astype(int)
            class_names_list = [image_prediction.class_names[cid] for cid in class_ids]
            object_locations = [(y1, x2, y2, x1) for x1, y1, x2, y2 in bboxes]
            # print(f"[OBJECT]Object Location: {(object_locations)}, {(class_names_list)}")

        return object_locations ,class_names_list

    def display_info(self, frame, object_locations,class_names_list, fps):
        for bbox, name in zip(object_locations, class_names_list):
            top, right, bottom, left = bbox
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 3)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (int(left) + 6, int(bottom) - 6), font, 0.5, (255, 255, 255), 1)
            # if self.isCombined == False:
            #     cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
    

    def calculate_fps(self, prev_time):
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        return fps

    def initialize_camera(self):
        print("[OBJECT]Initializing Camera...")
        if self.video_path:
            camera = cv2.VideoCapture(self.video_path)
        else:
            try:
                camera = cv2.VideoCapture(self.conf["url"])
                if not camera.isOpened():
                    raise Exception("Error: Unable to open video source from URL.")
            except Exception as e:
                print(f"[OBJECT]Error: {e}")
                print("[OBJECT]Falling back to the default camera.")
                camera = cv2.VideoCapture(0)  # 0 represents the default camera (usually the built-in webcam)
        return camera

    def save_image_locally(self, frame):
        local_path = f"ObjectImage/{self.image_counter}.jpg"
        cv2.imwrite(local_path, frame)
        print(f"[SAVED LOCALLY] [OBJECT] {datetime.datetime.now().strftime('%A %d %B %Y %I:%M:%S%p')}")
        self.image_counter+=1

    def willImageBeSaved(self,frame, isDetection):
        if isDetection:
            current_time = time.time()
            if current_time - self.last_save_time >= self.conf["save_interval_seconds"]:
                self.save_image_locally(frame)

                # Update the last save time
                self.last_save_time = current_time

    def run(self):
        prev_time = time.time()
        self.load_model()
        while self.camera.isOpened():
            ret, frame = self.camera.read()
            if not ret:
                self.conf["Object_image_counter"] = self.image_counter
                with open(self.config_path, 'w') as config_file:
                    json.dump(self.conf, config_file, indent=4)
                break
            fps = self.calculate_fps(prev_time)
            object_locations, class_names_list = self.process_frame(frame)
            self.display_info(frame, object_locations,class_names_list, fps)
            self.willImageBeSaved(frame, len(class_names_list))
            if self.conf["show_video"]:
                cv2.imshow('Object Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.conf["Object_image_counter"] = self.image_counter
                    with open(self.config_path, 'w') as config_file:
                        json.dump(self.conf, config_file, indent=4)
                    break
        self.camera.release()
        cv2.destroyAllWindows()

# Usage example
if __name__ == "__main__":
    # Specify the path to your JSON configuration file
    config_path = 'Project\conf.json'
    video_path = None
    face_recognition_system = ObjectDetectionSystem(config_path, video_path)
    face_recognition_system.run()