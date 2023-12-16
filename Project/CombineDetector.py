import cv2
import json
import time
from FaceDetection import FaceRecognitionSystem
from MotionDetection import MotionDetector
from ObjectDetection import ObjectDetectionSystem

class CombinedSystem:
    def __init__(self, roi, config_path='conf.json', video_path=None,use_FaceDetector=True,use_MotionDetector=True,use_ObjectDetector=True):
        self.config_path = config_path
        self.video_path = video_path
        self.conf = self.load_configuration()
        self.camera = self.initialize_camera()
        self.isCombined=True
        self.use_FaceDetector=use_FaceDetector
        self.use_MotionDetector=use_MotionDetector
        self.use_ObjectDetector=use_ObjectDetector
        if self.use_FaceDetector:
            self.face_system = self.initialize_face_system(config_path, video_path)
        if self.use_ObjectDetector:
            self.object_system = self.initialize_object_system(config_path, video_path)
        if self.use_MotionDetector:
            self.motion_system = self.initialize_motion_system(roi,config_path, video_path)
    def load_configuration(self):
        with open(self.config_path) as config_file:
            return json.load(config_file)

    def initialize_camera(self):
        print("Initializing Camera...")
        if self.video_path:
            camera = cv2.VideoCapture(self.video_path)
        else:
            try:
                camera = cv2.VideoCapture(self.conf["url"])
                if not camera.isOpened():
                    raise Exception("Error: Unable to open video source from URL.")
            except Exception as e:
                print(f"Error: {e}")
                print("Falling back to the default camera.")
                camera = cv2.VideoCapture(0)  # 0 represents the default camera (usually the built-in webcam)
        return camera
    def initialize_face_system(self, config_path, video_path):
        face_recognition_system = FaceRecognitionSystem(config_path, video_path,isCombined=self.isCombined)
        return face_recognition_system

    def initialize_object_system(self, config_path, video_path):
        face_recognition_system = ObjectDetectionSystem(config_path, video_path,isCombined=self.isCombined)
        return face_recognition_system
    
    def initialize_motion_system(self,roi, config_path, video_path):
        if roi is None:
            roi = (260,140,200,200)
        motion_detector = MotionDetector(config_path, video_path,roi=roi,isCombined=self.isCombined)
        return motion_detector
    
    def calculate_fps(self, prev_time):
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        return fps
    def run_MotionDetection(self,frame,fps):
        roi = self.motion_system.extract_small_square_area(frame)
        self.motion_system.draw_square_on_frame(frame)
        isOccupied = self.motion_system.process_frame(roi)
        self.motion_system.display_info(frame, isOccupied,fps)
        self.motion_system.willImageBeSaved(frame, isOccupied)
    
    def run_FaceDetection(self,frame,fps):
        face_locations, recognized_names = self.face_system.process_frame(frame,useModel)
        self.face_system.display_info(frame, face_locations, recognized_names, fps)
        self.face_system.willImageBeSaved(frame, recognized_names)

    def run_ObjectDetection(self,frame,fps):
        object_locations, class_names_list = self.object_system.process_frame(frame)
        self.object_system.display_info(frame, object_locations,class_names_list, fps)
        self.object_system.willImageBeSaved(frame, len(class_names_list))

    def run_combined_system(self):
        # Motion Detection
        print("[INFO] Warming up...")
        time.sleep(self.conf["camera_warmup_time"])

        # Face Detection
        if self.use_FaceDetector and useModel:
            self.use_FaceDetector.load_model()
        # Motion Detection
        if self.use_ObjectDetector:
            self.object_system.load_model()

        # Common
        prev_time = time.time()
        while self.camera.isOpened():
            ret, frame = self.camera.read()
            if not ret:
                break
            fps = self.calculate_fps(prev_time)

            # Motion Detection
            if self.use_MotionDetector:
                self.run_MotionDetection(frame,fps)
            # Face Detection and Recognition
            if self.use_FaceDetector:
                self.run_FaceDetection(frame,fps)
        
            if self.use_ObjectDetector:
                self.run_ObjectDetection(frame,fps)

            if self.conf["show_video"]:
                cv2.imshow("Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    if self.use_FaceDetector:
                        self.conf["Face_image_counter"] = self.face_system.image_counter
                    if self.use_MotionDetector:
                        self.conf["Motion_image_counter"] = self.motion_system.image_counter
                    if self.use_ObjectDetector:
                        self.conf["Object_image_counter"] = self.object_system.image_counter
                    with open(self.config_path, 'w') as config_file:
                        json.dump(self.conf, config_file, indent=4)
                    break

        self.camera.release()
        cv2.destroyAllWindows()

# Usage example
if __name__ == "__main__":
    # Specify the path to your JSON configuration file
    config_path = "Project\conf.json"
    useModel = False
    video_path = 'akashVideo.mp4'
    use_FaceDetector = True
    use_MotionDetector = True
    use_ObjectDetector = True
    # x=0 to 570,y= 0 to 330, w = 0 to 720 and h = 0 to 480 w>h
    # roi = (260,140,500,450)
    roi = None
    combined_system = CombinedSystem(roi,config_path, video_path,use_FaceDetector,use_MotionDetector,use_ObjectDetector)
    combined_system.run_combined_system()
