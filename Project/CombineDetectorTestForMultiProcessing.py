import cv2
import json
import time
from multiprocessing import Process, Manager

from face_recognition import load_image_file, face_encodings, face_locations, compare_faces
import sys
import datetime
import dropbox
import imutils
# def timing_decorator(func_name):
#     def decorator(func):
#         def wrapper(*args, **kwargs):
#             start_time = time.time()
#             result = func(*args, **kwargs)
#             end_time = time.time()
#             execution_time = end_time - start_time
#             print(f"[INFO] Execution Time for {func_name}: {execution_time:.4f} seconds")
#             return result
#         return wrapper
#     return decorator

class FaceRecognitionSystem:
    def __init__(self,conf):
        self.conf = conf
        self.last_save_time = time.time()
        self.image_counter = self.conf["Face_image_counter"]
        self.known_face_encodings, self.known_face_names = self.initialize_known_faces()

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

    # @timing_decorator("[FACE]process_frame") 
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
        return frame,face_locations, recognized_names

    def display_info(self, frame, face_locations, recognized_names, fps):
        for bbox, name in zip(face_locations, recognized_names):
            top, right, bottom, left = bbox
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 1)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (int(left) + 6, int(bottom) - 6), font, 0.5, (255, 255, 255), 1)
            # cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
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

class ObjectDetectionSystem:
    def __init__(self,conf):
        self.conf = conf
        self.last_save_time = time.time()
        self.image_counter = self.conf["Object_image_counter"]

    def load_model(self):
        print("[OBJECT]Loading Object Detection Model...")
        from super_gradients.training import models  #Comment out when not using model
        from torch.cuda import is_available
        sys.stdout = sys.__stdout__  # Resolve the error of models import
        self.best_model = models.get("yolo_nas_l", pretrained_weights="coco")
        # self.best_model = models.get('yolo_nas_l ', num_classes=len(['Face']), checkpoint_path=self.conf["face_detection_model_path"])
        self.best_model = self.best_model.to("cuda" if is_available() else "cpu")
        print("[OBJECT]Loading Completed")

    # @timing_decorator("[OBJECT]process_frame") 
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
        return frame,object_locations ,class_names_list
    
    def display_info(self, frame, object_locations,class_names_list, fps):
        for bbox, name in zip(object_locations, class_names_list):
            top, right, bottom, left = bbox
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 1)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (int(left) + 6, int(bottom) - 6), font, 0.5, (255, 255, 255), 1)
            # cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
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
        frame_count = 0
        while self.camera.isOpened():
            ret, frame = self.camera.read()
            if not ret:
                self.conf["Object_image_counter"] = self.image_counter
                with open(self.config_path, 'w') as config_file:
                    json.dump(self.conf, config_file, indent=4)
                break
            frame_count += 1
            fps = self.calculate_fps(prev_time)
            if (frame_count % self.conf["skip_frames"] == 0):   
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

class MotionDetector:
    def __init__(self ,conf,camera,roi = (260,140,200,200)):
        self.conf = conf
        if docenter == True:
            frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            x = int(frame_width / 2 - width / 2)
            y = int(frame_height / 2 - height / 2)
            roi = (x,y,width,height)
        self.small_square_area = roi
        print(f"[MOTION]ROI:{roi}")
        self.client = self.initialize_dropbox_client() if self.conf["use_dropbox"] else None
        self.avg = None
        self.last_uploaded = datetime.datetime.now()
        self.image_counter = self.conf["Motion_image_counter"]
        self.motion_counter = 0

    def initialize_dropbox_client(self):
        dropbox_access_token = self.conf["dropbox_access_token"]
        return dropbox.Dropbox(dropbox_access_token)

    # @timing_decorator("[MOTION]process_frame") 
    def process_frame(self, frame):
        # frame = imutils.resize(frame, width=400)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.avg is None:
            print("[MOTION] starting background model...")
            self.avg = gray.copy().astype("float")
            return (frame,"Unoccupied")

        cv2.accumulateWeighted(gray, self.avg, 0.5)
        frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(self.avg))
        thresh = cv2.threshold(frame_delta, self.conf["delta_thresh"], 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        for c in cnts:
            if cv2.contourArea(c) < self.conf["min_area"]:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            # print(f"Motion Location: {(x, y, w, h)}")
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            return (frame,"Occupied")

        return (frame,"Unoccupied")

    def display_info(self, frame, isOccupied,fps):
        ts = datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p")
        cv2.putText(frame, f"Room Status: {isOccupied}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        # cv2.putText(frame, f"FPS: {fps:.2f}", (10, 53), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    def save_image_locally(self, frame):
        local_path = f"MotionImage/{self.image_counter}.jpg"
        cv2.imwrite(local_path, frame)
        print(f"[SAVED LOCALLY] [MOTION] {datetime.datetime.now().strftime('%A %d %B %Y %I:%M:%S%p')}")
        self.image_counter+=1

    def calculate_fps(self, prev_time):
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        return fps
    def willImageBeSaved(self,frame, isOccupied):
        if isOccupied == "Occupied":
            if (datetime.datetime.now() - self.last_uploaded).seconds >= self.conf["min_upload_seconds"]:
                self.motion_counter += 1
                if self.motion_counter >= self.conf["min_motion_frames"]:
                    self.save_image_locally(frame)
                    self.last_uploaded = datetime.datetime.now()
                    self.motion_counter = 0 

        else:
            self.motion_counter = 0
    def extract_small_square_area(self, frame):
        x, y, w, h = self.small_square_area
        return frame[y:y+h, x:x+w]

    def draw_square_on_frame(self, frame):
        x, y, w, h = self.small_square_area
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    def run(self):
        print("[MOTION] Warming up...")
        time.sleep(self.conf["camera_warmup_time"])

        prev_time = time.time()
        while self.camera.isOpened():
            ret, frame = self.camera.read()
            if not ret:
                self.conf["Motion_image_counter"] = self.image_counter
                with open(self.config_path, 'w') as config_file:
                    json.dump(self.conf, config_file, indent=4)
                break
            fps = self.calculate_fps(prev_time)

            roi = self.extract_small_square_area(frame)
            self.draw_square_on_frame(frame)

            isOccupied = self.process_frame(roi)
            self.display_info(frame, isOccupied,fps)
            self.willImageBeSaved(frame, isOccupied)        
            if self.conf["show_video"]:
                cv2.imshow("Security Feed", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.conf["Motion_image_counter"] = self.image_counter
                    with open(self.config_path, 'w') as config_file:
                        json.dump(self.conf, config_file, indent=4)
                    break
        self.camera.release()
        cv2.destroyAllWindows()

class CombinedSystem:
    def __init__(self, roi, config_path='conf.json', video_path=None  ,use_FaceDetector=True,use_MotionDetector=True,use_ObjectDetector=True):
        self.config_path = config_path
        self.video_path = video_path
        self.conf = self.load_configuration()
        self.camera = self.initialize_camera()
        self.use_FaceDetector=use_FaceDetector
        self.use_MotionDetector=use_MotionDetector
        self.use_ObjectDetector=use_ObjectDetector
        if self.use_FaceDetector:
            self.face_system = self.initialize_face_system(self.conf)
        if self.use_ObjectDetector:
            self.object_system = self.initialize_object_system(self.conf)
        if self.use_MotionDetector:
            self.motion_system = self.initialize_motion_system(self.conf,self.camera,roi)
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
    def initialize_face_system(self,conf):
        face_detector = FaceRecognitionSystem(conf)
        return face_detector

    def initialize_object_system(self,conf):
        object_detector = ObjectDetectionSystem(conf)
        return object_detector
    
    def initialize_motion_system(self,conf,camera,roi):
        if roi is None:
            roi = (260,140,200,200)
        motion_detector = MotionDetector(conf,camera,roi=roi)
        return motion_detector
    
    def calculate_fps(self, prev_time):
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        return fps
    
    def run_MotionDetection(self,frame_queue,output_queue):
        while True:
            frame,prev_time= frame_queue.get()
            if frame is None:
                output_queue.put((None,None,None))
                break
            roi = self.motion_system.extract_small_square_area(frame)
            self.motion_system.draw_square_on_frame(frame)
            frame,isOccupied = self.motion_system.process_frame(roi)
            output_queue.put((frame, isOccupied,prev_time))
    
    def run_FaceDetection(self,frame_queue,output_queue):
        frame_count = 0
        while True:
            frame,prev_time = frame_queue.get()
            if frame is None:
                output_queue.put((None,None,None))
                break
            frame_count += 1
            if (frame_count % self.conf["skip_frames"] == 0):
                frame,face_locations, recognized_names = self.face_system.process_frame(frame,useModel)
                output_queue.put((frame,face_locations, recognized_names,prev_time))


    def run_ObjectDetection(self,frame_queue,output_queue):
        frame_count = 0
        while True:
            frame,prev_time = frame_queue.get()
            if frame is None:
                output_queue.put((None,None,None))
                break
            frame_count += 1
            if (frame_count % self.conf["skip_frames"] == 0):
                frame,object_locations, class_names_list = self.object_system.process_frame(frame)
                output_queue.put((frame,object_locations, class_names_list,prev_time))

    def capture_video(self,frame_queue, terminate_flag):
        prev_time = time.time()
        while not terminate_flag.is_set():
            ret, frame = self.camera.read()
            if not ret:
                if self.use_FaceDetector:
                    self.conf["Face_image_counter"] = self.face_system.image_counter
                if self.use_ObjectDetector:
                    self.conf["Object_image_counter"] = self.object_system.image_counter
                if self.use_MotionDetector:
                    self.conf["Motion_image_counter"] = self.motion_system.image_counter
                with open(self.config_path, 'w') as config_file:
                    json.dump(self.conf, config_file, indent=4)
                break
            frame_queue.put(frame,prev_time)
         # Signal the termination to frame processing functions
        frame_queue.put(None)
        frame_queue.put(None)
        self.camera.release()
    def display_frames(self,face_output_queue, object_output_queue, motion_output_queue,terminate_flag):
        while not terminate_flag.is_set():
            # Get the original frames and bounding box information from the output queues
            face_frame,face_locations, recognized_names,prev_time = face_output_queue.get()
            object_frame,object_locations, class_names_list,prev_time = object_output_queue.get()
            motion_frame, isOccupied,prev_time = motion_output_queue.get()

            if motion_frame is None or object_locations is None or face_locations is None:
                break  # Break the loop if termination signal is received
            fps = self.calculate_fps(prev_time)

            if self.use_MotionDetector:
                self.motion_system.display_info(motion_frame, isOccupied,fps)
                self.motion_system.willImageBeSaved(motion_frame, isOccupied)

            if self.use_FaceDetector:
                self.face_system.display_info(face_frame, face_locations, recognized_names, fps)
                self.face_system.willImageBeSaved(face_frame, recognized_names)

            if self.use_ObjectDetector:
                self.object_system.display_info(object_frame, object_locations,class_names_list, fps)
                self.object_system.willImageBeSaved(object_frame, len(class_names_list))

            if self.conf["show_video"]:
                cv2.putText(object_frame, f"FPS: {fps:.2f}", (10, 53), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Detection", object_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    if self.use_FaceDetector:
                        self.conf["Face_image_counter"] = self.face_system.image_counter
                        # face_process.terminate()
                    if self.use_MotionDetector:
                        self.conf["Motion_image_counter"] = self.motion_system.image_counter
                        # motion_process.terminate()
                    if self.use_ObjectDetector:
                        self.conf["Object_image_counter"] = self.object_system.image_counter
                        # object_process.terminate()
                    with open(self.config_path, 'w') as config_file:
                        json.dump(self.conf, config_file, indent=4)
                    terminate_flag.set()
        cv2.destroyAllWindows()

    def run_combined_system(self):
        pass


# Usage example
if __name__ == "__main__":
    with Manager() as manager:
        # Specify the path to your JSON configuration file
        config_path = "Project\conf.json"
        useModel = False
        video_path = None
        docenter= True
        width = 200
        height = 200
        # video_path = "static/vvvv.mp4"
        use_FaceDetector = True
        use_MotionDetector = True
        use_ObjectDetector = True
        # x=0 to 570,y= 0 to 330, w = 0 to 720 and h = 0 to 480 w>h
        # roi = (260,140,500,450)
        roi = None
        combined_system = CombinedSystem(roi,config_path, video_path,use_FaceDetector,use_MotionDetector,use_ObjectDetector)
        # combined_system.run_combined_system()

        # Motion Detection
        print("[INFO] Warming up...")
        time.sleep(combined_system.conf["camera_warmup_time"])

        # Face Detection
        if combined_system.use_FaceDetector and useModel:
            combined_system.face_system.load_model()
        # Motion Detection
        if combined_system.use_ObjectDetector:
            combined_system.object_system.load_model()

        frame_queue = manager.Queue()
        face_output_queue = manager.Queue()
        object_output_queue = manager.Queue()
        motion_output_queue = manager.Queue()
        terminate_flag = manager.Event()

        # Start the video capture process
        video_process = Process(target=combined_system.capture_video, args=(frame_queue, terminate_flag))
        video_process.start()

        # Start the frame processing processes
        if combined_system.use_FaceDetector:
            face_process = Process(target=combined_system.run_FaceDetection, args=(frame_queue, face_output_queue))
            face_process.start()

        if combined_system.use_ObjectDetector:
            object_process = Process(target=combined_system.run_ObjectDetection, args=(frame_queue, object_output_queue))
            object_process.start()
        if combined_system.use_MotionDetector:
            motion_process = Process(target=combined_system.run_MotionDetection, args=(frame_queue, motion_output_queue))
            motion_process.start()

        # Start the display process as daemon
        display_process = Process(target=combined_system.display_frames, args=(face_output_queue, object_output_queue, motion_output_queue,terminate_flag))
        display_process.daemon = True
        display_process.start()

        # Wait for the user to press 'q' to terminate the program
        while not terminate_flag.is_set():
            time.sleep(1)

        # Join the video process (not strictly necessary due to the daemon flag)
        video_process.join()

        face_process.join()
        object_process.join()
        motion_process.join()
