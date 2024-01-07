import cv2
import datetime
import time
import dropbox
import imutils
import json
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
class MotionDetector:
    def __init__(self ,config_path='conf.json',video_path=None,roi = (260,140,200,200)):
        self.config_path = config_path
        self.conf = self.load_configuration()
        self.video_path = video_path
        self.camera = self.initialize_camera()
        if docenter == True:
            frame_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
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

    def load_configuration(self):
        with open(self.config_path) as config_file:
            return json.load(config_file)

    def initialize_dropbox_client(self):
        dropbox_access_token = self.conf["dropbox_access_token"]
        return dropbox.Dropbox(dropbox_access_token)

    def initialize_camera(self):
        print("[MOTION]Initializing Camera...")
        if self.video_path:
            camera = cv2.VideoCapture(self.video_path)
        else:
            try:
                camera = cv2.VideoCapture(self.conf["url"])
                if not camera.isOpened():
                    raise Exception("Error: Unable to open video source from URL.")
            except Exception as e:
                print(f"[MOTION]Error: {e}")
                print("[MOTION]Falling back to the default camera.")
                camera = cv2.VideoCapture(0)  # 0 represents the default camera (usually the built-in webcam)
        return camera
    @timing_decorator("process_frame") 
    def process_frame(self, frame):
        # frame = imutils.resize(frame, width=400)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.avg is None:
            print("[MOTION] starting background model...")
            self.avg = gray.copy().astype("float")
            return "Unoccupied"

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
            # print("OCCUPIED")
            return "Occupied"

        return "Unoccupied"

    def display_info(self, frame, isOccupied,fps):
        ts = datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p")
        cv2.putText(frame, f"Room Status: {isOccupied}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 53), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

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

if __name__ == "__main__":
    # Specify the path to your JSON configuration file
    config_path = 'Project\conf.json'
    video_path = None 
    # x=0 to 570,y= 0 to 330, w = 0 to 720 and h = 0 to 480 w>h
    docenter= True  # Specify the 
    width = 200
    height = 200

    # Create an instance of the MotionDetector class
    motion_detector = MotionDetector(config_path, video_path)

    # Run the motion detector
    motion_detector.run()
