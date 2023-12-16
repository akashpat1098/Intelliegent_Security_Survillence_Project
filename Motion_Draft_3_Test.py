# ye testing h inttegration of ROI in Motion Detection
import cv2
import requests

class MotionDetection:
    def __init__(self, ip_address, port, telegram_bot_token, telegram_chat_id):
        self.ip_address = ip_address
        self.port = port
        self.url = f"http://{ip_address}:{port}/video"
        self.cap = cv2.VideoCapture(self.url)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(self.frame_width, self.frame_height)
        self.square_size = 150
        # x=0 to 570,y= 0 to 330
        # self.x, self.y = (570,330)
        self.x, self.y = self.get_user_coordinates()
        self.small_square_area = (self.x, self.y, self.square_size, self.square_size)
        print(f"small_square_area: {self.small_square_area}")
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        self.telegram_bot_token = telegram_bot_token
        self.telegram_chat_id = telegram_chat_id

    def get_user_coordinates(self):
        x= int(self.frame_width / 2 - self.square_size / 2)
        y = int(self.frame_height / 2 - self.square_size / 2)
        print(x,y)
        return x, y 

    def send_telegram_message(self, message):
        send_text = f'https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage?chat_id={self.telegram_chat_id}&parse_mode=Markdown&text={message}'
        response = requests.get(send_text)
        return response.json()

    def extract_small_square_area(self, frame):
        x, y, w, h = self.small_square_area
        contour_xywh = x, y, w, h 
        return frame[y:y+h, x:x+w] , contour_xywh

    def draw_square_on_frame(self, frame):
        x, y, w, h = self.small_square_area
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    def apply_background_subtraction(self, roi):
        return self.fgbg.apply(roi)

    def threshold_foreground_mask(self, fgmask):
        threshold = 50
        return cv2.threshold(fgmask, threshold, 255, cv2.THRESH_BINARY)

    def find_contours(self, thresh):
        return cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    def check_for_motion(self, contours, roi ,contour_xywh):
        motion_detected = False

        for contour in contours:
            if cv2.contourArea(contour) > 100:
                motion_detected = True
                contour_xywh = cv2.boundingRect(contour)
                x, y, w, h = contour_xywh
                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
                break

        return motion_detected , contour_xywh

    def display_frame(self, frame,roi,contour_xywh):
        x, y, w, h = contour_xywh
        frame[y:y+h, x:x+w] = cv2.resize(roi, (w, h))
        # resized_frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        cv2.imshow('Motion Detection', frame)

    def detect_motion(self):
        while True:
            ret, frame = self.cap.read()

            if not ret:
                break

            roi, contour_xywh = self.extract_small_square_area(frame)
            self.draw_square_on_frame(frame)

            fgmask = self.apply_background_subtraction(roi)
            _, thresh = self.threshold_foreground_mask(fgmask)

            contours, _ = self.find_contours(thresh)
    
            motion_detected ,contour_xywh= self.check_for_motion(contours, roi ,contour_xywh)

            if motion_detected:
                print("Motion detected in the smaller square area!")
                # self.send_telegram_message("Motion detected!")  # Uncomment to send Telegram notification

            self.display_frame(frame,roi,contour_xywh)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

# Replace 'Your Telegram Bot Token' and 'Your Chat ID' with your actual Telegram Bot Token and Chat ID
motion_detector = MotionDetection("192.168.35.22", "8080", 'Your Telegram Bot Token', 'Your Chat ID')
motion_detector.detect_motion()
