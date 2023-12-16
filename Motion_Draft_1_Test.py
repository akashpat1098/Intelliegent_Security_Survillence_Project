# ye h original file of Roi Motion detection System
import cv2
import requests


TELEGRAM_BOT_TOKEN = 'Your Telegram Bot Token'
TELEGRAM_CHAT_ID = 'YOUr Chat ID'


# Function to send a message to the Telegram bot
def send_telegram_message(message):
    send_text = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage?chat_id={TELEGRAM_CHAT_ID}&parse_mode=Markdown&text={message}'
    response = requests.get(send_text)  
    return response.json()

# IP address and port from the IP Webcam app
ip_address = "192.168.35.22"  # Replace with your phone's IP address
port = "8080"
# URL for accessing the video feed
url = f"http://{ip_address}:{port}/video"
window_width = 720
window_height = 480
# Create a VideoCapture object to capture video from a webcam or video file
cap = cv2.VideoCapture(url)  # Use 0 for the default camera, or specify a video file path

# Get the dimensions of the captured frame
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(frame_width,frame_height)

# Define the size of the smaller square area
square_size = 100  # You can adjust the size as needed

# Calculate the coordinates for the smaller square area at the center
x = int(frame_width / 2 - square_size / 2)
y = int(frame_height / 2 - square_size / 2)
small_square_area = (x, y, square_size, square_size)
print(small_square_area)
# Initialize the background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# Set the initial window size
# cv2.namedWindow('Motion Detection', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Motion Detection', 800, 600)  # Adjust the size as needed

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Extract the smaller square area from the frame
    x, y, w, h = small_square_area
    roi = frame[y:y+h, x:x+w]
    # Draw a green outline around the smaller square area
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle top-left corner, bottom-right corner, BGR ,format thickness

    # Apply background subtraction to the smaller square area
    fgmask = fgbg.apply(roi)

    # Threshold the foreground mask to identify motion
    threshold = 50  # Adjust the threshold value as needed
    _, thresh = cv2.threshold(fgmask, threshold, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False

    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Adjust the area threshold as needed
            motion_detected = True

            # Get the coordinates of the detected contour and draw a green rectangle around it
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle

            break

    if motion_detected:
        print("Motion detected in the smaller square area!")
        # send_telegram_message("Rahul Run, Crush is in Balcony❤️😍 ")  # Send Telegram notification

    # Resize and place the smaller square area within the larger window
    frame[y:y+h, x:x+w] = cv2.resize(roi, (w, h))
    resized_frame = cv2.resize(frame, (window_width, window_height))
    cv2.imshow('Motion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'Esc' key to exit
        break

cap.release()
cv2.destroyAllWindows()