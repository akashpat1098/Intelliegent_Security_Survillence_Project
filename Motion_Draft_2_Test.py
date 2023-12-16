# Ye h testing integration of Object oriented approach
import argparse
import warnings
import datetime
import dropbox
import imutils
import json
import time
import cv2

ip_address = "192.168.35.22"  # Replace with your phone's IP address
port = "8080"
url = f"http://{ip_address}:{port}/video"
video_path = None
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, 
	help="path to the JSON configuration file")
args = vars(ap.parse_args())
# filter warnings, load the configuration and initialize the Dropbox
# client
warnings.filterwarnings("ignore")
conf = json.load(open(args["conf"]))
client = None


# check to see if the Dropbox should be used
if conf["use_dropbox"]:
	# connect to dropbox and start the session authorization process
	client = dropbox.Dropbox(conf["dropbox_access_token"])
	print("[SUCCESS] dropbox account linked")
	
# initialize the camera
print("Initializing Camera...")
if video_path:
    camera = cv2.VideoCapture(video_path)
else:
    try:
        camera = cv2.VideoCapture(conf["url"])
        if not camera.isOpened():
            raise Exception("Error: Unable to open video source from URL.")
    except Exception as e:
        print(f"Error: {e}")
        print("Falling back to the default camera.")
        camera = cv2.VideoCapture(0)  # 0 represents the default camera (usually the built-in webcam)


print("[INFO] warming up...")
time.sleep(conf["camera_warmup_time"])
avg = None
lastUploaded = datetime.datetime.now()
motionCounter = 0
image_counter = 0



# capture frames from the camera
while True:

    grabbed, frame = camera.read()
    timestamp = datetime.datetime.now()
    text = "Unoccupied"
    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    # if the average frame is None, initialize it
    if avg is None:
        print("[INFO] starting background model...")
        avg = gray.copy().astype("float")
        continue
    # accumulate the weighted average between the current frame and
    # previous frames, then compute the difference between the current
    # frame and running average
    cv2.accumulateWeighted(gray, avg, 0.5)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
    # threshold the delta image, dilate the thresholded image to fill
    # in holes, then find contours on thresholded image
    thresh = cv2.threshold(frameDelta, conf["delta_thresh"], 255,
                           cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # loop over the contours
    i=0
    for c in cnts:
        print(i)
        i+=1
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < conf["min_area"]:
            continue
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Occupied"
        print("OCCUPIED")
        # draw the text and timestamp on the frame
    ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
    cv2.putText(frame, "Room Status: {}".format(text), (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (0, 0, 255), 1)
		
        
        # check to see if the room is occupied
    if text == "Occupied":
		# check to see if enough time has passed between uploads
        if (timestamp - lastUploaded).seconds >= conf["min_upload_seconds"]:
            # increment the motion counter
            motionCounter += 1
            # check to see if the number of frames with consistent motion is
            # high enough
            if motionCounter >= conf["min_motion_frames"]:
                print("reached===================")
                # Save the image locally
                local_path = f"trial/{image_counter}.jpg"  # Specify your local path
                image_counter+=1
                cv2.imwrite(local_path, frame)
                print("[SAVE LOCALLY] {}".format(ts))
                # Update the last uploaded timestamp and reset the motion counter
                lastUploaded = timestamp
                motionCounter = 0

	# otherwise, the room is not occupied
    else:
        motionCounter = 0
		

	# check to see if the frames should be displayed to screen
    if conf["show_video"]:
		# display the security feed
        # resized_frame = cv2.resize(frame, (window_width, window_height))
        cv2.imshow("Security Feed", frame)
        key = cv2.waitKey(1) & 0xFF
		# if the `q` key is pressed, break from the lop
        if key == ord("q"):
            break
