import cv2
import time
from multiprocessing import Process, Manager
import numpy as np
import random

def generate_random_bounding_boxes(num_boxes, image_width, image_height):
    bounding_boxes = []

    for _ in range(num_boxes):
        x1 = random.randint(0, image_width - 1)
        y1 = random.randint(0, image_height - 1)
        x2 = random.randint(x1 + 1, image_width)
        y2 = random.randint(y1 + 1, image_height)

        bounding_boxes.append((x1, y1, x2, y2))

    return bounding_boxes

def process_frame1(frame_queue, output_queue):
    while True:
        frame = frame_queue.get()
        if frame is None:
            break

        # No processing, just generate random bounding boxes
        bounding_boxes = generate_random_bounding_boxes(num_boxes=2, image_width=frame.shape[1], image_height=frame.shape[0])

        # Put the original frame and bounding boxes into the output queue 
        output_queue.put((frame, bounding_boxes))

def process_frame2(frame_queue, output_queue):
    while True:
        frame = frame_queue.get()
        if frame is None:
            break

        # No processing, just generate random bounding boxes
        bounding_boxes = generate_random_bounding_boxes(num_boxes=2, image_width=frame.shape[1], image_height=frame.shape[0])

        # Put the original frame and bounding boxes into the output queue
        output_queue.put((frame, bounding_boxes))

def display_frames(output_queue1, output_queue2, terminate_flag):
    start_time = time.time()
    frame_count = 0

    while not terminate_flag.is_set():
        # Get the original frames and bounding box information from the output queues
        frame1, bounding_boxes1 = output_queue1.get()
        frame2, bounding_boxes2 = output_queue2.get()

        if frame1 is None or frame2 is None:
            break  # Break the loop if termination signal is received

        # Copy frames to avoid modifying the original frames
        display_frame = frame1.copy()

        # Overlay bounding boxes on the frames with different colors
        for bbox in bounding_boxes1:
            cv2.rectangle(frame1, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)  # Green color

        for bbox in bounding_boxes2:
            cv2.rectangle(frame2, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)  # Blue color

        # Display the frame with overlaid bounding boxes
        cv2.imshow("Processed Frames", np.hstack((frame1, frame2)))

        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time

        # Display the FPS
        print(f"FPS: {fps:.2f}")

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            terminate_flag.set()

    cv2.destroyAllWindows()

def video_capture_process(frame_queue, terminate_flag):
    cap = cv2.VideoCapture(0)  # Open the default camera

    while not terminate_flag.is_set():
        ret, frame = cap.read()

        if not ret:
            break

        # Put the original frame into the input queue for processing
        frame_queue.put(frame)

    # Signal the termination to frame processing functions
    frame_queue.put(None)
    frame_queue.put(None)

    cap.release()

if __name__ == "__main__":
    with Manager() as manager:
        frame_queue = manager.Queue()
        output_queue1 = manager.Queue()
        output_queue2 = manager.Queue()
        terminate_flag = manager.Event()

        # Start the video capture process
        video_process = Process(target=video_capture_process, args=(frame_queue, terminate_flag))
        video_process.start()

        # Start the frame processing processes
        p1 = Process(target=process_frame1, args=(frame_queue, output_queue1))
        p2 = Process(target=process_frame2, args=(frame_queue, output_queue2))
        p1.start()
        p2.start()

        # Start the display process as daemon
        display_process = Process(target=display_frames, args=(output_queue1, output_queue2, terminate_flag))
        display_process.daemon = True
        display_process.start()

        # Wait for the user to press 'q' to terminate the program
        while not terminate_flag.is_set():
            time.sleep(1)

        # Join the video process (not strictly necessary due to the daemon flag)
        video_process.join()

        # Wait for the frame processing processes to finish
        p1.join()
        p2.join()








    
