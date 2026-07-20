import multiprocessing as mp
import time
import cv2


def _clamp(value, low, high):
    return max(low, min(high, value))


def _scale_point_to_screen(x, y, source_width, source_height, screen_width, screen_height):
    if source_width <= 0 or source_height <= 0:
        return screen_width // 2, screen_height // 2

    margin = 20
    scaled_x = int(_clamp(x / source_width, 0.0, 1.0) * screen_width)
    scaled_y = int(_clamp(y / source_height, 0.0, 1.0) * screen_height)
    scaled_x = _clamp(scaled_x, margin, max(margin, screen_width - margin))
    scaled_y = _clamp(scaled_y, margin, max(margin, screen_height - margin))
    return scaled_x, scaled_y


def _detect_gaze_upgraded(frame):
    """
    Upgraded CV Pipeline:
    1. Mirror flip -> 2. Grayscale -> 3. Gaussian Blur -> 4. Inverse Threshold -> 5. Contours
    Returns: (vector_x, vector_y), processed_frame_for_monitor
    """
    # 1. Mirror the frame for natural intuition
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    _, threshold = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a clean copy of the colored frame to draw interactive UI box for users
    monitor_frame = frame.copy()

    if not contours:
        cv2.putText(monitor_frame, "FACE/EYE LOST", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return None, monitor_frame

    # Assume the largest dark area is the eye region
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    
    # Calculate absolute pupil center
    pupil_x = x + w / 2
    pupil_y = y + h / 2
    
    # CORE UPGRADE: Calculate relative feature displacement vector from the eye corner (x, y)
    vector_x = pupil_x - x
    vector_y = pupil_y - y

    # Draw interactive GUI bounding boxes on the camera monitor window
    cv2.rectangle(monitor_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.circle(monitor_frame, (int(pupil_x), int(pupil_y)), 4, (255, 0, 0), -1)
    cv2.putText(monitor_frame, f"Vector: ({vector_x:.1f}, {vector_y:.1f})", (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return (vector_x, vector_y), monitor_frame


def run_camera_process(queue, stop_event, screen_width, screen_height):
    """
    Subprocess loop running the webcam and writing raw vectors to the queue,
    while creating a local interactive UI window for the user monitor.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        queue.put(('error', 'Unable to open the webcam.'))
        return

    # Warm-up hardware buffer
    for _ in range(5):
        cap.read()
        time.sleep(0.05)

    # Create the dedicated high-level interactive window for users
    cv2.namedWindow("Eye Tracker Monitor", cv2.WINDOW_AUTOSIZE)

    while not stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            queue.put(('error', 'Camera frame read failed.'))
            break

        # Process frame and extract vector
        vector, monitor_frame = _detect_gaze_upgraded(frame)
        
        # Display the live camera monitor feedback window
        cv2.imshow("Eye Tracker Monitor", monitor_frame)
        
        # Critical for OpenCV GUI refresh window threads; also checks if 'q' pressed to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if vector is None:
            queue.put(('point', None, None))
        else:
            vx, vy = vector
            queue.put(('point', vx, vy))

        time.sleep(0.03)

    cap.release()
    cv2.destroyAllWindows()
