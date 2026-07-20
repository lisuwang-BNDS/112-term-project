import cv2
import json
import os
import time

FRAME_DELAY = 0.03


def _clamp_unit(value):
    return max(0.0, min(1.0, value))


def _write_state(state_path, data):
    try:
        temp_path = state_path + '.tmp'
        with open(temp_path, 'w') as f:
            json.dump(data, f)
        os.replace(temp_path, state_path)
    except Exception:
        pass


def _detect_gaze_cv(frame):
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    _, threshold = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    monitor_frame = frame.copy()

    if not contours:
        cv2.putText(
            monitor_frame,
            'FACE/EYE LOST',
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        return None, monitor_frame

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    moments = cv2.moments(largest)

    if moments['m00'] != 0:
        centroid_x = moments['m10'] / moments['m00']
        centroid_y = moments['m01'] / moments['m00']
    else:
        centroid_x = x + w / 2
        centroid_y = y + h / 2

    vector_x = centroid_x - x
    vector_y = centroid_y - y

    cv2.rectangle(monitor_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.circle(monitor_frame, (int(centroid_x), int(centroid_y)), 4, (255, 0, 0), -1)
    cv2.putText(
        monitor_frame,
        f'Vector: ({vector_x:.1f}, {vector_y:.1f})',
        (x, max(20, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )

    return (vector_x, vector_y), monitor_frame


def run_camera_process(state_path):
    """后台独立进程：用 OpenCV 读取眼睛位置并输出到 state 文件。"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        message = 'Unable to open default webcam.'
        print(f'[Camera] {message}')
        _write_state(state_path, {
            'status': 'error',
            'error': message,
            'raw_x': None,
            'raw_y': None,
            'timestamp': time.time(),
        })
        return

    for _ in range(5):
        cap.read()
        time.sleep(0.05)

    cv2.namedWindow('Eye Tracker Monitor', cv2.WINDOW_AUTOSIZE)
    print('[Camera] Eye tracker process started.')

    try:
        while True:
            success, frame = cap.read()
            if not success:
                _write_state(state_path, {
                    'status': 'error',
                    'error': 'Cannot read camera frame.',
                    'raw_x': None,
                    'raw_y': None,
                    'timestamp': time.time(),
                })
                time.sleep(0.5)
                continue

            vector, monitor_frame = _detect_gaze_cv(frame)
            cv2.imshow('Eye Tracker Monitor', monitor_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if vector is None:
                _write_state(state_path, {
                    'status': 'lost',
                    'error': 'No eye vector detected.',
                    'raw_x': None,
                    'raw_y': None,
                    'timestamp': time.time(),
                })
            else:
                vx, vy = vector
                _write_state(state_path, {
                    'status': 'tracking',
                    'error': None,
                    'x': float(vx),
                    'y': float(vy),
                    'raw_x': float(vx),
                    'raw_y': float(vy),
                    'timestamp': time.time(),
                })

            time.sleep(FRAME_DELAY)
    except KeyboardInterrupt:
        print('[Camera] Keyboard interrupt received, stopping.')
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print('[Camera] Camera closed.')


if __name__ == '__main__':
    import sys

    state_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), 'camera_state.json')
    run_camera_process(state_path)
