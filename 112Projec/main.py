import multiprocessing as mp
import os
import queue
import threading
import time
from cmu_graphics import *
import eye_tracker

MIN_CALIB_SAMPLES = 5


def get_gazerecorder_style_pointer(app, current_vx, current_vy): #why calib helpsss

    Sumweight = 0.0
    screenX = 0.0
    screenY = 0.0

    for target_id, calib_vector in app.calib_raw_results.items():
        calib_vx, calib_vy = calib_vector
        screen_pos = app.calib_targets[target_id] 

        dist = ((current_vx - calib_vx) ** 2 + (current_vy - calib_vy) ** 2) ** 0.5

        if dist < 0.05:
            return screen_pos[0], screen_pos[1]

        weight = 1.0 / (dist ** 2)
        Sumweight += weight
        screenX += screen_pos[0] * weight
        screenY += screen_pos[1] * weight

    if Sumweight == 0:
        return app.width // 2, app.height // 2

    return int(screenX / Sumweight), int(screenY / Sumweight)


def _capture_calibration_point(app):
    if app.calib_index >= len(app.calib_order):
        return

    if len(app.stable_samples) < 120:
        app.camera_message = f'Need least {MIN_CALIB_SAMPLES} '
        return

    avg_vx = sum(p[0] for p in app.stable_samples) / len(app.stable_samples)
    avg_vy = sum(p[1] for p in app.stable_samples) / len(app.stable_samples)

    current_target_id = app.calib_order[app.calib_index]
    app.calib_raw_results[current_target_id] = (avg_vx, avg_vy)

    app.stable_samples = []
    app.calib_timer = 0
    app.calib_index += 1

    if app.calib_index >= len(app.calib_order):
        app.state = 'game'
        app.camera_message = 'calibration complete'
    else:
        app.camera_message = (
            f'Captured point {app.calib_index} / {len(app.calib_order)}. '
            'Move by press Space or Enter.'
        )


def _camera_queue_listener(app):
    while not getattr(app, 'stop_event', None).is_set():
        try:
            payload = app.camera_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        with app.lock:
            if not payload:
                continue

            kind = payload[0]
            if kind == 'point':
                _, vx, vy = payload
                if vx is None or vy is None:
                    app.raw_vx = None
                    app.raw_vy = None
                    app.camera_status = 'lost'
                    app.camera_message = 'No eye vector detected.'
                else:
                    app.raw_vx = vx
                    app.raw_vy = vy
                    app.camera_status = 'tracking'
                    app.camera_message = 'Tracking eye vector.'
            elif kind == 'error':
                _, message = payload
                app.raw_vx = None
                app.raw_vy = None
                app.camera_status = 'error'
                app.camera_message = message


def onAppStart(app):
    app.width = 800
    app.height = 600
    app.stepsPerSecond = 50

    app.stop_event = threading.Event()
    app.lock = threading.Lock()

    app.raw_vx = None
    app.raw_vy = None
    app.gaze_x = None
    app.gaze_y = None

    app.camera_status = 'starting'
    app.camera_message = 'Starting camera monitor...'

    app.state = 'calibration'
    app.calib_targets = {
        '1': (100, 100), '2': (400, 100), '3': (700, 100),
        '4': (100, 300), '5': (400, 300), '6': (700, 300),
        '7': (100, 500), '8': (400, 500), '9': (700, 500)
    }
    app.calib_order = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    app.calib_index = 0
    app.stable_samples = []
    app.calib_timer = 0
    app.calib_raw_results = {}

    app.camera_queue = mp.Queue(maxsize=20)
    app.camera_stop_event = mp.Event()
    app.camera_process = mp.Process(
        target=eye_tracker.run_camera_process,
        args=(app.camera_queue, app.camera_stop_event, app.width, app.height),
        daemon=True,
    )
    app.camera_process.start()

    app.queue_thread = threading.Thread(target=_camera_queue_listener, args=(app,), daemon=True)
    app.queue_thread.start()


def onStep(app):
    with app.lock:
        current_vx = app.raw_vx
        current_vy = app.raw_vy

    if app.state == 'calibration':
        if current_vx is not None and current_vy is not None:
            app.stable_samples.append((current_vx, current_vy))
            if len(app.stable_samples) > 120:
                app.stable_samples.pop(0)
            app.calib_timer = len(app.stable_samples)
    elif app.state == 'game':
        if current_vx is not None and current_vy is not None:
            cx, cy = get_gazerecorder_style_pointer(app, current_vx, current_vy)
            if app.gaze_x is None:
                app.gaze_x, app.gaze_y = cx, cy
            else:
                ALPHA = 0.20
                app.gaze_x = int(app.gaze_x + ALPHA * (cx - app.gaze_x))
                app.gaze_y = int(app.gaze_y + ALPHA * (cy - app.gaze_y))
        else:
            app.gaze_x, app.gaze_y = None, None


def onKeyPress(app, key):
    if app.state != 'calibration':
        return

    if key in ['space', 'enter']:
        _capture_calibration_point(app)


def redrawAll(app):
    if app.state == 'calibration':
        drawRect(0, 0, app.width, app.height, fill='aliceBlue')
        
        current_target_id = app.calib_order[app.calib_index]
        tx, ty = app.calib_targets[current_target_id]
        drawCircle(tx, ty, 18, fill='crimson')
        drawCircle(tx, ty, 6, fill='white')

        progress_width = min(len(app.stable_samples), MIN_CALIB_SAMPLES) / MIN_CALIB_SAMPLES * 120
        drawRect(340, 300, 120, 8, fill=None, border='lightGray')
        if progress_width > 0:
            drawRect(340, 300, progress_width, 8, fill='limeGreen')

        
    elif app.state == 'game':
        drawRect(0, 0, app.width, app.height, fill='ghostWhite')
        
        if app.gaze_x is not None and app.gaze_y is not None:
            drawCircle(app.gaze_x, app.gaze_y, 12, fill=None, border='cyan', borderWidth=2)
            drawCircle(app.gaze_x, app.gaze_y, 3, fill='cyan')


def onAppStop(app):
    if getattr(app, 'stop_event', None) is not None:
        app.stop_event.set()
    if getattr(app, 'camera_stop_event', None) is not None:
        app.camera_stop_event.set()

    if getattr(app, 'queue_thread', None) is not None:
        app.queue_thread.join(timeout=1)

    if getattr(app, 'camera_process', None) is not None:
        try:
            if app.camera_process.is_alive():
                app.camera_process.terminate()
                app.camera_process.join(timeout=1)
        except Exception:
            pass


if __name__ == '__main__':
    runApp()
