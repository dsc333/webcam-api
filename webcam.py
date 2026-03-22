from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, Response
import cv2
import time

app = FastAPI()

# Open default webcam (0)
camera = cv2.VideoCapture(0)

# Optional: set resolution
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def get_frame():
    success, frame = camera.read()
    if not success:
        return None
    return frame


def generate_frames():
    while True:
        frame = get_frame()
        if frame is None:
            time.sleep(0.1)
            continue

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
        <head>
            <title>Webcam Stream</title>
        </head>
        <body>
            <h1>FastAPI Webcam Stream</h1>
            <p><a href="/snapshot" target="_blank">Open snapshot</a></p>
            <img src="/video" width="800" />
        </body>
    </html>
    """


@app.get("/video")
def video():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/snapshot")
def snapshot():
    frame = get_frame()
    if frame is None:
        raise HTTPException(status_code=500, detail="Could not read frame from webcam")

    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode frame")

    return Response(content=buffer.tobytes(), media_type="image/jpeg")
