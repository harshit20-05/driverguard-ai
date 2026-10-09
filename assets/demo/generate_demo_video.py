"""
Generates a lightweight synthetic driver video for Demo Mode.
Simulates a driver looking ahead, blinking naturally, and experiencing a prolonged closure event.
"""

import os
import cv2
import numpy as np


def generate_sample_video(output_path: str = "assets/demo/sample_driver.mp4", duration_sec: int = 8, fps: int = 24):
    """Generates an MP4 video of an illustrative stylized face driver."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    width, height = 640, 480
    total_frames = duration_sec * fps

    # Try MP4V or XVID codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    face_center = (320, 240)
    left_eye_center = (270, 210)
    right_eye_center = (370, 210)

    for i in range(total_frames):
        t = i / fps
        frame = np.full((height, width, 3), (25, 20, 30), dtype=np.uint8)

        # Draw vehicle interior background
        cv2.rectangle(frame, (0, 380), (640, 480), (40, 35, 45), -1)
        # Steering wheel arc
        cv2.ellipse(frame, (320, 490), (180, 110), 0, 180, 360, (60, 60, 70), 22)

        # Head / Face
        cv2.ellipse(frame, face_center, (95, 125), 0, 0, 360, (190, 205, 235), -1)

        # Determine eye state based on timestamp:
        # 0s - 2s: Open
        # 2s - 2.2s: Normal Blink
        # 2.2s - 4.5s: Open
        # 4.5s - 7.0s: Prolonged Drowsy Eye Closure!
        # 7.0s - 8.0s: Recover Open
        is_closed = False
        if 2.0 <= t <= 2.25:
            is_closed = True
        elif 4.5 <= t <= 6.8:
            is_closed = True

        for eye_c in [left_eye_center, right_eye_center]:
            if is_closed:
                # Eyelid line
                cv2.line(frame, (eye_c[0] - 22, eye_c[1]), (eye_c[0] + 22, eye_c[1]), (50, 40, 60), 3)
            else:
                # Open eye white
                cv2.ellipse(frame, eye_c, (22, 14), 0, 0, 360, (255, 255, 255), -1)
                # Iris
                cv2.circle(frame, eye_c, 8, (90, 60, 30), -1)
                # Pupil
                cv2.circle(frame, eye_c, 4, (10, 10, 10), -1)
                # Highlight
                cv2.circle(frame, (eye_c[0] - 3, eye_c[1] - 3), 2, (255, 255, 255), -1)

        # Nose & Mouth
        cv2.line(frame, (320, 225), (320, 255), (160, 175, 205), 3)
        cv2.ellipse(frame, (320, 290), (28, 8), 0, 0, 180, (140, 110, 130), 3)

        # Header banner in frame
        cv2.putText(frame, "DRIVERGUARD AI - DEMO STREAM", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 180), 2, cv2.LINE_AA)
        status_label = "CLOSED (Drowsy Simulation)" if is_closed else "OPEN (Attentive)"
        cv2.putText(frame, f"Simulated State: {status_label}", (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        out.write(frame)

    out.release()
    return output_path


if __name__ == "__main__":
    generate_sample_video()
