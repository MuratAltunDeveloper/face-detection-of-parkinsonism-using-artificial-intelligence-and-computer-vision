import cv2
import numpy as np
from py_feat import PyFeat

def process_video(video_path):
    # Py-Feat modelini başlat
    py_feat = PyFeat()

    # Video akışını başlat
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Frame'i Py-Feat ile işle
        action_units = py_feat.detect(frame)

        # AU25 değerini al (varsa)
        au25_intensity = action_units.get(25, 0) if action_units is not None else 0

        # Görselleştirme
        if action_units is not None:
            # AU25 yoğunluğunu ekranda göster
            cv2.putText(frame, f"AU25: {au25_intensity:.2f}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2)

        # Sonucu göster
        cv2.imshow('AU25 Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Video yolunu belirle ve işle
video_path = r"C:\Users\murat\OneDrive\Masaüstü\parkinson\Canan_Karaman_Durus_TekAyak.mp4"
process_video(video_path)
