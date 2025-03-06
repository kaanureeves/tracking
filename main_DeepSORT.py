import sys
import time
import subprocess
import datetime
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

CONFIDENCE_THRESHOLD=0.5
GREEN=(0,255,0)
WHITE=(255,255,255)

model_path="C:/Users/kaang/tracking/codes/yolo11n.pt"
video_path="C:/Users/kaang/tracking/ucan_goruntu/los_angeles.mp4"

cap=cv2.VideoCapture(0)

model=YOLO(model=model_path)
tracker=DeepSort(max_age=100)

allowed_class=[0,4] # person,airplane
class_names = {0: "person", 4: "airplane"}

while True:
    start=datetime.datetime.now()
    
    ret,frame=cap.read()
    if not ret:
        break
    
    detections=model(frame)[0]
    
    results=[]
    for data in detections.boxes.data.tolist():
        confidence=data[4]
        if float(confidence)<CONFIDENCE_THRESHOLD:
            continue
        
        xmin,ymin,xmax,ymax=int(data[0]),int(data[1]),int(data[2]),int(data[3])
        class_id=int(data[5])
        
        if class_id not in allowed_class:
            continue
        
        results.append(([[xmin,ymin,xmax-xmin,ymax-ymin],confidence,class_id]))
        
    tracks=tracker.update_tracks(results,frame=frame)
    
    for track in tracks:
        if not track.is_confirmed():
            continue
        
        
        track_id=track.track_id
        ltrb=track.to_ltrb()
        xmin,ymin,xmax,ymax=int(ltrb[0]),int(ltrb[1]),int(ltrb[2]),int(ltrb[3])
        
        det_class=None
        if hasattr(track,'detection') and track.detection is not None:
            det_class=track.detection[2]
        label_text = f"{track_id}: {class_names.get(det_class, 'Unknown')}" if det_class is not None else str(track_id)
        
        cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),GREEN,2)
        cv2.rectangle(frame,(xmin,ymin-20),(xmin+20,ymin),GREEN,-1)
        cv2.putText(frame,str(track_id),(xmin+5,ymin-8),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,WHITE,2)
        
    end=datetime.datetime.now()
    print(f"Time to process 1 frame: {(end - start).total_seconds() * 1000:.0f} milliseconds")
        
    fps = f"FPS: {1/(end-start).total_seconds():.2f}"
    cv2.putText(frame,fps,(50,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),8)
        
    cv2.imshow("Frame",frame)
    if cv2.waitKey(1)==ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
