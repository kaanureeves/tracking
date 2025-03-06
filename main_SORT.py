import sys
import time
sys.path.append('C:/Users/kaang/tracking/codes/sort')
from ultralytics import YOLO    
from sort import Sort
import torch
import cv2
from stabilizers import AffineStabilizer, PerspectiveStabilizer
import numpy as np

video_path="C:/Users/kaang/tracking/ucan_goruntu/bogaziçi.mp4"
yolo_model_path="C:/Users/kaang/tracking/codes/yolov8n.pt"

# stabilizer=AffineStabilizer()

cap=cv2.VideoCapture(video_path)

model=YOLO(yolo_model_path)
tracker=Sort(max_age=100,min_hits=3,iou_threshold=0.3)

track_history={}
max_history=1000

start_time=time.time()
frame_count=0

while True:
    ret,frame=cap.read()
    
    if not ret:
        break
    
    # stabilized_frame=stabilizer.stabilize(frame)
    
    results=model(frame)
    
    if results[0].boxes is not None and len(results[0].boxes)>0:
        boxes=results[0].boxes.xyxy.cpu().numpy()
        confs=results[0].boxes.conf.cpu().numpy()
        dets=np.hstack((boxes,confs.reshape(-1, 1)))
    else:
        dets=np.empty((0,5))
        
    tracks=tracker.update(dets)
    print("Tracks: ",tracks)
    
    for track in tracks:
        x1,y1,x2,y2,track_id=track.astype(int)
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
        cv2.putText(frame,f"ID: {track_id}",(x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
        
                
    frame_count+=1
    elapsed_time=time.time()-start_time
    fps=frame_count/elapsed_time
    
    cv2.putText(frame,f"FPS:{fps:.2f}",(10,30),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    
    cv2.imshow("Frame",frame)
    
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

