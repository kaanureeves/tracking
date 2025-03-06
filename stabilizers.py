import cv2
import numpy as np

class BaseStabilizer:
    def __init__(self,max_corners=200,quality_level=0.01,min_distance=30,block_size=3):
        self.max_corners=max_corners
        self.quality_level=quality_level
        self.min_distance=min_distance
        self.block_size=block_size
        self.prev_gray=None
        self.prv_pts=None
        
    # ilk frameyi alıp griye çevirir ve good featureleri tespit eder (heralde)
    def initialize(self,frame):
        self.prev_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        self.prev_pts=cv2.goodFeaturesToTrack(
            self.prev_gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=self.block_size
        )
        
    # loopta sonraki kare için good featureleri günceller
    def update_points(self,gray_frame):
        self.prev_pts=cv2.goodFeaturesToTrack(
            gray_frame,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=self.block_size
        )
     
    # optical flowu hesaplar
    def compute_optical_flow(self,gray_frame):
        p1,st,err=cv2.calcOpticalFlowPyrLK(self.prev_gray,gray_frame,self.prev_pts,None)
        good_new=p1[st==1]
        good_old=self.prev_pts[st==1]
        return good_old,good_new,gray_frame
    
    # (?)
    def stabilize(self,frame):
        raise NotImplementedError("Bu metot alt sınıflarda uygulanmalıdır.")
    
class AffineStabilizer(BaseStabilizer):
    # affine transformation
    def stabilize(self,frame):
        if self.prev_gray is None or self.prev_pts is None:
            self.initialize(frame)
            return frame
        
        gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        good_old,good_new,gray_frame=self.compute_optical_flow(gray_frame)
        
        if len(good_old)>=3:
            M,inliers=cv2.estimateAffinePartial2D(good_old,good_new)
            if M is not None:
                stabilized_frame=cv2.warpAffine(frame,M,(frame.shape[1],frame.shape[0]))
            else:
                stabilized_frame = frame.copy()
        else:
            stabilized_frame = frame.copy()
            
        self.prev_gray=gray_frame
        self.update_points(gray_frame)
        return stabilized_frame
    
class PerspectiveStabilizer(BaseStabilizer):
    # perspective transformation
    def stabilize(self,frame):
        if self.prev_gray is None or self.prev_pts is None:
            self.initialize(frame)
            return frame
        
        gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        good_old,good_new,gray_frame=self.compute_optical_flow(gray_frame)
        
        if len(good_old) >= 4:
            H, status=cv2.findHomography(good_old,good_new,cv2.RANSAC, 5.0)
            if H is not None:
                stabilized_frame=cv2.warpPerspective(frame,H,(frame.shape[1],frame.shape[0]))
            else:
                stabilized_frame=frame.copy()
        else:
            stabilized_frame=frame.copy()
            
        self.prev_gray = gray_frame
        self.update_points(gray_frame)
        return stabilized_frame

