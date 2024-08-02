from .QReader import qreader

import datetime
import time
import numpy as np
from ultralytics import YOLO


class Qreaderxp:

    def __init__(self,model_weight):
        self.qrdetector = YOLO(model_weight) ### YOLOv8 qr code detector 
        self.qreaderx = qreader.QReader(model_size = 'n') ### QR code decoder

        ### warming YOLO model
        print(f"[INFO] {datetime.datetime.now()}: YOLOv8 QR detection model warming up!!!\n")
         
        self.qrdetector.predict(device="cuda",source=np.zeros((640,640,3)),verbose=False)

        print(f"[INFO] {datetime.datetime.now()}: Qreader loaded!!!\n")



    # Method to pass image through the model
    def forward(self, image):
        st = time.time()
        pred = self.qrdetector.predict(device="cuda", source=image,verbose=False)
        pred_boxes = pred[0].boxes.xyxy.tolist()
        print(f"\n\n[INFO] {datetime.datetime.now()}:time for QR detection only:{time.time()-st}")

        decoded_text = self.qreaderx.mod_decode(image=image,detection_result=pred_boxes[0]) ### decoding detected QR code
        print(f"[INFO] {datetime.datetime.now()}:time for QR reader full process:{time.time()-st}\n\n")

        return decoded_text
 
    # Method to call for direct inferencing
    def __call__(self, image):
        predicted_text = self.forward(image)
        return predicted_text
