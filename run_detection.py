import os 
import torch
import cv2
import json
import time 
from models.fasterrcnn_inference import FasterRCNN
from models.paddleocr.tools.infer.predict_rec import PaddleOCRx
from models.QReader_repo.qreader_inference import Qreaderxp

import argparse
import traceback
from tqdm import tqdm 
import datetime
import sys
import numpy as np
import random

from camera.transmitor_class import Transmittor
from collection import OrderedDict
from dotenv import load_dotenv
load_dotenv()


#### transmittor code 

try:    
    # is_active= is_active as the model should be running in background. Hence read frames from the working RTSP only
    # thread_master = ImageQServer(tcp_port=GP.TCP_PORT).start()
    transmittor = Transmittor(camera_ip=os.getenv("CAMERA_IP"))
    # thread_master = ImageQServer(tcp_port=5678).start()

    print(f"[INFO] {datetime.datetime.now()}:Transmittor created")
except Exception as e:
    print(f"[ERROR] {datetime.datetime.now()}:Error at Transmittor initiation ")
    traceback.print_exception(*sys.exc_info())


def generate_unique_colors(n):
    colors = []
    for i in range(n):
        color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        colors.append(color)
    return colors

def split_image(image):
    height, width = image.shape[:2]
    split_width = width // 3
    roi_1 = image[:, :split_width]
    roi_2 = image[:, split_width:2*split_width]
    roi_3 = image[:, 2*split_width:]
    return [roi_1, roi_2, roi_3]

def draw_dashed_line(img, start, end, color, thickness=1, dash_length=10, gap_length=10):
    dist = ((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2) ** 0.5
    dash_points = []
    for i in np.arange(0, dist, dash_length + gap_length):
        r_start = i / dist
        r_end = (i + dash_length) / dist
        x_start = int((1 - r_start) * start[0] + r_start * end[0])
        y_start = int((1 - r_start) * start[1] + r_start * end[1])
        x_end = int((1 - r_end) * start[0] + r_end * end[0])
        y_end = int((1 - r_end) * start[1] + r_end * end[1])
        dash_points.append(((x_start, y_start), (x_end, y_end)))

    for dash in dash_points:
        cv2.line(img, dash[0], dash[1], color, thickness)

def create_result_image(original_image, roi_results):
    height, width = original_image.shape[:2]
    canvas_height = height
    canvas_width = width * 2
    
    result_image = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    result_image[:, :width] = original_image
    
    roi_width = width // 3
    cv2.line(result_image, (roi_width, 0), (roi_width, height), (0, 255, 0), 2)
    cv2.line(result_image, (2*roi_width, 0), (2*roi_width, height), (0, 255, 0), 2)
    
    for i in range(3):
        cv2.putText(result_image, f"ROI-{i+1}", (i*roi_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    line_type = 2
    font_color = (255, 255, 255)
    
    cv2.putText(result_image, "frinks.ai medOCR", (width + 10, 30), font, 1, font_color, line_type)
    cv2.line(result_image, (width, 50), (canvas_width, 50), font_color, 1)
    
    # Add legends
    legend_font_scale = 0.4
    cv2.putText(result_image, "Red color - NG", (canvas_width - 200, 20), font, legend_font_scale, (0, 0, 255), 1)
    cv2.putText(result_image, "Green color - Passed", (canvas_width - 200, 40), font, legend_font_scale, (0, 255, 0), 1)
    
    # Define layout parameters
    start_y = 80
    start_x = width + 10
    column_width = (canvas_width - width) // 2
    row_height = 150  # Adjust this value based on your needs
    
    for i, results in enumerate(roi_results):
        current_x = start_x + (i % 2) * column_width
        current_y = start_y + (i // 2) * row_height
        
        cv2.putText(result_image, f"ROI-{i+1} Results:", (current_x, current_y), font, font_scale, font_color, line_type)
        current_y += 30
        
        for text, color in results:
            cv2.putText(result_image, text, (current_x + 10, current_y), font, font_scale, color, line_type)
            current_y += 30
        
        cv2.line(result_image, (current_x, current_y), (current_x + column_width - 20, current_y), font_color, 1)
    
    return result_image

def process_roi(roi, model, ocr_model, det_th, classes, qr_reader, match_txt):
    qr_text = qr_reader(roi)
    dtt = time.time()
    boxes, class_names, scores = model(roi)
    print(f"[INFO] {datetime.datetime.now()}: time taken for text detection:{time.time()-dtt} seconds")
    results = []
    
    if qr_text:
        results.append((f"QR: {qr_text}", (255, 255, 255)))  # White color for QR code text
    else:
        results.append(("[ No QR detected!!! ]", (0, 0, 255)))  # Red color for no QR detected
    
    text_detected = False
    for i in range(len(class_names)):
        if scores[i] >= det_th:
            x1, y1, x2, y2 = boxes[i]
            cname = class_names[i]
            color = (100, 100, 255)  # Light red color for bounding box
            cv2.rectangle(roi, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            cropped_image = roi[int(y1):int(y2), int(x1):int(x2)]
            ocrtt = time.time()
            ocr_text = ocr_model(cropped_image)
            print(f"[INFO] {datetime.datetime.now()}: time taken for OCR text recog:{time.time()-ocrtt} seconds")

            if ocr_text:
                text_detected = True
                ocr_text_upper = ocr_text.upper()
                if ocr_text_upper in [txt.upper() for txt in match_txt]:
                    text_color = (0, 255, 0)  # Green for matched text
                else:
                    text_color = (0, 0, 255)  # Red for unmatched text
                results.append((ocr_text_upper, text_color))
            
            text = f"{cname}: {ocr_text_upper}"
            cv2.putText(roi, text, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    if not text_detected:
        results.append(("[ No texts detected!!! ]", (0, 0, 255)))  # Red color for no text detected
    
    return roi, results
def img_inferencing(image_dir, out_path, ocr_model, model, det_th, custom_name, classes, qr_reader, match_txt):
    print(f"[INFO] {datetime.datetime.now()}: --------- IMAGE INFERENCING STARTED --------- \n")

    out_img_path = f"{out_path}/{custom_name}"
    os.makedirs(out_img_path, exist_ok=True)

    for im_name in tqdm(os.listdir(image_dir)):
        print(f"\n[INFO] {datetime.datetime.now()}: working with {im_name}\n")
        img_path = os.path.join(image_dir, im_name)
        st = time.time()
        img = cv2.imread(img_path)
        
        rois = split_image(img)
        processed_rois = []
        roi_results = []
        
        for i, roi in enumerate(rois):           
            processed_roi, results = process_roi(roi, model, ocr_model, det_th, classes, qr_reader, match_txt)
            processed_rois.append(processed_roi)
            roi_results.append(results)
        
        # Merge processed ROIs
        merged_image = np.concatenate(processed_rois, axis=1)
        
        # Create final result image
        result_image = create_result_image(merged_image, roi_results)
        
        # Save the result image
        cv2.imwrite(f"{out_img_path}/{im_name[:-4]}_result.png", result_image)
        print(f"[INFO] {datetime.datetime.now()}: Result image saved at {out_img_path}/{im_name[:-4]}_result.png.\n time for whole process:{time.time()-st}")

    print(f"[INFO] {datetime.datetime.now()}: --- IMAGE INFERENCING COMPLETED ---")

# ... (rest of the code remains the same)

def dirchecks(file_path):
    if not os.path.exists(file_path):
        print(f"[INFO] {datetime.datetime.now()}: Can not find this directory:\n{file_path}. Please check.\n Exiting!!!!\n")
        sys.exit(1)
    else:
        print(f"[INFO] {datetime.datetime.now()}: Found this directory:\n{file_path}.\n")
def extractFrameVCO(img_master,thread_master):

    tobj = thread_master.read()

    print(f"[INFO] {datetime.datetime.now()}:length of thread master from where we are taking images---{len(tobj)}")

    for i in range(len(tobj)):
        idb, img= tobj[i]["name"], tobj[i]["frame"]

        img_master[idb] = img

    print(f"[INFO] {datetime.datetime.now()}:length of image master going into python---{len(img_master.keys())}")

    return img_master

def main(params):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] {datetime.datetime.now()}: device available: {device}  ------xxxxxxxxx \n")

    dirchecks(params["image_dir"])
    transmittor.run()

    if params["use_model"] == "fasterrcnn":
        model = FasterRCNN(model_weights=params["models"]["fasterrcnn"]["model_weights"], classes=params["classes"], device=device, detection_thr=params["models"]["fasterrcnn"]["det_th"])
        detection_thr = params["models"]["fasterrcnn"]["det_th"]
    else:
        model = None

    print(f"[INFO] {datetime.datetime.now()}: Text Detection Model Loading Completed!!!\n")

    if params["use_ocr_model"] == "paddleocr":
        ocr_model = PaddleOCRx(model_weights=params["ocr_models"]["paddleocr"]["model_weights"])
    else:
        ocr_model = None

    print(f"[INFO] {datetime.datetime.now()}: OCR Model Loading Completed!!!\n" if params["use_ocr_model"] is not None else f"[INFO] {datetime.datetime.now()}: NO OCR model!!! working with text detection only \n")

    # Initialize QR code reader
    qr_reader = Qreaderxp(model_weight=params["qr_model"]["yolov8_weights"])
    print(f"[INFO] {datetime.datetime.now()}: QR Code Reader Initialized!!!\n")

    out_img_path = f"{params["output_dir"]}/{params["custom_name"]}"
    os.makedirs(out_img_path, exist_ok=True)

    ### cv2 result window
    cv2.namedWindow("medOCR_results", cv2.WINDOW_NORMAL)



    while True:
        try:
            stx = time.time()
            img_master_dict = OrderedDict()
            img_master_dict = extractFrameVCO(img_master=img_master_dict,thread_master=transmittor)  # extracting frames from video capture objects



            if len(img_master_dict.keys()) != 0:
                print("*"*100)
                print(time.time())
                print("Time taken for 1 loop: ", time.time() - main_st, time.time())
                print("QUEUE SIZE: ", transmittor.saveQueue.qsize())
                main_st = time.time()

                frame=list(img_master_dict.values())[0]
                if len(list(frame.shape)) == 2:
                    frame=cv2.merge([frame,frame,frame])
                img_count+=1
                live_frame=frame.copy()

                rois = split_image(live_frame)
                processed_rois = []
                roi_results = []
                
                for i, roi in enumerate(rois):           
                    processed_roi, results = process_roi(roi, model, ocr_model, detection_thr, params["classes"], qr_reader, params["match_txt"])
                    processed_rois.append(processed_roi)
                    roi_results.append(results)
                
                # Merge processed ROIs
                merged_image = np.concatenate(processed_rois, axis=1)
                
                # Create final result image
                result_image = create_result_image(merged_image, roi_results)

                ### cv2 imshow
                cv2.imshow("medOCR_results",result_image)
                
                # Save the result image
                im_name = time.time()
                cv2.imwrite(f"{out_img_path}/{im_name}_result.png", result_image)
                print(f"[INFO] {datetime.datetime.now()}: Result image saved at {out_img_path}/{im_name}_result.png.\n time for whole process:{time.time()-stx}")                
        except Exception as e:
            print(f"[ERROR] {datetime.datetime.now()}:Error at while loop in main()")
            traceback.print_exception(*sys.exc_info())
            sys.exit(1)

    # if params["image_dir"] is not None:
    #     start = time.time()
    #     img_inferencing(params["image_dir"], out_path=params["output_dir"], ocr_model=ocr_model, model=model, qr_reader=qr_reader, det_th=detection_thr, custom_name=params["custom_name"], classes=params["classes"], match_txt=params["match_txt"])
    #     print(f"[INFO] {datetime.datetime.now()}: total time taken: {time.time() - start}")
    # else:
    #     print(f"[INFO] {datetime.datetime.now()}: no img path given. Exiting\n")
    #     sys.exit(1)

if __name__ == '__main__':
    try:
        with open('./model_jsons/paramx.json', 'r') as f:
            params = json.load(f)

        print(f"[INFO] {datetime.datetime.now()}: ------------- PROCESS STARTED -------------\n\n\n params:\n{params}\n\n")
        main(params)
        print(f"[INFO] {datetime.datetime.now()}: ------------- PROCESS COMPLETED -------------\n\n\n")

    except:
        print(f"\n [ERROR] {datetime.datetime.now()} \n ")
        traceback.print_exception(*sys.exc_info())