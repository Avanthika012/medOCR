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
from collections import OrderedDict
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

def create_result_image(original_image, roi_results):
    height, width = original_image.shape[:2]
    
    # Create the main image area
    result_image = original_image.copy()
    
    # Calculate the required dashboard height based on the maximum number of results
    max_results = max(len(results) for results in roi_results)
    dashboard_height = 200 + max_results * 40  # Increased base height for more spacing
    
    # Create the dashboard area below the main image
    dashboard = np.zeros((dashboard_height, width, 3), dtype=np.uint8)
    
    # Add title and border to dashboard
    title = "Frinks.ai medOCR"
    title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)[0]
    title_x = (width - title_size[0]) // 2
    title_y = 100  # Adjusted y-position to be in the middle of dashed lines

    # Draw dashed lines
    def draw_dashed_line(img, start_point, end_point, color, thickness=1, dash_length=10, gap_length=10):
        x1, y1 = start_point
        x2, y2 = end_point
        dist = ((x1-x2)**2 + (y1-y2)**2)**.5
        dashes = int(dist / (dash_length + gap_length))
        for i in range(dashes):
            start = int(i * (dash_length + gap_length))
            end = int(start + dash_length)
            if end > dist: end = int(dist)
            x1_temp = int(x1 + (x2-x1) * (start/dist))
            x2_temp = int(x1 + (x2-x1) * (end/dist))
            y1_temp = int(y1 + (y2-y1) * (start/dist))
            y2_temp = int(y1 + (y2-y1) * (end/dist))
            cv2.line(img, (x1_temp, y1_temp), (x2_temp, y2_temp), color, thickness)

    draw_dashed_line(dashboard, (0, 60), (width, 60), (255, 255, 255), 2)
    draw_dashed_line(dashboard, (0, 140), (width, 140), (255, 255, 255), 2)
    
    cv2.putText(dashboard, title, (title_x, title_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    
    # Add legends to dashboard (both on the right side, between dashed lines)
    legend_y1 = 85
    legend_y2 = 115
    cv2.putText(dashboard, "Red color - NG", (width - 300, legend_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(dashboard, "Green color - Passed", (width - 300, legend_y2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Draw ROI separators and labels on the main image
    roi_width = width // 3
    for i in range(3):
        if i >0:
            cv2.line(result_image, (i*roi_width, 0), (i*roi_width, height), (0, 255, 0), 2)
        label = f"ROI-{i+1}"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
        text_x = i * roi_width + (roi_width - text_size[0]) // 2
        cv2.putText(result_image, label, (text_x, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    # Add ROI results to dashboard
    for i, results in enumerate(roi_results):
        start_x = i * (width // 3)
        start_y = 180  # Increased starting y-position
        cv2.putText(dashboard, f"ROI-{i+1} Results:", (start_x + 10, start_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Reorder and display results
        ordered_results = []
        for text, color in results:
            if text.startswith("QR"):
                ordered_results.insert(0, (text, color))
            elif text.startswith("B.NO"):
                ordered_results.insert(1, (text, color))
            elif text.startswith("MFG"):
                ordered_results.insert(2, (text, color))
            elif text.startswith("EXP"):
                ordered_results.insert(3, (text, color))
            elif text.startswith("M.R.P"):
                ordered_results.insert(4, (text, color))
            else:
                ordered_results.append((text, color))
        
        for j, (text, color) in enumerate(ordered_results):
            cv2.putText(dashboard, text, (start_x + 10, start_y + 40 + j*40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Combine main image and dashboard
    final_image = np.vstack((result_image, dashboard))
    
    return final_image


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

    # print(f"[INFO] {datetime.datetime.now()}:length of thread master from where we are taking images---{len(tobj)}")

    for i in range(len(tobj)):

        img_master['frame'] = tobj[i]

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
        ocr_model = PaddleOCRx(model_weights=params["ocr_models"]["paddleocr"]["model_weights"],rec_char_dict_path=params["ocr_models"]["paddleocr"]["rec_char_dict_path"] )
    else:
        ocr_model = None

    print(f"[INFO] {datetime.datetime.now()}: OCR Model Loading Completed!!!\n" if params["use_ocr_model"] is not None else f"[INFO] {datetime.datetime.now()}: NO OCR model!!! working with text detection only \n")

    # Initialize QR code reader
    qr_reader = Qreaderxp(model_weight=params["qr_model"]["yolov8_weights"])
    print(f"[INFO] {datetime.datetime.now()}: QR Code Reader Initialized!!!\n")

    out_img_path = f"{params['output_dir']}/{params['custom_name']}"
    os.makedirs(out_img_path, exist_ok=True)

    ### cv2 result window
    # cv2.namedWindow("medOCR_results", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("medOCR_results", )




    while True:
        try:
            stx = time.time()
            img_master_dict = OrderedDict()
            img_master_dict = extractFrameVCO(img_master=img_master_dict,thread_master=transmittor)  # extracting frames from video capture objects



            if len(img_master_dict.keys()) != 0:
                print("*"*100)
                print(time.time())
                # print("Time taken for 1 loop: ", time.time() - main_st, time.time())
                print("QUEUE SIZE: ", transmittor.saveQueue.qsize())
                main_st = time.time()

                frame=list(img_master_dict.values())[0]
                if len(list(frame.shape)) == 2:
                    frame=cv2.merge([frame,frame,frame])
                # img_count+=1
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

                if result_image is not None and result_image.size > 0:
                    # Resize the image for display
                    # display_image = cv2.resize(result_image, (result_image.shape[1]//2, result_image.shape[0]//2))
                    display_image = cv2.resize(result_image, (result_image.shape[1]//3, result_image.shape[0]//3))

                    print(f"\n\n\n display_image size:{display_image.shape}")
                
                    cv2.imshow("medOCR_results", display_image)
                    cv2.waitKey(1)  # Add this line to allow window updates
                
                # Save the result image
                im_name = time.time()
                cv2.imwrite(f"{out_img_path}/{im_name}_result.png", result_image)
                print(f"[INFO] {datetime.datetime.now()}: Result image saved at {out_img_path}/{im_name}_result.png.\n time for whole process:{time.time()-stx}")                
            # else:
            #     print("no frame data!!")
            #     cv2.imshow("medOCR_results",np.zeros((640,640,3)))

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