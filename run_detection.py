import os 
import torch
import cv2
import json
import time 
from models.fasterrcnn_inference import FasterRCNN
from models.paddleocr.tools.infer.predict_rec import PaddleOCRx

import argparse
import traceback
from tqdm import tqdm 
import datetime
import sys
import numpy as np

from models.create_colors import Colors  
colors = Colors()  # create instance for 'from utils.plots import colors'


def dirchecks(file_path):
    if not os.path.exists(file_path):
        print(f"[INFO] {datetime.datetime.now()}: Can not find this directory:\n{file_path}. Please check.\n Exiting!!!!\n")
        sys.exit(1)
    else:
        print(f"[INFO] {datetime.datetime.now()}: Found this directory:\n{file_path}.\n")


def save_cropped_image(image, crop_box, output_dir, base_name, crop_index, extra_pixels=20):
    x1, y1, x2, y2 = crop_box
    
    # Adjust the bounding box with extra pixels
    x1 = max(0, x1 - 5)
    y1 = max(0, y1 - 5)
    x2 = min(image.shape[1], x2 + extra_pixels)
    y2 = min(image.shape[0], y2 + 5)
    
    crop_img = image[y1:y2, x1:x2]
    crop_filename = os.path.join(output_dir, f"{base_name}_crop_{crop_index}.png")
    cv2.imwrite(crop_filename, crop_img)
    print(f"[INFO] {datetime.datetime.now()}: Cropped image saved at {crop_filename}")



def plot_results(img, boxes, class_names, scores, det_th, classes, ocr_model, crop_output_dir, img_name, results_file):
    print(f"[INFO] {datetime.datetime.now()}: detections: boxes:{boxes}\n, class_names:{class_names}\n, scores:{scores}\n, det_th:{det_th}\n")

    for i in range(len(class_names)): 
        if scores[i] >= det_th: 
            x1, y1, x2, y2 = boxes[i]
            cname = class_names[i]

            # Save cropped images
            save_cropped_image(img, (x1, y1, x2, y2), crop_output_dir, img_name, i)

            #### --------    OCR WORK    ----------------
            if ocr_model is not None:
                cropped_image = img[y1:y2, x1:x2]
                st = time.time() 
                ocr_text = ocr_model(cropped_image)
                print(f"[INFO] {datetime.datetime.now()}: time taken for text recognition {time.time() - st } seconds")
                print(f"------------------------------ ocr_text:{ocr_text}")

                # Save image name and OCR text to file
                results_file.write(f"{img_name}_crop_{i}.png\t{ocr_text}\n")

                ### if you want to put class name and recognised text without probability 
                cv2.putText(img, cname + f" : {ocr_text}", (int(x1), int(y1)-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors(classes.index(cname)), 2)
            else:  
                ### if you want to put class name and probability 
                cv2.putText(img, cname + f" {round(float(scores[i]),2)}", (int(x1), int(y1)-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors(classes.index(cname)), 2)    

            ### plotting bboxes, class name and threshold 
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), colors(classes.index(cname)), 2)  # BBox       

    return img  # all bounding boxes are plotted on this image


def img_inferencing(image_dir, out_path, ocr_model, model, det_th, custom_name, classes):
    print(f"[INFO] {datetime.datetime.now()}: --------- IMAGE INFERENCING STARTED --------- \n")

    ### creating output paths
    out_img_path = f"{out_path}/img_out/{custom_name}"
    out_crop_path = f"{out_img_path}/crops"
    os.makedirs(out_img_path, exist_ok=True)
    os.makedirs(out_crop_path, exist_ok=True)
    print(f"[INFO] {datetime.datetime.now()}: output paths created.")

    results_file_path = os.path.join(out_img_path, "results.txt")
    with open(results_file_path, 'w') as results_file:
        print(f"[INFO] {datetime.datetime.now()}: looping through all images.")
        ### reading images and inferencing
        for im_name in tqdm(os.listdir(image_dir)):
            img_path = os.path.join(image_dir, im_name)
            img = cv2.imread(img_path)
            st = time.time()
            boxes, class_names, scores = model(img)
            print(f"[INFO] {datetime.datetime.now()}: time taken for text detection {time.time() - st } seconds")

            ### filtering and plotting results on image
            res_img = plot_results(img, boxes, class_names, scores, det_th, classes, ocr_model, out_crop_path, im_name, results_file)

            ### saving output frames
            cv2.imwrite(f"{out_img_path}/{im_name[:-4]}.png", res_img)
            print(f"[INFO] {datetime.datetime.now()}: result img saved at {out_img_path}/{im_name[:-4]}.png res_img: {res_img.shape} \n ")

    print(f"[INFO] {datetime.datetime.now()}: --- IMAGE INFERENCING COMPLETED ---")


def vid_inferencing(folder_path, output_folder, model, det_th, custom_name, classes, ocr_model, lowfpsvid):
    """
    Process all videos in the specified folder using a detection algorithm on each frame.

    Args:
    folder_path (str): The path to the folder containing videos.
    output_folder (str): The path to the folder where processed video frames should be saved.
                        If None, frames will not be saved.
    """
    print(f"[INFO] {datetime.datetime.now()}: --- VIDEO INFERENCING STARTED --- \n")

    # Create the output folder if it doesn't exist
    out_vid_path = f"{output_folder}/vid_out/{custom_name}"
    out_crop_path = f"{out_vid_path}/crops"
    os.makedirs(out_vid_path, exist_ok=True)
    os.makedirs(out_crop_path, exist_ok=True)
    print(f"[INFO] {datetime.datetime.now()}: output paths created.")

    results_file_path = os.path.join(out_vid_path, "results.txt")
    with open(results_file_path, 'w') as results_file:
        # List all files in the folder_path
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        # Process each file
        for file in tqdm(files):
            video_path = os.path.join(folder_path, file)
            if not (video_path.endswith('.mp4') or video_path.endswith('.avi')):  # check for video files
                continue

            # Capture video
            cap = cv2.VideoCapture(video_path)
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            frame_rate = int(cap.get(5))

            # Prepare output video writer if output_folder is specified
            if output_folder:
                output_path = os.path.join(out_vid_path, f'result_{file}')
                out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate // 1, (frame_width, frame_height))

                if lowfpsvid:
                    output_path_low = os.path.join(out_vid_path, f'lowfpsresult_{file}')
                    outlow = cv2.VideoWriter(output_path_low, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate // 4, (frame_width, frame_height))

            frameno = 0

            while cap.isOpened():
                try:
                    ret, frame = cap.read()
                    if not ret or frameno == 668:
                        break

                    frameno += 1
                    print(f"[INFO] {datetime.datetime.now()}: working with frame: {frameno} ")
                    # Apply detection model on the frame

                    st = time.time()
                    boxes, class_names, scores = model(frame)
                    print(f"[INFO] {datetime.datetime.now()}: time taken for text detection {time.time() - st } ")

                    ### filtering and plotting results on image
                    res_img = plot_results(frame, boxes, class_names, scores, det_th, classes, ocr_model, out_crop_path, file[:-4] + f'_frame_{frameno}', results_file)
                    # res_img = cv2.putText(res_img, f"frame:{frameno}", (500,500), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)

                    if output_folder:
                        out.write(res_img)  # Save processed frame
                        if lowfpsvid:
                            outlow.write(res_img)
                except Exception as e:
                    cap.release()
                    out.release()
                    if lowfpsvid:
                        outlow.release()
                    exit()

            # Release resources
            cap.release()
            if output_folder:
                out.release()
                if lowfpsvid:
                    outlow.release()
                print(f"[INFO] {datetime.datetime.now()}: result video saved at {output_path}\n ")

    print(f"[INFO] {datetime.datetime.now()}: --- VIDEO INFERENCING COMPLETED ---")


def main(params):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] {datetime.datetime.now()}: device available: {device}  ------xxxxxxxxx \n")

    ### checking directory
    dirchecks(params["image_dir"] if params["image_dir"] is not None else params["video_dir"])

    ### loading the model
    if params["use_model"] == "fasterrcnn":
        model = FasterRCNN(model_weights=params["models"]["fasterrcnn"]["model_weights"], classes=params["classes"], device=device, detection_thr=params["models"]["fasterrcnn"]["det_th"])
        detection_thr = params["models"]["fasterrcnn"]["det_th"]
    else:
        model = None

    print(f"[INFO] {datetime.datetime.now()}: Text Detection Model Loading Completed!!!\n")

    # if params["use_ocr_model"] == "trocr":
    #     ocr_model = TROCR(model_weights=params["ocr_models"]["trocr"]["model_weights"], MODEL_NAME=params["ocr_models"]["trocr"]["MODEL_NAME"])
    if params["use_ocr_model"] == "paddleocr":
        ocr_model = PaddleOCRx(model_weights=params["ocr_models"]["paddleocr"]["model_weights"])
    else:
        ocr_model = None

    print(f"[INFO] {datetime.datetime.now()}: OCR Model Loading Completed!!!\n" if params["use_ocr_model"] is not None else f"[INFO] {datetime.datetime.now()}: NO OCR model!!! working with text detection only \n")

    ##### ------------------------ INFER ON IMAGES OR VIDS -------------------
    if params["image_dir"] is not None:  ### if image run this
        start = time.time()
        img_inferencing(params["image_dir"], out_path=params["output_dir"], ocr_model=ocr_model, model=model, det_th=detection_thr, custom_name=params["custom_name"], classes=params["classes"])
        print(f"total time taken: {time.time() - start}")
    elif params["video_dir"] is not None:
        vid_inferencing(params["video_dir"], params["output_dir"], model, detection_thr, params["custom_name"], params["classes"], ocr_model=ocr_model, lowfpsvid=params["lowfpsvid"])
    else:
        print(f"[INFO] {datetime.datetime.now()}: no img path or vid path given. Exiting\n")
        sys.exit(1)


# Driver code
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
