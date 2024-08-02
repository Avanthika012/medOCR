import functools
print = functools.partial(print, flush=True)
import os
import cv2
import sys
import time, datetime
import logging
import threading
import numpy as np
from ctypes import *
from dotenv import load_dotenv
from logging.handlers import RotatingFileHandler
# from pyzbar.pyzbar import decode
from ping3 import ping


MVS_DIR = "/opt/MVS/Samples/64/Python/MvImport/"
sys.path.insert(1, MVS_DIR)
# QUEUE = Queue(maxsize=10)

from MvCameraControl_class import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False  # wont show output on console
formatter = logging.Formatter(
    '%(asctime)s [%(levelname)8s:%(name)1s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
os.makedirs("./camera_logs/", exist_ok=True)
handler = RotatingFileHandler("./camera_logs/camera.log",
                              maxBytes=10000000,
                              backupCount=5,)

handler.setFormatter(formatter)
logger.addHandler(handler)

script_dir = os.path.dirname(os.path.abspath(__file__))
env_file_path = os.path.join(script_dir, '..', '.env')

load_dotenv(env_file_path)
time.sleep(5)

# Set up the WebSocket connection to the Node.js server
# server_url = f"ws://{os.environ.get('BASE_IP_ADDRESS')}:9000/qrCapture"

assigned_camera_ip = sys.argv[1]

ping_timeout = 0.2
next_ping_time = time.time()
time_between_restarts = 1

def is_camera_reachable(camera_ip, timeout=0.2):
    result = ping(camera_ip, timeout=timeout)
    return True if result is not None else False

def work_thread(cam=0, pData=0, nDataSize=0, g_bExit=False):
    stFrameInfo = MV_FRAME_OUT_INFO_EX()
    global sender,next_ping_time#,QUEUE
    memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
   
    while True:
        current_time = time.time()
        if current_time >= next_ping_time:
            camera_status = is_camera_reachable(assigned_camera_ip, timeout=ping_timeout)
            if camera_status == False:
                print("Unable to ping camera")
                logger.info("Unable to ping camera")
                g_bExit = True
            next_ping_time = current_time + 5

        ret = cam.MV_CC_GetOneFrameTimeout(byref(pData), nDataSize, stFrameInfo, 1000)
        if ret!=0:
            logger.info("no data[0x%x]" % ret)
            print("no data[0x%x]" % ret)
        else:
            numpy_array = np.array(pData, dtype=np.uint8)
            image = numpy_array.reshape(
                stFrameInfo.nHeight, stFrameInfo.nWidth)
            try:
                # ADD IMAGE "image" TO QUEUE HERE
                print("chal rha, frame aya")
            except Exception as e:
                print("nhi gaya ",e)

        if g_bExit == True:
            print("Breaking from outer loop")
            logger.info("Breaking from outer loop")
            break

if __name__ == "__main__":
    try:
        g_bExit = False
        isThreadCreated = False
        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
        cam = None
        hThreadHandle = None
        nConnectionNum = -1

        while True:
            # Check if the thread has stopped
            if (hThreadHandle is None) or (not hThreadHandle.is_alive()):
                # Enum device
                if isThreadCreated == True:
                    hThreadHandle.join()
                    nConnectionNum = -1
                    # Stop grab image
                    cam.MV_CC_StopGrabbing()
                    # Close device
                    cam.MV_CC_CloseDevice()
                    # Destroy handle
                    cam.MV_CC_DestroyHandle()

                ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)

                if deviceList.nDeviceNum == 0:
                    # print("No device found")
                    continue

                for i in range(0, deviceList.nDeviceNum):
                    mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(
                        MV_CC_DEVICE_INFO)).contents
                    if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
                        nip1 = (
                            (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
                        nip2 = (
                            (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
                        nip3 = (
                            (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
                        nip4 = (
                            mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
                        composed_ip = f"{nip1}.{nip2}.{nip3}.{nip4}"
                        logger.info(f"Found camera--{composed_ip}")
                        print(f"Found camera--{composed_ip}")
                        if composed_ip == assigned_camera_ip:
                            nConnectionNum = i

                if nConnectionNum == -1:
                    print("Not my camera")
                    continue

                if deviceList.nDeviceNum > 0:
                    if int(nConnectionNum) >= deviceList.nDeviceNum:
                        print("intput error!")
                        logger.info("intput error!")
                        sys.exit()

                    # Creat Camera Object
                    cam = MvCamera()

                    # Select device and create handle
                    stDeviceList = cast(deviceList.pDeviceInfo[int(
                        nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents

                    ret = cam.MV_CC_CreateHandle(stDeviceList)
                    if ret != 0:
                        print("create handle fail! ret[0x%x]" % ret)
                        logger.info("create handle fail! ret[0x%x]" % ret)
                        sys.exit()

                    # Open device
                    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
                    if ret != 0:
                        print("open device fail! ret[0x%x]" % ret)
                        logger.info("open device fail! ret[0x%x]" % ret)
                        sys.exit()

                    # Detection network optimal package size(It only works for the GigE camera)
                    if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
                        nPacketSize = cam.MV_CC_GetOptimalPacketSize()
                        if int(nPacketSize) > 0:
                            ret = cam.MV_CC_SetIntValue(
                                "GevSCPSPacketSize", nPacketSize)
                            if ret != 0:
                                print(
                                    "Warning: Set Packet Size fail! ret[0x%x]" % ret)
                                logger.info(
                                    "Warning: Set Packet Size fail! ret[0x%x]" % ret)
                        else:
                            print(
                                "Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)
                            logger.info(
                                "Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)
                    
                    # Set trigger mode as off
                    ret = cam.MV_CC_SetEnumValue(
                        "TriggerMode", MV_TRIGGER_MODE_ON)
                    if ret != 0:
                        print("set trigger mode fail! ret[0x%x]" % ret)
                        logger.info("set trigger mode fail! ret[0x%x]" % ret)
                        sys.exit()

                    cam.MV_CC_SetEnumValueByString("TriggerSource", "Line0")
                    ret = cam.MV_CC_SetEnumValueByString("TriggerActivation", "RisingEdge")
                    if ret != 0:
                        print("set trigger mode fail! ret[0x%x]" % ret)
                        logger.info("set trigger mode fail! ret[0x%x]" % ret)
                        sys.exit()

                    # Set trigger delay and debounce time
                    ret = cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", False)
                    if ret != 0:
                        print("set acquisition frame rate enable fail! ret[0x%x]" % ret)
                        logger.info("set acquisition frame rate enable fail! ret[0x%x]" % ret)
                        sys.exit()

                    # Get payload size
                    expParam = MVCC_INTVALUE()
                    memset(byref(expParam), 0, sizeof(MVCC_INTVALUE))

                    # Get payload size
                    stParam = MVCC_INTVALUE()
                    memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))

                    ret = cam.MV_CC_GetIntValue("PayloadSize", stParam)
                    if ret != 0:
                        print("get payload size fail! ret[0x%x]" % ret)
                        logger.info("get payload size fail! ret[0x%x]" % ret)
                        sys.exit()
                    nPayloadSize = stParam.nCurValue

                    # Start grab image
                    ret = cam.MV_CC_StartGrabbing()
                    if ret != 0:
                        print("start grabbing fail! ret[0x%x]" % ret)
                        logger.info("start grabbing fail! ret[0x%x]" % ret)
                        sys.exit()

                    data_buf = (c_ubyte * nPayloadSize)()

                    try:
                        print("Creating thread\n")
                        logger.info("Creating thread\n")
                        hThreadHandle = threading.Thread(
                            target=work_thread, args=(cam, data_buf, nPayloadSize, g_bExit))
                        hThreadHandle.start()
                        isThreadCreated = True
                    except:
                        print("error: unable to start thread")
                        logger.info("error: unable to start thread")
                        isThreadCreated = False

    except KeyboardInterrupt:
        print("Keyboard Interrupt")
        logger.info("Keyboard Interrupt")
        g_bExit = True

    except Exception as error:
        print("An error occurred:", error)