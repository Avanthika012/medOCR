import os
import sys
import cv2
import time
import threading
import numpy as np
from ctypes import *
from ping3 import ping

MVS_DIR = "/opt/MVS/Samples/64/Python/MvImport/"

sys.path.append(MVS_DIR)
from MvCameraControl_class import *

import queue

class Transmittor:
    def __init__(self, camera_ip, queueSize=128):
        self.saveQueue = queue.Queue(maxsize=queueSize)
        self.assigned_camera_ip = camera_ip
        self.stop = False

        self.cam = None
        self.model_update = False
        self.model_update_data = None
        self.ping_timeout_retries = 5
        self.ping_timeout = 0.2
        self.next_ping_time = time.time()
        self.time_between_restarts = 1

        os.makedirs('./frames/', exist_ok=True)

        # self.temp_queue = []

        
    def read(self):
        ct = 1
        image_res = []
        if self.saveQueue.qsize()>0:
            
            while (ct > 0 and self.saveQueue.qsize() > 0):
                # loggerObj.saveQueue_logger.info('Queu size is greater than 0 so reading from it ')
                if self.stop:
                    break
                image_res.append(self.saveQueue.get())
                ct -= 1
            return image_res
        else:
            time.sleep(0.001)
            return image_res

    def queue_thread(self):
        print("starting saving")
        # global temp_queue
        while True:
            if self.saveQueue.qsize()>0:
                st = time.time()
                tempItem=self.saveQueue.get()
                # print("time taken for getting : ", time.time() - st)
                
                cv2.imwrite(f'./frames/{int(time.time()*1000)}.png',tempItem["frame"])
                print("Saved frame: ", int(time.time()*1000), self.saveQueue.qsize())
            else:
                time.sleep(0.0001)

    def is_camera_reachable(self, timeout=0.2):
        result = ping(self.assigned_camera_ip, timeout=timeout)
        return True if result is not None else False
    
    def read(self):
        ct = 1
        image_res = []
        if self.saveQueue.qsize()>0:
            
            while (ct > 0 and self.saveQueue.qsize() > 0):
                # loggerObj.saveQueue_logger.info('Queu size is greater than 0 so reading from it ')
                if self.stop:
                    break
                image_res.append(self.saveQueue.get())
                ct -= 1
            return image_res
        else:
            time.sleep(0.001)
            return image_res

    def work_thread(self, cam, data_buf=0, nPayloadSize=0, g_bExit=False):
        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
        ping_retries=0

        while True:
            current_time = time.time()
            if current_time >= self.next_ping_time:
                camera_status = self.is_camera_reachable()
                if camera_status == False:
                    print("Unable to ping camera")
                    ping_retries=ping_retries+1
                    if ping_retries>self.ping_timeout_retries:
                        self.stop=True
                else:
                    ping_retries=0
                
                self.next_ping_time = current_time + 5

            ret = cam.MV_CC_GetOneFrameTimeout(byref(data_buf), nPayloadSize, stFrameInfo, 1000)
            if ret!=0:
                time.sleep(0.001)
                print("no data[0x%x]" % ret)
            else:
                numpy_array = np.array(data_buf, dtype=np.uint8)
                image = numpy_array.reshape(
                    stFrameInfo.nHeight, stFrameInfo.nWidth)
                try:
                    # ADD IMAGE "image" TO QUEUE HERE
                    if self.saveQueue.qsize() != 128:
                        self.saveQueue.put(image)
                    else: 
                        time.sleep(0.001)
                        pass
                except Exception as e:
                    print("Got error while pushing to queue: ",e)

            if self.stop:
                print("Breaking from work thread")
                break

    def main_thread(self):
        try:
            g_bExit = False
            isThreadCreated = False
            deviceList = MV_CC_DEVICE_INFO_LIST()
            tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
            hThreadHandle = None
            nConnectionNum = -1
            # save_thread = threading.Thread(target=self.queue_thread)
            # # loggerObj.sender_logger.info(f'New ImageServer thread created ')
            # save_thread.daemon = True
            # save_thread.start()
            self.cam = None
            while True:
                    # if self.stop:
                    #     if hThreadHandle is not None:
                    #         hThreadHandle.join()
                    #         break

                # Check if the thread has stopped
                # if (hThreadHandle is None) or (not hThreadHandle.is_alive()):
                    # Enum device
                    # if isThreadCreated == True:
                    if self.cam is not None:
                        print("Killing thred")
                        # logger.info("Killing thred")
                        # hThreadHandle.join()
                        nConnectionNum = -1
                        # Stop grab image
                        self.cam.MV_CC_StopGrabbing()
                        # Close device
                        self.cam.MV_CC_CloseDevice()
                        # Destroy handle
                        self.cam.MV_CC_DestroyHandle()

                    ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)

                    if deviceList.nDeviceNum == 0:
                        # print("No device found")
                        continue

                    for i in range(0, deviceList.nDeviceNum):
                        print("Finding devices")
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
                            # logger.info(f"Found camera--{composed_ip}")
                            print(f"Found camera--{composed_ip}")
                            if composed_ip == self.assigned_camera_ip:
                                nConnectionNum = i

                    if nConnectionNum == -1:
                        continue

                    if deviceList.nDeviceNum > 0:
                        if int(nConnectionNum) >= deviceList.nDeviceNum:
                            print("intput error!")
                            continue

                        # Creat Camera Object
                        self.cam = MvCamera()

                        # Select device and create handle
                        stDeviceList = cast(deviceList.pDeviceInfo[int(
                            nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents

                        ret = self.cam.MV_CC_CreateHandle(stDeviceList)
                        if ret != 0:
                            print("create handle fail! ret[0x%x]" % ret)
                            # logger.info("create handle fail! ret")
                            continue

                        # Open device
                        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
                        if ret != 0:
                            print("open device fail! ret[0x%x]" % ret)
                            # logger.info("open device fail! ret")
                            continue

                        # Detection network optimal package size(It only works for the GigE self.camera)
                        if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
                            nPacketSize = self.cam.MV_CC_GetOptimalPacketSize()
                            if int(nPacketSize) > 0:
                                ret = self.cam.MV_CC_SetIntValue(
                                    "GevSCPSPacketSize", nPacketSize)
                                if ret != 0:
                                    print("Warning: Set Packet Size fail! ret[0x%x]" % ret)
                                    # logger.info("Warning: Set Packet Size fail! ret")
                            else:
                                print("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)
                                # logger.info("Warning: Get Packet Size fail! ret")

                        # Set trigger mode as off
                        ret = self.cam.MV_CC_SetEnumValue(
                        "TriggerMode", MV_TRIGGER_MODE_ON)
                        if ret != 0:
                            print("set trigger mode fail! ret[0x%x]" % ret)
                            sys.exit()

                        self.cam.MV_CC_SetEnumValueByString("TriggerSource", "Line0")
                        ret = self.cam.MV_CC_SetEnumValueByString("TriggerActivation", "RisingEdge")
                        if ret != 0:
                            print("set trigger mode fail! ret[0x%x]" % ret)
                            sys.exit()

                        # Set trigger delay and debounce time
                        ret = self.cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", False)
                        if ret != 0:
                            print("set acquisition frame rate enable fail! ret[0x%x]" % ret)
                            sys.exit()


                        # Get payload size
                        expParam = MVCC_INTVALUE()
                        memset(byref(expParam), 0, sizeof(MVCC_INTVALUE))

                        # Get payload size
                        stParam = MVCC_INTVALUE()
                        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))

                        ret = self.cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_Mono8)
                        if ret != 0:
                            print("Set Pixel format change failed! ret[0x%x]" % ret)
                            continue

                        ret = self.cam.MV_CC_GetIntValue("PayloadSize", stParam)
                        if ret != 0:
                            print("get payload size fail! ret[0x%x]" % ret)
                            continue
                        nPayloadSize = stParam.nCurValue

                        # Start grab image
                        ret = self.cam.MV_CC_StartGrabbing()
                        if ret != 0:
                            print("start grabbing fail! ret[0x%x]" % ret)
                            # logger.info("start grabbing fail! ret")
                            continue

                        data_buf = (c_ubyte * nPayloadSize)()

                        try:
                            print("Creating thread\n")
                            # logger.info("Creating thread\n")
                            # hThreadHandle = threading.Thread(
                            #     target=self.work_thread, args=(self.cam, data_buf, nPayloadSize, g_bExit))
                            # hThreadHandle.start()

                            self.work_thread(self.cam, data_buf, nPayloadSize, g_bExit)

                            isThreadCreated = True
                        except Exception as e:
                            print("error: unable to start thread", e)
                            # logger.info("error: unable to start thread")
                            isThreadCreated = False

        except KeyboardInterrupt:
            if g_bExit == False:
                g_bExit = True
            sys.exit()

    
    def run(self):
        self.mainThread = threading.Thread(target=self.main_thread)
        self.mainThread.start()

    def stop_stream(self):
        if not self.stop:
            self.stop = True

            self.mainThread.join()




if __name__ == "__main__":
    assigned_camera_ip = sys.argv[1]
    rpi_name = sys.argv[2]
    height = int(sys.argv[3])
    width = int(sys.argv[4])
    fps = float(sys.argv[5])
    pin = int(sys.argv[6])
    gain = None
    exposure = None
    gamma = None
    trans = Transmittor(camera_ip=assigned_camera_ip, rpi_name=rpi_name, height=height, width=width, fps=fps, pin=pin)

    trans.run()
