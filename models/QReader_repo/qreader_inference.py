from .QReader import qreader

import datetime
import time


class Qreaderxp:

    def __init__(self):
        self.qreaderx = qreader.QReader(model_size = 'n')

        print(f"[INFO] {datetime.datetime.now()}: Qreader loaded!!!\n")


    # Method to pass image through the model
    def forward(self, image):
        st = time.time()
        decoded_text = self.qreaderx.detect_and_decode(image=image)
        print(f"\n time for QR reader:{time.time()-st}")

        return decoded_text


    # Method to call for direct inferencing
    def __call__(self, image):
        predicted_text = self.forward(image)
        return predicted_text
