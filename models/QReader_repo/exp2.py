from QReader import qreader
import cv2


# Create a QReader instance

qreader = qreader.QReader(model_size = 'n')

# Get the image that contains the QR code
image = cv2.cvtColor(cv2.imread("/home/frinksserver/Downloads/IMG-20240801-WA0014.jpg"), cv2.COLOR_BGR2RGB)
# image = image[:570, :].copy()

# print(image.dtype, image.shape)
import time
# Use the detect_and_decode function to get the decoded QR data
st= time.time()
for _ in range(10):
    t1 = time.time()
    decoded_text = qreader.detect_and_decode(image=image)
    print(time.time() - t1)

print(decoded_text)
print(time.time()-st)