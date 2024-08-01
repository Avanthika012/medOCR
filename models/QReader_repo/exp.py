import cv2
from pyzbar.pyzbar import decode

def detect_and_decode_qr_code(image_path):
    # Load the image
    image = cv2.imread(image_path)
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Decode the QR code
    qr_codes = decode(gray_image)

    for qr_code in qr_codes:
        # Get the bounding box coordinates
        x, y, w, h = qr_code.rect
        # Draw a rectangle around the QR code
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Decode the QR code data
        qr_code_data = qr_code.data.decode("utf-8")
        print("Decoded QR code data:", qr_code_data)

        # Put the QR code data text on the image
        cv2.putText(image, qr_code_data, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the image with the detected QR code
    cv2.imshow("QR Code Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Path to the image containing the QR code
    image_path = "hotelqr.jpg"
    detect_and_decode_qr_code(image_path)
