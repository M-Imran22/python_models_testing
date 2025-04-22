from pyzbar.pyzbar import decode
from PIL import Image

# Load the image
image_path = "cakes_barcode.png"  # Replace with your image file
image = Image.open(image_path)

# Decode barcodes in the image
barcodes = decode(image)

# Process the results
if barcodes:
    for barcode in barcodes:
        barcode_type = barcode.type  # e.g., 'EAN13', 'UPCA'
        # The barcode number as a string
        barcode_data = barcode.data.decode("utf-8")
        print(f"Barcode Type: {barcode_type}, Data: {barcode_data}")
else:
    print("No barcodes found in the image.")
