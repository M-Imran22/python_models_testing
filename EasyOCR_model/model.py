import easyocr
import matplotlib.pyplot as plt
import cv2


def preprocess(img_path):
    img = cv2.imread(img_path)
    # 1. Crop to the label region if possible (helps focus the OCR)
    #    You can hard‑code a rough ROI or use OpenCV-based edge detection.
    #    For now we’ll skip cropping.

    # 2. Resize to make text larger
    h, w = img.shape[:2]
    img = cv2.resize(img, (w*2, h*2), interpolation=cv2.INTER_CUBIC)

    # 3. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 4. Contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 5. Denoise
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    # 6. (Optional) Adaptive thresholding
    # gray = cv2.adaptiveThreshold(gray, 255,
    #                             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                             cv2.THRESH_BINARY, 31, 2)

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


# Initialize once
reader = easyocr.Reader(['en'], gpu=False)

# Preprocess & OCR
img_rgb = preprocess('images/Tropicana-Juice-Smooth_007.jpg')
results = reader.readtext(
    img_rgb,
    min_size=20,           # ignore tiny text
    text_threshold=0.3,    # lower so weak text isn’t discarded
    link_threshold=0.35,   # for grouping characters into words
    contrast_ths=0.05,     # tune contrast sensitivity
    adjust_contrast=0.7,   # boost contrast further internally
)

# Visualize & print
plt.figure(figsize=(6, 6))
plt.imshow(img_rgb)
for bbox, text, prob in results:
    pts = bbox + [bbox[0]]
    xs, ys = zip(*pts)
    plt.plot(xs, ys, 'g-')
    x, y = bbox[0]
    plt.text(x, y-5, f"{text} ({prob:.2f})",
             color='yellow',
             bbox=dict(facecolor='black', alpha=0.6, pad=2))
plt.axis('off')
plt.show()

print("Filtered Results:")
for _, text, prob in results:
    if prob > 0.4:
        print(f"» {text!r} @ {prob:.2f}")

# 2nd method using Pillow

# import easyocr
# from PIL import Image

# # Initialize reader (English-only for speed)
# reader = easyocr.Reader(["en"], gpu=False)  # Use GPU=True if available

# # Load image (replace with your product label image)
# image_path = "images/Arla-Sour-Cream_006.jpg"
# image = Image.open(image_path)

# # Perform OCR
# # detail=0 returns only text
# results = reader.readtext(image_path, text_threshold=0.7, detail=0)

# print("Extracted Text:")
# for text in results:
#     print(text)
