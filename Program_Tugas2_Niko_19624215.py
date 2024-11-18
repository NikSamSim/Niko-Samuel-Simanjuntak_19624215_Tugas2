import cv2

# Membuka video file
video = cv2.VideoCapture('object_video.mp4')
if not video.isOpened():
    print("Error: Unable to open video file.")
    exit()

# Mengatur ukuran tampilan
display_width = 960
display_height = 540

# Rentang warna merah dalam HSV
red_lower_range1 = (0, 120, 70)
red_upper_range1 = (10, 255, 255)
red_lower_range2 = (170, 120, 70)
red_upper_range2 = (180, 255, 255)

while video.isOpened():
    ret, frame_original = video.read()
    if not ret:
        break

    # Konversi frame ke HSV
    frame_hsv = cv2.cvtColor(frame_original, cv2.COLOR_BGR2HSV)

    # Masking warna merah
    red_mask1 = cv2.inRange(frame_hsv, red_lower_range1, red_upper_range1)
    red_mask2 = cv2.inRange(frame_hsv, red_lower_range2, red_upper_range2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Membersihkan noise
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, morph_kernel)

    # Deteksi kontur
    detected_contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in detected_contours:
        # Filter area kontur
        contour_area = cv2.contourArea(contour)
        if contour_area > 500:
            x, y, width, height = cv2.boundingRect(contour)
            cv2.rectangle(frame_original, (x, y), (x + width, y + height), (255, 0, 0), 2)
            cv2.putText(frame_original, "Target", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Ubah ukuran frame untuk tampilan
    resized_frame = cv2.resize(frame_original, (display_width, display_height))
    resized_mask = cv2.resize(clean_mask, (display_width, display_height))

    # Menampilkan hasil
    cv2.imshow('Detected Targets', resized_frame)
    cv2.imshow('Red Mask', resized_mask)

    # Keluar dengan Enter (13) atau Spasi (32)
    key = cv2.waitKey(1)
    if key == 13 or key == 32:  # Enter atau Spasi
        break

# Release resources
video.release()
cv2.destroyAllWindows()