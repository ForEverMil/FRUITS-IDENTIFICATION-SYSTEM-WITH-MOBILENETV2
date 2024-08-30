from keras.models import load_model
import cv2
import numpy as np

# Vô hiệu hóa biểu diễn số học khoa học để dễ đọc
np.set_printoptions(suppress=True)

# Tải mô hình
model = load_model("model.h5", compile=False)
print("Mô hình được tải thành công")

# Tải nhãn
class_names = open("labels.txt", "r", encoding="utf-8").readlines()

# CAMERA có thể là 0 hoặc 1 dựa trên camera mặc định
camera = cv2.VideoCapture(0)
print("Kết nối với camera thành công")

# Định nghĩa hệ số tỉ lệ để mở rộng tầm nhìn của camera
scale_factor = 1

while True:
    # Chụp hình từ webcam
    ret, image = camera.read()

    # Điều chỉnh kích thước hình ảnh gốc thành kích thước (224-cao,224-rộng) pixel
    resized_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

    # Hiển thị hình ảnh trong một cửa sổ
    cv2.imshow("Hình ảnh từ Webcam", resized_image)

    # Chuyển hình ảnh thành mảng numpy và thay đổi kích thước thành hình dạng đầu vào của mô hình.
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Chuẩn hóa mảng hình ảnh
    image = (image / 127.5) - 1

    # Dự đoán từ mô hình
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # In ra dự đoán và điểm tin cậy
    print("Đây là quả:", class_name[2:], end="")
    print("Độ chính xác:", str(np.round(confidence_score * 100))[:-2], "%")

    # Lắng nghe đầu vào từ bàn phím
    keyboard_input = cv2.waitKey(1)

    # Kiểm tra xem phím 'q' đã được nhấn chưa
    if keyboard_input & 0xFF == ord('q'):
        break

# Giải phóng camera và đóng cửa sổ OpenCV
camera.release()
cv2.destroyAllWindows()
