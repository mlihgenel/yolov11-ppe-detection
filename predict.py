from ultralytics import YOLO

# Modeli yükle
model = YOLO("runs/detect/train/weights/best.pt")

# VIDEO ÜZERİNDEN NESNE TESPİTİ 
# ===================================================================

# results = model.predict(
#     source="video2.mp4",  
#     conf=0.25,          
#     save=True,          
#     project="video_results",     
#     name="ppe_video_test", 
# )

# print("Tahminler kaydedildi:", results)

# RESİM DOSYALARI ÜZERİNDEN NESNE TESPİTİ 
# ===================================================================
results = model.predict(
    source="images/",
    conf=0.25,
    save=True,
    project="images/predictions",
    name="ppe_img_test"
)

import matplotlib.pyplot as plt
import os
import cv2

# Tahminlerin kaydedildiği klasör
pred_dir = "images/predictions/ppe_img_test"

# Klasördeki resimleri listele
image_files = [os.path.join(pred_dir, f) for f in os.listdir(pred_dir) if f.endswith((".jpg", ".png"))]

# Figure boyutunu ayarla (kaç resim olduğuna göre değişebilir)
plt.figure(figsize=(15, 10))

for i, img_path in enumerate(image_files):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV BGR -> RGB dönüşümü
    
    # subplot oluştur (örn. 3 sütun olacak şekilde)
    plt.subplot((len(image_files) + 2) // 3, 3, i + 1)
    plt.imshow(img)
    plt.axis("off")

plt.tight_layout()
plt.savefig("all_results.png")
plt.show()

print("Tahminlet kaydedildi: ", results)
