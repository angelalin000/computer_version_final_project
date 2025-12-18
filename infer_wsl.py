import libreface
import os
import time
import csv
from datetime import datetime

# 監控檔案列表（Windows -> WSL 路徑）
paths = ["/mnt/c/shared/frame_0.jpg", "/mnt/c/shared/frame_1.jpg"]
# 記錄每張圖片上次處理的修改時間
last_processed = {p: 0 for p in paths}

# CSV 輸出路徑
csv_file = "/mnt/c/shared/facial_attributes.csv"
au = [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]
# 欄位名稱
aus = [f"au_{i}" for i in au]
au_intensities = [f"au_{i}_intensity" for i in au]
# pose = ["pitch", "yaw", "roll"]
facial_expression = ["facial_expression"]

all_fields = ["timestamp"] + facial_expression + aus + au_intensities

# 如果 CSV 不存在，先建立
if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(all_fields)

print("WSL + Windows real-time inference started. Ctrl+C to stop.")

while True:
    for p in paths:
        if not os.path.exists(p):
            continue

        mtime = os.path.getmtime(p)
        if mtime <= last_processed[p]:
            continue  # 已處理過，不重複

        last_processed[p] = mtime  # 更新最後處理時間
        # print(f"Processing: {p}")

        try:
            res = libreface.get_facial_attributes_image(
                image_path=p,
                temp_dir="/tmp/libreface",
                device="cpu"
            )

            if res and len(res) > 0:
                f = res
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                row = [timestamp]

                detected_aus = f.get("detected_aus", {})

                # 規則判斷：AU4 + AU7 + AU24
                if (
                    (detected_aus.get("au_4", 0) == 1 and
                    detected_aus.get("au_7", 0) == 1 and
                    detected_aus.get("au_24", 0) == 1) or
                    (detected_aus.get("au_4", 0) == 1 and
                    detected_aus.get("au_24", 0) == 1)or
                    (detected_aus.get("au_4", 0) == 1 and
                    detected_aus.get("au_7", 0) == 1)or
                    (detected_aus.get("au_24", 0) == 1 and
                    detected_aus.get("au_7", 0) == 1)
                ):
                    facial_expression = "uncomfortable"
                else:
                    facial_expression = f.get("facial_expression")
                
                # facial_expression（已被你自訂規則覆寫）
                row.append(facial_expression)
                print(f"Logged: Facial Expression={facial_expression}")

                # detected_aus
                for au in aus:
                    row.append(f.get("detected_aus", {}).get(au, 0))

                # AU intensities
                for au_int in au_intensities:
                    row.append(f.get("au_intensities", {}).get(au_int, 0.0))

                # pose
                # row += [f.get("pitch", 0.0), f.get("yaw", 0.0), f.get("roll", 0.0)]

                # # landmarks
                # for i in range(68):
                #     lm = f.get("lm_mp", {}).get(str(i), {})
                #     row += [lm.get("x", 0.0), lm.get("y", 0.0), lm.get("z", 0.0)]

                # 寫入 CSV
                with open(csv_file, "a", newline="") as f_csv:
                    writer = csv.writer(f_csv)
                    writer.writerow(row)

                # print(f"Logged: Facial Expression={f.get('facial_expression')}")

            else:
                print(f"No face detected in {p}")

        except Exception as e:
            print(f"Inference error for {p}: {e}")

    time.sleep(0.05)
