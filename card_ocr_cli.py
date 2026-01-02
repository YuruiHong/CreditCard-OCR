import argparse
import pyperclip  # <--- 新加
from ultralytics import YOLO
from PIL import Image
from crnn.predict import predict
from process_result import get_info
from rectify import rectify
import os

def main(image_path):
    yolo = YOLO("./models/yolo_best.pt")
    try:
        image = rectify(image_path)
        image = image[:, :, ::-1]
        image = Image.fromarray(image)
    except Exception as e:
        print("矫正失败，直接处理原图。error:", e)
        image = Image.open(image_path)
    result = yolo(image)
    boxes = result[0].boxes.xyxy.to('cpu').numpy().astype(int)
    confidences = result[0].boxes.conf.to('cpu').numpy().astype(float)
    labels = result[0].boxes.cls.to('cpu').numpy().astype(int)

    info = {
        "filename": os.path.basename(image_path),
        "card_number": "",
        "valid_date": "",
        "bank_name": "",
        "card_type": "",
        "is_unionpay": ""
    }
    for box, conf, label in zip(boxes, confidences, labels):
        if conf > 0.5:
            x_min, y_min, x_max, y_max = box
            image_crop = image.crop((x_min, y_min, x_max, y_max))
            result_text = ''
            if label == 0 and conf > 0.55:
                result = predict(image_crop, category='card_number')
                for i in result[0]:
                    if i != '/':
                        result_text += i
                info["card_number"] = result_text
                processed_result = get_info(result_text)
                info["bank_name"] = processed_result[0]
                cardtype_map = {'DC': '储蓄卡', 'CC': '信用卡', 'SCC': '准贷记卡', 'PC': '预付费卡'}
                info["card_type"] = cardtype_map.get(processed_result[1], processed_result[1])
                info["is_unionpay"] = processed_result[2]
            elif label == 1:
                result = predict(image_crop, category='date')
                for i in result[0]:
                    result_text += i
                info["valid_date"] = result_text
            elif label == 2:
                info["is_unionpay"] = "是(图中存在)"
    # 输出结果
    for key in info:
        print(f"{key}: {info[key]}")
    # 复制卡号到剪贴板
    if info["card_number"]:
        pyperclip.copy(info["card_number"])
        print(f"卡号已复制到剪贴板: {info['card_number']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bank/Card OCR CLI")
    parser.add_argument("image", help="Input image path")
    args = parser.parse_args()
    main(args.image)