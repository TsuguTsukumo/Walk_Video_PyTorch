import cv2
import numpy as np

# YOLOの設定ファイルと重みファイルのパス
config_path = 'yolov3.cfg'
weights_path = 'yolov3.weights'
classes_path = 'coco.names'

# クラス名の読み込み
with open(classes_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# ネットワークの読み込み
net = cv2.dnn.readNet(weights_path, config_path)

# 出力レイヤーの取得
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# カメラまたはビデオファイルのキャプチャ
cap = cv2.VideoCapture(0)  # カメラの場合
# cap = cv2.VideoCapture('video.mp4')  # ビデオファイルの場合

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 画像の前処理
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == 'person':
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Image', frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESCキーで終了
        break

cap.release()
cv2.destroyAllWindows()
