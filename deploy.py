from yolov5.predict import YOLODetector 
import torch
import numpy as np
import sys
import cv2
from calib import  CoordinateTransformer
import lib.magician.DobotDllType as dType
import time

CON_STR = {
    dType.DobotConnect.DobotConnect_NoError:  "DobotConnect_NoError",
    dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
    dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied"}

if __name__ == '__main__':
    back = False
    detector = YOLODetector(
        weights='./yolov5/runs/train/exp5/weights/best.pt',
        data='./yolov5//dataset/data.yaml',
        device=''                                    
    )
    cube_pos={}
    cube_robo_pos = {}
    cap = cv2.VideoCapture(0) 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        detections = detector.detect(frame)
        
        for det in detections:

            x1, y1, x2, y2, confidence, class_id, class_name = det
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f'{class_name} {confidence:.2f}'
            cube_pos[class_name]=((x1+x2)//2,(y1+y2)//2)

            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - label_height - baseline), (x1 + label_width, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        cv2.imshow('YOLOv5 Real-Time Detection', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

        # coordinate trans

    api = dType.load()
    state = dType.ConnectDobot(api, "", 115200)[0]
    print("Connect status:",CON_STR[state])

    if (state == dType.DobotConnect.DobotConnect_NoError):
        dType.SetQueuedCmdClear(api)
        dType.SetEndEffectorSuctionCup(api, 1, 1, isQueued = 1)
        dType.SetHOMEParams(api, -0.4019,149.6464,12.2097,90.1539, isQueued = 1)
        dType.SetHOMECmd(api, temp = 0, isQueued = 1)

    transformer = CoordinateTransformer()
    for key,value in cube_pos.items():
        cube_robo_pos[key] = transformer.transform_pixel_to_robot(value)

    lang = input("请输入语言指令（例如 'pick the green cube'）: ").strip()
    if not back:
        dType.SetQueuedCmdStopExec(api)
        dType.SetQueuedCmdClear(api)

    for key,value in cube_robo_pos.items():
        if lang.find(key) != -1:
            dType.SetEndEffectorSuctionCup(api, 1, 1, isQueued = 1)
            dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, value[0], value[1], -10, 85, isQueued=1)
            dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, value[0], value[1], -56, 85, isQueued=1)
            dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, value[0], value[1], -10, 85, isQueued=1)
            dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, value[0], value[1], 5, 85, isQueued=1)
            dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, 273.1274, -10.3756, 55, -2.1755, isQueued=1)
            dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, 273.1274, -10.3756, 50, -2.1755, isQueued=1)
            lastIndex = dType.SetEndEffectorSuctionCup(api, 0, 1, isQueued = 1)[0]
            dType.SetQueuedCmdStartExec(api)

            while lastIndex > dType.GetQueuedCmdCurrentIndex(api)[0]:
                dType.dSleep(1000)

            dType.SetQueuedCmdStopExec(api)
            dType.SetQueuedCmdClear(api)
            


    dType.DisconnectDobot(api)
    # 273.1274,-10.3756,50,-2.1755
