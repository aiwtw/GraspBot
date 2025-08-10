import json
import google.generativeai as genai
from PIL import Image  

from yolov5.predict import YOLODetector 
import torch
import numpy as np
import sys
import cv2
from calib import CoordinateTransformer
import lib.magician.DobotDllType as dType
import time

# =================================================================
# ======= EXTENSION SWITCH (Set to True to enable Gemini API) =======
USE_GEMINI_EXTENSION = True
# =================================================================

# --- Gemini API Configuration ---
if USE_GEMINI_EXTENSION:
    try:
        # Read the API key from the 'api_key.txt' file
        with open('api_key.txt', 'r') as f:
            GOOGLE_API_KEY = f.readline().strip()
        
        if not GOOGLE_API_KEY:
            raise ValueError("Error: 'api_key.txt' is empty or the first line is blank.")

        genai.configure(api_key=GOOGLE_API_KEY)
        
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        print("Gemini API configured successfully.")
        
    except FileNotFoundError:
        print("Error: 'api_key.txt' not found. Please create it and add your API key.")
        USE_GEMINI_EXTENSION = False
        print("Gemini extension has been disabled.")
    except Exception as e:
        print(f"Failed to configure Gemini API: {e}")
        USE_GEMINI_EXTENSION = False
        print("Gemini extension has been disabled.")

def get_targets_from_gemini_vision(instruction, image_frame, available_cubes):
    """
    Uses Gemini Vision API to analyze an image and a text instruction to determine a pick-and-place plan.
    Args:
        instruction (str): The user's natural language command.
        image_frame (np.array): The camera image frame (in OpenCV BGR format).
        available_cubes (list): A list of cube colors detected by YOLOv5.
    Returns:
        list: A list of color strings in the order they should be picked. Returns an empty list on failure.
    """
    valid_colors = ["green", "yellow", "black"] 
    
    prompt = f"""
    You are an intelligent control system for a robot arm. Your task is to analyze an image of a scene and a user's instruction, 
    then create a precise plan for the robot to pick up colored blocks.

    **Inputs:**
    1.  **Image:** An image showing several colored blocks on a surface.
    2.  **User Command:** "{instruction}"
    3.  **Available Blocks:** The object detection system (YOLOv5) has confirmed the presence of the following blocks: 
    {available_cubes}. You should only consider these as potential targets.

    **Your Goal:**
    Based on the user's command and the visual evidence in the image, determine which blocks to pick and in what order. 
    Consider spatial relationships (e.g., 'left', 'right', 'between'), sequences (e.g., 'first... then...'), and logic 
    (e.g., 'all except...').

    **Output Format:**
    You MUST respond with a JSON-formatted list of strings. Each string must be the color of a block to pick, in the correct sequence.
    - The colors must be from this list of valid colors: {valid_colors}.
    - For example, If the command is to pick a green block then a yellow one, your output should be: ["green", "yellow"]
    - If the command is ambiguous, impossible, or requires no action, return an empty list: []

    Do not add any explanations or introductory text. Your response must be only the JSON list.
    """

    print("\n--- Calling Gemini Vision API ---")
    print(f"User Command: {instruction}")
    print(f"Available Cubes (from YOLO): {available_cubes}")

    try:
        rgb_frame = cv2.cvtColor(image_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        response = gemini_model.generate_content([prompt, pil_image])
        
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
        print(f"Gemini's Raw Response: {cleaned_response}")

        target_list = json.loads(cleaned_response)

        if not isinstance(target_list, list):
            print("Error: Gemini did not return a list.")
            return []

        valid_targets = [target for target in target_list if target in available_cubes]
        if len(valid_targets) != len(target_list):
            print(f"Warning: Gemini suggested a block not in the available list. Filtering to valid targets.")

        print(f"Gemini's Final Plan: {valid_targets}")
        return valid_targets

    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from Gemini's response.")
        return []
    except Exception as e:
        print(f"An error occurred while calling the Gemini API: {e}")
        return []


CON_STR = {
    dType.DobotConnect.DobotConnect_NoError:  "DobotConnect_NoError",
    dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
    dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied"}

if __name__ == '__main__':
    detector = YOLODetector(
        weights='./yolov5/runs/train/exp5/weights/best.pt',
        data='./yolov5/dataset/data.yaml',
        device=''                                    
    )
    cube_pos = {}
    cube_robo_pos = {}
    cap = cv2.VideoCapture(0) 

    last_frame = None 
    print("Camera started. Position the cubes and press 'q' to capture the scene and give a command.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        last_frame = frame.copy()
            
        detections = detector.detect(frame)
        
        current_frame_cubes = {}
        for det in detections:
            x1, y1, x2, y2, confidence, class_id, class_name = det
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{class_name} {confidence:.2f}'
            current_frame_cubes[class_name] = ((x1 + x2) // 2, (y1 + y2) // 2)

            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - label_height - baseline), (x1 + label_width, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        cube_pos = current_frame_cubes

        cv2.imshow('YOLOv5 Real-Time Detection', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("Scene captured.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if last_frame is None:
        print("No frame was captured from the camera. Exiting.")
        sys.exit()

    if not cube_pos:
        print("No cubes were detected in the final scene. Exiting.")
        sys.exit()

    # --- Post-Capture Processing ---
    transformer = CoordinateTransformer()
    for key, value in cube_pos.items():
        cube_robo_pos[key] = transformer.transform_pixel_to_robot(value)
    
    print("\nDetection complete. Blocks available in the scene:")
    available_colors = list(cube_robo_pos.keys())
    print(available_colors)

    lang = input("\nPlease enter your command (e.g., 'pick up the cube on the left'): ").strip()
    
    targets_to_pick = []
    
    if USE_GEMINI_EXTENSION:
        targets_to_pick = get_targets_from_gemini_vision(lang, last_frame, available_colors)
    else:
        print("Using basic keyword matching (Gemini extension is disabled)...")
        for color in available_colors:
            if color in lang:
                targets_to_pick.append(color)
        print(f"Matched targets: {targets_to_pick}")

    if not targets_to_pick:
        print("No targets were identified to pick up. The program will now exit.")
        sys.exit()

    # --- Robot Arm Control ---
    api = dType.load()
    state = dType.ConnectDobot(api, "", 115200)[0]
    print("Dobot Connect status:", CON_STR[state])

    if state == dType.DobotConnect.DobotConnect_NoError:
        dType.SetQueuedCmdClear(api)
        dType.SetHOMEParams(api, -0.4019, 149.6464, 12.2097, 90.1539, isQueued=1)
        dType.SetHOMECmd(api, temp=0, isQueued=1)

        print(f"\nExecuting plan: Picking up {targets_to_pick} in order.")
        for i, target_color in enumerate(targets_to_pick):
            print(f"--- Task {i+1}/{len(targets_to_pick)}: Picking up the {target_color} block ---")
            value = cube_robo_pos[target_color]

            dType.SetEndEffectorSuctionCup(api, 1, 1, isQueued=1)
            dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, value[0], value[1], 5, 85, isQueued=1) 
            dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, value[0], value[1], -56, 85, isQueued=1)
            dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, value[0], value[1], 5, 85, isQueued=1)
            
            # Move to a consistent drop-off location
            dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, 273.1274, -10.3756, 55, -2.1755, isQueued=1)
            dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, 273.1274, -10.3756, 50, -2.1755, isQueued=1) 
            dType.SetEndEffectorSuctionCup(api, 0, 1, isQueued=1)
            
            # Set the last command index on the final action of the sequence
            if i == len(targets_to_pick) - 1:
                lastIndex = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, 273.1274, -10.3756, 55, -2.1755, isQueued=1)[0]
            else:
                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, 273.1274, -10.3756, 55, -2.1755, isQueued=1)

        dType.SetQueuedCmdStartExec(api)

        # Wait for the entire queue to finish
        while lastIndex > dType.GetQueuedCmdCurrentIndex(api)[0]:
            time.sleep(1) 

        dType.SetQueuedCmdStopExec(api)
        print("\nAll tasks completed successfully.")

    dType.DisconnectDobot(api)