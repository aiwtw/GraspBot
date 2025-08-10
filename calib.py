import cv2
import numpy as np
import os
import sys


TRANSFORM_MATRIX_FILE = 'transform_matrix.npy'

def calibrate_and_save(camera_index=0):
    pixel_points = []
    window_name = "Calibration: Click 3 points (A, B, C), then press 'c' to confirm"

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pixel_points) < 3:
            pixel_points.append((x, y))
            print(f"Point {len(pixel_points)}/3 registered at ({x}, {y}).")

    cap = cv2.VideoCapture(camera_index)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print("--- Starting Calibration ---")
    print("1. Click on 3 reference points in the window in a specific order (e.g., A, B, C).")
    print("2. After clicking 3 points, press the 'c' key to continue to coordinate input.")
    print("3. Press 'r' to reset points. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        display_frame = frame.copy()
        
        if len(pixel_points) < 3:
            text = f"Click point {len(pixel_points) + 1}/3"
        else:
            text = "3 points selected. Press 'c' to confirm."
        cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        for i, point in enumerate(pixel_points):
            cv2.circle(display_frame, point, 5, (0, 255, 0), -1)
            cv2.putText(display_frame, f"{i+1}", (point[0] + 10, point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow(window_name, display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            print("Calibration cancelled.")
            return
        elif key == ord('r'):
            pixel_points = []
            print("Points reset. Please click 3 points again.")
        elif key == ord('c') and len(pixel_points) == 3:
            break 

    cap.release()
    cv2.destroyAllWindows()
    
    if len(pixel_points) != 3:
        print("Calibration failed: Not enough points selected.")
        return

    print("\n--- Input Robot Coordinates ---")
    print("Please enter the robot coordinates for the 3 points you clicked, in the same order.")
    robot_points = []
    for i in range(3):
        while True:
            try:
                coord_str = input(f"Enter robot coords for Point {i+1} (format: x,y): ")
                x_str, y_str = coord_str.split(',')
                robot_points.append((float(x_str.strip()), float(y_str.strip())))
                break
            except ValueError:
                print("Invalid format. Please use 'x,y', e.g., 100.5,250.0")
                
    try:
        src_pts = np.float32(pixel_points)
        dst_pts = np.float32(robot_points)
        transform_matrix = cv2.getAffineTransform(src_pts, dst_pts)
        
        np.save(TRANSFORM_MATRIX_FILE, transform_matrix)
        print(f"\nTransformation matrix calculated and saved to '{TRANSFORM_MATRIX_FILE}'")
        print("Matrix content:\n", transform_matrix)
    except cv2.error as e:
        print(f"Error calculating transform matrix: {e}")
        print("The 3 points might be collinear. Please try again.")

class CoordinateTransformer:
    """
    A class to handle coordinate transformations using a pre-saved matrix.
    """
    def __init__(self):
        """Loads the transformation matrix upon initialization."""
        self.matrix = None
        if os.path.exists(TRANSFORM_MATRIX_FILE):
            self.matrix = np.load(TRANSFORM_MATRIX_FILE)
            print(f"'{TRANSFORM_MATRIX_FILE}' loaded successfully.")
        else:
            print(f"Error: Transformation matrix file '{TRANSFORM_MATRIX_FILE}' not found.")
            print("Please run the `calibrate_and_save()` function first.")
            sys.exit()

    def transform_pixel_to_robot(self, pixel_coord):
        """
        Transforms a single pixel coordinate to a robot coordinate.
        :param pixel_coord: A tuple (x, y) of the pixel coordinate.
        :return: A tuple (rx, ry) of the robot coordinate.
        """
        if self.matrix is None:
            raise Exception("Transformation matrix is not loaded.")
        
        # cv2.transform requires input in the shape (1, 1, 2)
        pixel_np = np.array([[pixel_coord]], dtype=np.float32)
        transformed_point = cv2.transform(pixel_np, self.matrix)
        
        return tuple(transformed_point[0][0])

    def run_interactive_mode(self, camera_index=0):
        """
        Starts an interactive session where the user can click on the camera feed
        to get real-time robot coordinates.
        """
        if self.matrix is None:
            print("Cannot run interactive mode without a loaded transformation matrix.")
            return

        window_name = "Interactive Mode (Click to get coords, 'q' to quit)"
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Error: Cannot open camera {camera_index}")
            return
            
        print("\n--- Starting Interactive Mode ---")
        print("Click anywhere in the window to see the corresponding robot coordinates.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow(window_name, frame)

            # Use waitKeyEx for better cross-platform mouse event handling inside a loop
            key = cv2.waitKeyEx(1)
            
            if key == ord('q'):
                break
            # 27 is the ESC key code
            elif key == 27:
                break

            def interactive_mouse_callback(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    robot_coord = self.transform_pixel_to_robot((x, y))
                    print(f"Pixel: ({x}, {y})  =>  Robot: ({robot_coord[0]:.2f}, {robot_coord[1]:.2f})")

            cv2.setMouseCallback(window_name, interactive_mouse_callback)

        cap.release()
        cv2.destroyAllWindows()
        print("Interactive mode finished.")


# ================== MAIN PROGRAM ==================
if __name__ == '__main__':
    print("### PART 1: CALIBRATION ###")
    calibrate_and_save()

    if not os.path.exists(TRANSFORM_MATRIX_FILE):
        print("\nCalibration did not produce a matrix file. Exiting test.")
    else:
        print("\n### PART 2: TESTING THE TRANSFORMER ###")
        transformer = CoordinateTransformer()

        print("\n--- Test Mode 1: Transform a specific pixel coordinate ---")
        test_pixel_coord = (320, 240) 
        robot_coord_mode1 = transformer.transform_pixel_to_robot(test_pixel_coord)
        print(f"Programmatically transforming pixel {test_pixel_coord}...")
        print(f"Resulting Robot Coordinate: {robot_coord_mode1}")
        
        print("\n--- Test Mode 2: Interactive clicking ---")
        print("The interactive window will now open.")
        print(f"Try to click near the pixel coordinate {test_pixel_coord} to see if the robot coordinate matches the result above.")
        transformer.run_interactive_mode()

        print("\n### TEST COMPLETE ###")