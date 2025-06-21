import cv2
import mediapipe as mp

def get_hand_bbox_and_width(image: cv2.Mat, hand_landmarks: mp.solutions.hands.HandLandmark) -> tuple:
    """
    Calculates the pixel width and screen coordinates of the bounding box around the hand.
    
    Args:
        image (cv2.Mat): The image in which the hand is detected.
        hand_landmarks (mp.solutions.hands.HandLandmark): The detected hand landmarks.
    
    Returns:
        tuple: A tuple containing the pixel width of the hand and the bounding box coordinates.
    """
    
    h, w, _ = image.shape
    x_coords = []
    y_coords = []

    if not hand_landmarks:
        return 0, None

    for landmark in hand_landmarks.landmark:
        x_coords.append(landmark.x * w)
        y_coords.append(landmark.y * h)

    if not x_coords or not y_coords:
        return 0, None

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    bbox_pixel_width = max_x - min_x
    bbox_for_drawing = (int(min_x), int(min_y), int(max_x), int(max_y))

    return bbox_pixel_width, bbox_for_drawing

def calibrate_camera() -> tuple:
    """
    Calibrates the camera to find the focal length based on a known hand width and distance.

    Returns:
        tuple: A tuple containing the calculated focal length and real hand width.
    """

    real_hand_width_cm = 0.0
    known_distance_cm = 0.0

    while True:
        try:
            real_hand_width_cm = float(input("Enter the REAL width of your hand (e.g., 8.5 cm): "))
            if real_hand_width_cm > 0:
                break
            else:
                print("Please enter a positive value for hand width.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    while True:
        try:
            known_distance_cm = float(input(f"Enter the distance you will hold your hand from the camera for calibration (e.g., 50 cm): "))
            if known_distance_cm > 0:
                break
            else:
                print("Please enter a positive value for distance.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    print("\n--- Camera Calibration Instructions ---")
    print(f"1. You've stated your hand width is: {real_hand_width_cm} cm.")
    print(f"2. You will place your hand: {known_distance_cm} cm away from the camera.")
    print(f"3. Keep your hand steady, palm facing the camera, so its width is clearly presented.")
    print(f"4. The current pixel width of the detected hand bbox will be shown.")
    print(f"5. When you are ready and your hand is positioned correctly, press the 'c' key to capture.")
    print(f"6. Press 'q' to quit calibration.")

    cap = cv2.VideoCapture(0) # 0 for default CSI camera or first USB camera
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return None

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=1,
                           min_detection_confidence=0.7,
                           min_tracking_confidence=0.7)
    mp_drawing = mp.solutions.drawing_utils

    focal_length_calculated = None
    captured_pixel_width = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        image = cv2.flip(image, 1)
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        results = hands.process(image_rgb)
        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        instruction_text = f"Hold hand at {known_distance_cm}cm. Press 'c' to capture."
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                current_pixel_width, bbox = get_hand_bbox_and_width(image_bgr, hand_landmarks)

                if bbox and current_pixel_width > 0:
                    captured_pixel_width = current_pixel_width
                    cv2.rectangle(image_bgr, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    cv2.putText(image_bgr, f"Current Px Width: {current_pixel_width:.0f}", 
                                (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    instruction_text = f"Px Width: {current_pixel_width:.0f}. Hold at {known_distance_cm}cm. Press 'c'."

        cv2.putText(image_bgr, instruction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(image_bgr, "Press 'q' to quit.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        cv2.imshow('Camera Calibration - Press "c" to Capture', image_bgr)

        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            if captured_pixel_width > 0 and real_hand_width_cm > 0 and known_distance_cm > 0:
                focal_length_calculated = (captured_pixel_width * known_distance_cm) / real_hand_width_cm
                
                print(f"\n--- Calibration Complete ---")
                print(f"Real Hand Width used: {real_hand_width_cm} cm")
                print(f"Known Distance used: {known_distance_cm} cm")
                print(f"Pixel Width measured at this distance: {captured_pixel_width:.2f} pixels")
                print(f"==> CALCULATED FOCAL LENGTH (F): {focal_length_calculated:.2f} <==")
                print("\nIMPORTANT: Save this 'Focal Length' value and the 'Real Hand Width' you used.")
                print("You will need them for the distance measurement program.")
                
                final_msg_img = image_bgr.copy()
                cv2.putText(final_msg_img, f"Focal Length (F): {focal_length_calculated:.2f}", 
                            (final_msg_img.shape[1]//2 - 200, final_msg_img.shape[0]//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
                cv2.putText(final_msg_img, "SAVED! Press any key to exit calibration.", 
                            (final_msg_img.shape[1]//2 - 250, final_msg_img.shape[0]//2 + 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)
                
                cv2.imshow('Camera Calibration - Press "c" to Capture', final_msg_img)
                cv2.waitKey(0) # Wait indefinitely until a key is pressed
                
                break 
            else:
                print("\nError during calibration capture attempt:")
                if not (captured_pixel_width > 0) : print("  - Hand pixel width is zero. Ensure hand is clearly detected.")
                if not (real_hand_width_cm > 0) : print("  - Real hand width is not a positive number.")
                if not (known_distance_cm > 0) : print("  - Known distance is not a positive number.")
                print("  Please ensure your hand is steadily in view and all input values are correct. Try again or press 'q' to quit.")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    return focal_length_calculated, real_hand_width_cm

def calculate_hand_distance() -> None:
    """
    Calculates and displays the distance of the hand from the camera using the calibrated focal length and known hand width.
    """
    
    focal_length_from_calibration = 0.0
    real_hand_width_cm_for_measurement = 0.0

    while True:
        try:
            focal_length_from_calibration = float(input("Enter the CALIBRATED Focal Length (F) value from Program 1: "))
            if focal_length_from_calibration > 0:
                break
            else:
                print("Focal length must be a positive value.")
        except ValueError:
            print("Invalid input. Please enter a number for focal length.")
    
    while True:
        try:
            real_hand_width_cm_for_measurement = float(input("Enter the REAL Hand Width (cm) used during calibration (e.g., 8.5): "))
            if real_hand_width_cm_for_measurement > 0:
                break
            else:
                print("Hand width must be a positive value.")
        except ValueError:
            print("Invalid input. Please enter a number for hand width.")


    print("\n--- Hand Distance Measurement Program ---")
    print(f"Using: Focal Length = {focal_length_from_calibration:.2f}")
    print(f"Using: Real Hand Width = {real_hand_width_cm_for_measurement} cm")
    print("Move your hand in front of the camera. The estimated distance will be displayed.")
    print("Press 'q' to quit.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=1,
                           min_detection_confidence=0.7,
                           min_tracking_confidence=0.7)
    mp_drawing = mp.solutions.drawing_utils

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.flip(image, 1)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        results = hands.process(image_rgb)
        
        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                current_pixel_width, bbox = get_hand_bbox_and_width(image_bgr, hand_landmarks)

                if bbox and current_pixel_width > 0:
                    cv2.rectangle(image_bgr, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    
                    estimated_distance_cm = (real_hand_width_cm_for_measurement * focal_length_from_calibration) / current_pixel_width
                    estimated_distance_m = estimated_distance_cm * (1/100)

                    if estimated_distance_m < 2:
                        cv2.putText(image_bgr, f"Move Back!! {estimated_distance_m:.2f} m", 
                                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    cv2.putText(image_bgr, f"Dist: {estimated_distance_cm:.1f} cm", 
                                (bbox[0], bbox[1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                    cv2.putText(image_bgr, f"Dist: {estimated_distance_m:.2f} m",
                                (bbox[0], bbox[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                    cv2.putText(image_bgr, f"Px Width: {current_pixel_width:.0f}", 
                                (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # cv2.putText(image_bgr, "Press 'q' to quit.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imshow('Hand Distance Measurement', image_bgr)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == '__main__':
    while True:
        print("\nHand Distance Measurement Utility")
        print("---------------------------------")
        choice = input("Select program to run:\n  1. Calibrate Camera (Program 1)\n  2. Measure Hand Distance (Program 2)\n  3. Exit\nEnter your choice (1, 2, or 3): ")
        
        if choice == '1':
            print("\nStarting Camera Calibration (Program 1)...")
            focal_length, hand_width = calibrate_camera()
            if focal_length is not None:
                print(f"\nCalibration finished.")
                print(f"  Calculated Focal Length (F): {focal_length:.2f}")
                print(f"  Real Hand Width used: {hand_width} cm")
                print("Make sure to use these exact values in Program 2.")
            else:
                print("\nCalibration was not completed or was exited by the user.")
            print("-" * 40)
        
        elif choice == '2':
            print("\nStarting Hand Distance Measurement (Program 2)...")
            calculate_hand_distance()
            print("\nDistance measurement program finished.")
            print("-" * 40)
        
        elif choice == '3':
            print("Exiting program.")
            break
        
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")