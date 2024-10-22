import cv2
import base64
import time
import requests
import json  # Import the correct JSON module
import os
from datetime import datetime

# Function to capture image from webcam
def capture_image():
    # Open the webcam with DSHOW backend (use MSMF if DSHOW doesn't work)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Set camera resolution to Full HD (1920x1080)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Allow the camera time to initialize (2 seconds)
    time.sleep(2)

    # Discard the first few frames to let the camera stabilize
    for i in range(10):
        ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        cap.release()
        return None

    # Display the captured frame to check if it's working
    cv2.imshow("Captured Frame", cv2.resize(frame, (960, 540)))  # Resize for display
    cv2.waitKey(500)  # Display for 0.5 seconds
    cv2.destroyAllWindows()

    # Generate a timestamp-based filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = f"webcam_image_{timestamp}.png"
    cv2.imwrite(image_path, frame)  # Save the captured frame
    
    # Release the camera after capture
    cap.release()
    return image_path

# Function to convert image to base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to send image to LLaVA and get description
def send_to_llava(image_base64):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llava",
        "prompt": "What's in this image?",
        "images": [image_base64]
    }
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=payload, headers=headers, stream=True)
        
        full_response = ""
        
        # Read the response stream line by line
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                print(f"Raw line: {decoded_line}")  # Debugging output
                try:
                    # Parse each line as JSON using the correct json module
                    json_line = json.loads(decoded_line)
                    full_response += json_line.get("response", "")
                    # Check if the response is completed
                    if json_line.get("done", False):
                        break
                except Exception as e:
                    print(f"Error processing line: {e}")
                    print(f"Full response so far: {full_response}")  # Print the response so far
                    return "Invalid JSON response"
        
        return full_response
    
    except Exception as e:
        print(f"Error: {e}")
        return "Error in sending request"

# Main function to capture and describe images every 20 seconds
def main():
    while True:
        # Capture an image from the webcam
        image_path = capture_image()
        
        if image_path:
            # Convert the captured image to base64
            image_base64 = image_to_base64(image_path)
            
            # Send the base64 image to LLaVA and get the description
            description = send_to_llava(image_base64)
            
            # Write the description to a text file
            with open("llava_output.txt", "a") as file:
                file.write(f"{time.ctime()}: {description}\n")
            
            # Print the description for debugging
            print(f"{time.ctime()}: {description}")
        
        # Wait for 20 seconds before capturing the next image
        time.sleep(20)

if __name__ == "__main__":
    main()
