import cv2 
import os

output_path= "dataset_2"
cap = cv2.VideoCapture("input_2.mp4")

def enroll_face(output:str,cap):

    framecount=0
    capturecount=0
    os.makedirs(output, exist_ok=True)
    
    while True:

        ret,frame=cap.read()
        if not ret:
            print("error: unable to capture frame")
            break
        framecount +=1   
        # cv2.imshow("video feed",frame[0:250,150:400]) # to visualize video 
        # print(framecount)

        key = cv2.waitKey(1) & 0xFF 

        if framecount % 10 == 0 :
            capturecount +=1
            image_name = os.path.join(output, f"image{capturecount}.jpg")
            cv2.imwrite(image_name, frame)
            print(capturecount)
            # print(f"Saved {image_name}")

        if key == ord('c'):            
            cap.release()
            cv2.destroyAllWindows()

enroll_face(output_path,cap=cap)