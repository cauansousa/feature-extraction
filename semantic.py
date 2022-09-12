import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import numpy as np

filename = '/content/img/img.jpg'
model="yolov3"
confidence=0.2
# Read the image into a numpy array
#img = cv2.imread(filename)

frame = cv2.VideoCapture(0)    

while frame.isOpened():
    ret, img = frame.read()
        # Perform the object detection
    bbox, label, conf = cv.detect_common_objects(img, confidence=confidence, model=model)
        
        # Print current image's filename
    #print(f"========================\nImage processed: {filename}\n")
        
        # Print detected objects with confidence level
    #for l, c in zip(label, conf):
    #    print(f"Detected object: {l} with confidence level of {c}\n")
        
        # Create a new image that includes the bounding boxes
    output_image = draw_bbox(img, bbox, label, conf)
        
        # Save the image in the directory images_with_boxes
    #cv2.imwrite(f'images_with_boxes/{filename}', output_image)
        
        # Display the image with bounding boxes
    cv2.imshow('frame', output_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
      
img.release()
cv2.destroyAllWindows()