import os
import cv2
import time
import uuid

IMAGE_PATH = "CollectedImages"

labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis','snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
           'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
           'scissors', 'teddy bear', 'hair drier', 'toothbrush']

number_of_images= 5

for label in labels:
    img_path = os.path.join(IMAGE_PATH, label)
    os.makedirs(img_path)

    #open camera
    cap = cv2.VideoCapture(1)
    print(f"collecting images for{label}")
    time.sleep(10)

    for imgnum in range(number_of_images):
        ret, frame = cap.read()
        imagename = os.path.join(IMAGE_PATH,label,label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
        cv2.imwrite(imagename,frame)
        time.sleep(5)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()