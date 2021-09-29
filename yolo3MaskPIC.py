import numpy as np  
import cv2
from matplotlib import pyplot as plt
import os

objectnessThreshold = 0.3 
confThreshold = 0.3       
nmsThreshold = 0.3        
inpWidth = 320            
inpHeight = 320           

classesFile = r'D:\proG\computerVision\cVcourse\projects\trainingCustomFaceMaskYolo\yoloMask\class.names'
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')


#modelConfiguration = r'D:\proG\computerVision\cVcourse\projects\trainingCustomFaceMaskYolo\yoloMask\yolov4-tiny608.cfg.txt'
#modelWeights = r'D:\proG\computerVision\cVcourse\projects\trainingCustomFaceMaskYolo\yoloMask\yolov4-tiny608_best.weights'

#modelConfiguration = r'D:\proG\computerVision\cVcourse\projects\trainingCustomFaceMaskYolo\yoloMask\yolov3-mask.cfg'
#modelWeights = r'D:\proG\computerVision\cVcourse\projects\trainingCustomFaceMaskYolo\yoloMask\yolov3-mask_2000.weights'
#
#modelConfiguration = r'D:\proG\computerVision\cVcourse\projects\trainingCustomFaceMaskYolo\yolov4\version1\yolov4-mask.cfg'
#modelWeights = r'D:\proG\computerVision\cVcourse\projects\trainingCustomFaceMaskYolo\yolov4\version1\yolov4-mask_final.weights'


net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)


def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] -1] for i in net.getUnconnectedOutLayers()]


def drawPred(classId, conf, left, top, right, bottom):
    
    if classId == 1:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)
    elif classId == 0:
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 3)
       
    if classes:
        assert(classId < len(classes))
        label = '%s' % (classes[classId])

    
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
    

def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            if detection[4] > objectnessThreshold :
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)

def detect(frame):
    blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
    net.setInput(blob)
    outs = net.forward(getOutputsNames(net))
    postprocess(frame, outs)




imagesP = r'D:\proG\computerVision\cVcourse\projects\trainingCustomFaceMaskYolo\testData'

lab = os.listdir(imagesP)
li = []
for i in lab:
    if i.endswith('.jpg'):
        fullP = os.path.join(imagesP,i)
        li.append(fullP)

imgS = []
for i in li:
    frame = cv2.imread(i)
    detect(frame)
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    print(label)
    imgS.append(frame)


plt.figure(figsize=(12,10))
plt.subplot(221)
plt.imshow(imgS[0][:,:,::-1])
plt.subplot(222)
plt.imshow(imgS[1][:,:,::-1])
plt.subplot(223)
plt.imshow(imgS[2][:,:,::-1])
plt.subplot(224)
plt.imshow(imgS[3][:,:,::-1])


plt.show()

































