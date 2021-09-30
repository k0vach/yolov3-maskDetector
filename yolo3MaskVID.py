import numpy as np
import cv2


objectnessThreshold = 0.4 
confThreshold = 0.4       
nmsThreshold = 0.4        
inpWidth = 608            
inpHeight = 608          

classesFile = r'D:\proG\computerVision\cVcourse\projects\trainingCustomFaceMaskYolo\class.names'
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')


#modelConfiguration = r'D:\proG\computerVision\cVcourse\projects\trainingCustomFaceMaskYolo\yoloMask\yolov4-tiny608.cfg.txt'
#modelWeights = r'D:\proG\computerVision\cVcourse\projects\trainingCustomFaceMaskYolo\yoloMask\yolov4-tiny608_best.weights'

#modelConfiguration = r'D:\proG\computerVision\cVcourse\projects\trainingCustomFaceMaskYolo\yoloMask\yolov3-mask.cfg'
#modelWeights = r'D:\proG\computerVision\cVcourse\projects\trainingCustomFaceMaskYolo\yoloMask\yolov3-mask_2000.weights'
#
modelConfiguration = r'D:\proG\computerVision\cVcourse\projects\trainingCustomFaceMaskYolo\yolov4\version1\yolov4-mask.cfg'
modelWeights = r'D:\proG\computerVision\cVcourse\projects\trainingCustomFaceMaskYolo\yolov4\version1\yolov4-mask_final.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    if classId == 1:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)
    elif classId == 0:
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 3)
    label = '%.2f' % conf
        
    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

# Remove the bounding boxes with low confidence using non-maxima suppression
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
    #outs = net.forward()
    postprocess(frame, outs)
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(frame, label, (0, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255))


video = cv2.VideoCapture(r'D:\proG\computerVision\cVcourse\projects\trainingCustomFaceMaskYolo\testData\test-video2.mp4')
width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
#print(width,height)
out = cv2.VideoWriter('yolov4608-test.mp4',cv2.VideoWriter_fourcc(*'mp4v'),30,(int(width),int(height)))
k = 0
scale = 0.7
count = 0
skipFrame = 2

while k != 27:
    count += 1
    status,frame = video.read()
    
    if status:        
        detect(frame)
        #cv2.imshow('Video', frame)
        out.write(frame)
        cv2.VideoWriter    
        k = cv2.waitKey(1)       
        
    else:
        break

cv2.destroyAllWindows()
video.release()
out.release()