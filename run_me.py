# no she-bang in Windows

# ref https://www.youtube.com/watch?v=2yQqg_mXuPQ&t=654s
# initial setup 4/9/2023

from Detector import *

### this first model works fine so far on my machine with static images
modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz"

### this model is sketchy, crashes VS Code with OOM unless you bounce VS Code before running
# modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz"

### can properly load or run this model
# modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20210210/centernet_mobilenetv2fpn_512x512_coco17_od.tar.gz"

classFile = "coco.names"
# imagePath = "test/example-livingRoom.jpg"
imagePath = "test/example-bluePlayer.PNG"
threshold = 0.5

detector = Detector()
detector.readClasses(classFile)
detector.downloadModel(modelURL)
detector.loadModel()
detector.predictImage(imagePath, threshold)