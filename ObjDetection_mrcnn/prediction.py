import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import os
from matplotlib import pyplot
from matplotlib.patches import Rectangle

# load the class label names from disk, one label per line
# CLASS_NAMES = open("coco_labels.txt").read().strip().split("\n")

CLASS_NAMES = ['BG','Budgie']

class SimpleConfig(mrcnn.config.Config):
    # Give the configuration a recognizable name
    NAME = "coco_inference"
    
    # set the number of GPUs to use along with the number of images per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

	# Number of classes = number of classes + 1 (+1 for the background). The background class is named BG
    NUM_CLASSES = len(CLASS_NAMES)

# draw an image with detected objects
def draw_image_with_boxes(filename, boxes_list):
     # plot the image
     pyplot.imshow(filename)
     # get the context for drawing boxes
     ax = pyplot.gca()
     
     box = boxes_list[0]
     # get coordinates
     y1, x1, y2, x2 = box
     # calculate width and height of the box
     width, height = x2 - x1, y2 - y1
     # create the shape
     rect = Rectangle((x1, y1), width, height, fill=False, color='red')
     # draw the box
     ax.add_patch(rect)

     # plot each box
     '''for box in boxes_list:
          # get coordinates
          y1, x1, y2, x2 = box
          # calculate width and height of the box
          width, height = x2 - x1, y2 - y1
          # create the shape
          rect = Rectangle((x1, y1), width, height, fill=False, color='red')
          # draw the box
          ax.add_patch(rect)'''
     # show the plot
     pyplot.show()


# Initialize the Mask R-CNN model for inference and then load the weights.
# This step builds the Keras model architecture.
model = mrcnn.model.MaskRCNN(mode="inference", 
                             config=SimpleConfig(),
                             model_dir=os.getcwd())

# Load the weights into the model.
model.load_weights(filepath="Three_mask_rcnn_trained-100.h5", 
                   by_name=True)

# load the input image, convert it from BGR to RGB channel
image = cv2.imread("images/validation4.jpg")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform a forward pass of the network to obtain the results
r = model.detect([image])

# Get the results for the first image.
r = r[0]

# Visualize the detected objects.
mrcnn.visualize.display_instances(image=image, 
                                  boxes=r['rois'], 
                                  masks=r['masks'], 
                                  class_ids=r['class_ids'], 
                                  class_names=CLASS_NAMES, 
                                  scores=r['scores'],
                                  show_mask=False, 
                                  show_mask_polygon=False, 
                                  min_score=0.98)
    
#draw_image_with_boxes(image , r['rois'])
