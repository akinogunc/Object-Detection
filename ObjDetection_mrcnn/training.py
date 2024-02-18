import os
import json
from numpy import zeros, asarray

import mrcnn.utils
import mrcnn.config
import mrcnn.model

class BudgieDataset(mrcnn.utils.Dataset):

    def load_dataset(self, dataset_dir, is_train=True):
        self.add_class("dataset", 1, "budgie")

        images_dir = dataset_dir + '/train/images/'
        annotations_dir = dataset_dir + '/train/annots/'

        for filename in os.listdir(images_dir):
            image_id = filename[:-4]
            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + 'json'

            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        path = info['annotation']
        boxes, w, h = self.extract_boxes(path)
        masks = zeros([h, w, len(boxes)], dtype='uint8')

        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('budgie'))
        return masks, asarray(class_ids, dtype='int32')

    def extract_boxes(self, filename):
        with open(filename) as f:
            data = json.load(f)

        boxes = list()
        for key in data.keys():
            regions = data[key]['regions']
            for _, region in regions.items():
                shape_attributes = region['shape_attributes']
                all_points_x = shape_attributes['all_points_x']
                all_points_y = shape_attributes['all_points_y']
                xmin = min(all_points_x)
                ymin = min(all_points_y)
                xmax = max(all_points_x)
                ymax = max(all_points_y)
                coors = [xmin, ymin, xmax, ymax]
                boxes.append(coors)

        width = 1024  # replace with your image width
        height = 1024  # replace with your image height
        return boxes, width, height


class BudgieConfig(mrcnn.config.Config):
    NAME = "three_cfg"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    NUM_CLASSES = 2

    STEPS_PER_EPOCH = 131

# Train
train_dataset = BudgieDataset()
train_dataset.load_dataset(dataset_dir='data', is_train=True)
train_dataset.prepare()

# Validation
validation_dataset = BudgieDataset()
validation_dataset.load_dataset(dataset_dir='data', is_train=False)
validation_dataset.prepare()

# Model Configuration
budgie_config = BudgieConfig()

# Build the Mask R-CNN Model Architecture
model = mrcnn.model.MaskRCNN(mode='training', 
                             model_dir='./', 
                             config=budgie_config)

model.load_weights(filepath='mrcnn/mask_rcnn_coco.h5', 
                   by_name=True, 
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

model.train(train_dataset=train_dataset, 
            val_dataset=validation_dataset, 
            learning_rate=budgie_config.LEARNING_RATE, 
            epochs=1, 
            layers='heads')

model_path = 'Three_mask_rcnn_trained-100.h5'
model.keras_model.save_weights(model_path)