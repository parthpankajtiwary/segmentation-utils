# sourcery skip: avoid-builtin-shadow
from segmentation_utils import SegmentationMaskVisualization

height, width, channel = (224, 224)
n_classes = 3
n_samples = 8
labels = ['']

"""Supports standard COCO formats polygon and rle"""
format = "rle"

visualizer = SegmentationMaskVisualization(height=height,
                                           width=width,
                                           n_classes=n_classes,
                                           n_samples=n_samples, 
                                           labels=labels)

"""Visualize a batch (n, h, w, c) of image + mask in a grid (n samples)"""
image_batch, mask_batch = [], []
visualizer.visualize_samples(image_batch, 
                             mask_batch)

"""Visualize a batch of (n, h, w, c) of image + prediction + mask in grid (n samples)"""
image_batch, mask_batch = [], []
prediction_batch = []
visualizer.visualize_mask_prediction_samples(image_batch, 
                                             prediction_batch, 
                                             mask_batch)

"""Visualize agreement/disagreement b/w a batch of (n, h, w, c) of prediction + mask (n samples)"""
image_batch, mask_batch = [], []
prediction_batch = []
visualizer.visualize_agreement_disagreement(image_batch, 
                                            prediction_batch, 
                                            mask_batch)

