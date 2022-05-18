## Visualization framework for your deep learning segmentation tasks.
For a batch of numpy/tensor image, mask and prediction array in (n, h, w, c) format, this framework supports following visualizations:
* **Plot segmentation data:** generates a grid of 2 x n (n_samples) with images in the first column and ground truth mask in the second column.
* **Plot predicted segmentation samples:** generates a grid of 3 x n (n_samples) with images in first, ground truth mask in second and prediction mask in the third column.
* **Plot agreement-disagreement b/w mask and prediction:** generates a grid of 3 x n (n_samples) with images in first, ground truth mask in second, prediction mask in the third and agreement-agreement b/w mask and prediction in the fourth column.
&nbsp;


### Getting started

**Step 1:** Install Segmentation Mask Visualization library
```bash
pip install segmaskviz
```
**Step 2:** Import the library
```python
from segmaskviz import visualize

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
prediction_batch = []
visualizer.visualize_mask_prediction_samples(image_batch, 
                                             prediction_batch, 
                                             mask_batch)

"""Visualize agreement/disagreement b/w a batch of (n, h, w, c) of prediction + mask (n samples)"""
visualizer.visualize_agreement_disagreement(image_batch, 
                                            prediction_batch, 
                                            mask_batch)
```

### People behind Segmentation Mask Visualization framework
Created with :heart: by the [Parth](https://twitter.com/datawithparth)