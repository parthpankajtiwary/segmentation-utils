import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from color_utils import ColorUtils

class SegmentationMaskVisualizer:

    def __init__(self, height, width, n_classes, n_samples, labels):
        self.height = height
        self.width = width
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.labels = labels
        self.color_utils = ColorUtils()


    def visualize_samples(self, images, masks):
        """Logic for visualising images + mask"""
        fig = plt.figure(figsize=(10, 25))
        gs = gridspec.GridSpec(nrows=self.n_samples, ncols=2)
        colors = self.color_utils.create_n_hex_colors(self.n_classes)
        patches = [mpatches.Patch(color=colors[i], label=f"{self.labels[i]}") for i in range(len(self.labels))]
        cmap = [mpl.colors.ListedColormap(color) for color in colors]

        for index, (image, mask) in enumerate(zip(images, masks)):
            if index >= self.n_samples:
                break
            image = image[:,:,0]
            ax0 = fig.add_subplot(gs[index, 0])
            im = ax0.imshow(image, cmap='bone')

            ax1 = fig.add_subplot(gs[index, 1])
            if index == 0:
                ax0.set_title("Image", fontsize=12, weight='bold', y=1.02)
                ax1.set_title("Mask", fontsize=12, weight='bold', y=1.02)
           
            plt.legend(handles=patches, 
                    bbox_to_anchor=(1.1, 0.65), 
                    loc=2, 
                    borderaxespad=0.4,
                    fontsize=14,
                    title='Mask Labels', 
                    title_fontsize=14, 
                    edgecolor="black",  
                    facecolor='#c5c6c7')
            
            l0 = ax1.imshow(image, cmap='bone')
            for index, value in enumerate(range(self.n_classes)):
                class_mask = mask[:,:,index]
                l1 = ax1.imshow(np.ma.masked_where(class_mask == False, class_mask), 
                                                   cmap=cmap[index], 
                                                   alpha=1)
                
            _ = [ax.set_axis_off() for ax in [ax0, ax1]]
            

    def visualize_mask_prediction_samples(self, images, predictions, masks):
        """Logic for visualising images + predictions + groundtruth"""
        fig = plt.figure(figsize=(15, 30))
        gs = gridspec.GridSpec(nrows=self.n_samples, ncols=3)
        colors = self.color_utils.create_n_hex_colors(self.n_classes)
        patches = [mpatches.Patch(color=colors[i], label=f"{self.labels[i]}") for i in range(len(self.labels))]

        cmap = [mpl.colors.ListedColormap(color) for color in colors]
        for index, (image, prediction, mask) in enumerate(zip(images, predictions, masks)):
            if index >= self.n_samples:
                break
            image = image[:,:,0]
            ax0 = fig.add_subplot(gs[index, 0])
            ax1 = fig.add_subplot(gs[index, 1])
            ax2 = fig.add_subplot(gs[index, 2])
                
            if index == 0:
                ax0.set_title("Image", fontsize=12, weight='bold', y=1.01)
                ax1.set_title("Mask", fontsize=12, weight='bold', y=1.01)
                ax2.set_title("Prediction", fontsize=12, weight='bold', y=1.01)
            
            plt.legend(handles=patches, 
                    bbox_to_anchor=(1.1, 0.65), 
                    loc=2, 
                    borderaxespad=0.4,
                    fontsize=14,
                    title='Mask Labels', 
                    title_fontsize=14, 
                    edgecolor="black",  
                    facecolor='#c5c6c7')
            
            im = ax0.imshow(image, cmap='bone')
            
            l0 = ax1.imshow(image, cmap='bone')
            for index, _ in enumerate(range(self.n_classes)):
                class_mask = mask[:,:,index]
                l1 = ax1.imshow(np.ma.masked_where(class_mask == False, class_mask), 
                                                   cmap=cmap[index], 
                                                   alpha=1)
            l2 = ax2.imshow(image, cmap='bone')
            threshold = 0.01
            for index, _ in enumerate(range(self.n_classes)):
                pred_class_mask = (prediction[:,:,index] > threshold).astype(np.float32)
                l3 = ax2.imshow(np.ma.masked_where(pred_class_mask == False, pred_class_mask), 
                                                   cmap=cmap[index], 
                                                   alpha=1)
                
            _ = [ax.set_axis_off() for ax in [ax0, ax1, ax2]]


    def visualize_agreement_disagreement(self, images, predictions, masks):
        """Logic for visualising agreement b/w ground truth and predictions"""



