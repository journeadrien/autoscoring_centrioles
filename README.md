# autoscoring_centrioles

## Project Goal:
Machine learning based model that takes as input any 3D images from a microscope with DAPI channel for the nuclei and two others channel for the centrioles and gives as output the segmentation of nuclei and their corresponding centrioles.

## Global Strategy: 

1-	Nuclei segmentation 

Nuclei segmentation will be based on DAPI channel. We don’t need 3D images, but rather the focal image. We used an external images from Kaggle challenge to develop a more Mask-RCNN (instacne segmentation) algorithm: https://www.kaggle.com/c/data-science-bowl-2018.

2-	Nuclei annotation

Nucleus will then be separately detected and annotated based on their morphology in the hope of extracting their division stage. We used  deep learning algorithms (Resnet, efficientNet) that take every nucleus as input and different stages as output.

3-	Centriole segmentation

Centriole segmentation is a difficult task for a deep learning model because of the high dimensionality: dataset*position*Z_stack*channel*image_x*image_y which is huge (it is the same problem as detect small stars in a image of space with a lot of noises). We need to carefully choose the values to reduce at maximum. For instance, we reduced the z_stack with normal RGB input by putting: (R: mean(z), G: std(z), B: max_projection(z)).

4-	Centriole annotation 

Based on centriole morphology, we could as for nuclei annotation, detect the division stage of the cell which can be put into perspective from the nuclei annotation.
