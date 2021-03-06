# autoscoring_centrioles

## Project Goal:
Centrioles are barrel-shaped structures that are essential for cellular division. Deregulation of centrosome numbers may contribute to genome instability and tumor formation, whereas mutations in centrosomal proteins have recently been genetically linked to microcephaly and dwarfism. In this project, we proposed an automated counting of the centrosome (currently done manually). It is a machine learning-based model that takes as input any 3D images from a microscope with DAPI channel for the nuclei and two other channels for the centrioles and gives as output the segmentation of nuclei and their corresponding centrioles.

## Global Strategy: 

1-	Nuclei segmentation 

<p float="left">
  <img src=https://github.com/journeadrien/autoscoring_centrioles/blob/master/images/segmentation.png width="48%" />
  <img src=https://github.com/journeadrien/autoscoring_centrioles/blob/master/images/segmentation2.png width="48%" /> 
</p>

Nuclei segmentation will be based on DAPI channel. We don’t need 3D images, but rather the focal image. We used an external images from Kaggle challenge to develop a more Mask-RCNN (instacne segmentation) algorithm: https://www.kaggle.com/c/data-science-bowl-2018.


2-	Nuclei annotation

Nucleus will then be separately detected and annotated based on their morphology in the hope of extracting their division stage. We used  deep learning algorithms (Resnet, efficientNet) that take every nucleus as input and different stages as output.




3-	Centriole segmentation

<p float="left">
  <img src=https://github.com/journeadrien/autoscoring_centrioles/blob/master/images/centriole2.png width="32%" />
  <img src=https://github.com/journeadrien/autoscoring_centrioles/blob/master/images/centriole4.png width="32%" /> 
  <img src=https://github.com/journeadrien/autoscoring_centrioles/blob/master/images/centriole3.png width="32%" /> 
</p>

Centriole segmentation is a difficult task for a deep learning model because of the high dimensionality: dataset * position * Z_stack * channel * image_x * image_y which is huge (it is the same problem as detect small stars in a image of space with a lot of noises). We need to carefully choose the values to reduce at maximum. For instance, we reduced the z_stack with normal RGB input by putting: (R: mean(z), G: std(z), B: max_projection(z)).

Multiple complex models were trained on labeled images. However we could only get to 65% detection rate with a tendency to no dissociate two foci. In order to resolve it, I build a homemade neural network of FasterRCNN architecture with as backbone a the only five layers of a ResNET.

4-	Centriole annotation 

Based on centriole morphology, we could as for nuclei annotation, detect the division stage of the cell which can be put into perspective from the nuclei annotation.
