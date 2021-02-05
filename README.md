# autoscoring_centrioles

## Project Goal:
Machine learning based model that takes as input any 3D images from a microscope with DAPI channel for the nuclei and two others channel for the centrioles and gives as output the segmentation of nuclei and their corresponding centrioles.

## Global Strategy: 
1-	Nuclei segmentation
2-	Nuclei annotation 
3-	Centriole segmentation 
4-	Centriole annotation

1-	Nuclei segmentation 

Nuclei segmentation will be based on DAPI channel. We don’t need 3D images, but rather the focal image. There are here two possibilities, we could either used our datasets and create mask with a well optimize image processing algorithm such as Ilastik or Cellprofiler and train a neural network on them. Or we could use external images from Kaggle challenge to develop a more global algorithm: https://www.kaggle.com/c/data-science-bowl-2018.

2-	Nuclei annotation

Nucleus will then be separately detected and annotated based on their morphology in the hope of extracting their division stage. There are two possibilities: we can either used a machine learning based on morphology features such as intensity, size, shape and thanks to an unsupervised model find the correct features to classify stages. Or we could create a deep learning algorithm that would take every nucleus as input and different stages as output. However, a labeling for a training and testing set has to be done. Maybe take 3D images.

3-	Centriole segmentation

Centriole segmentation will be a difficult task for a deep learning model because of the high dimensionality: dataset*position*Z_stack*channel*image_x*image_y which is huge. We need to carefully choose the values to reduce at maximum. For instance, at the first look at the data Z_stack could be in my opinion be lower as for maybe the image size.
The first step will be to do a deconvolution to have better input images. Then create 3D mask from ilastik software. And finally train a deep learning algorithm. 

4-	Centriole annotation 

Based on centriole morphology, we could as for nuclei annotation, detect the division stage of the cell which can be put into perspective from the nuclei annotation.
