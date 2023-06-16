# Brain Tumor Classifier

## Dataset

| Tumor typology  | Instances |
|-----------------|-----------|
| Glioma tumor    | 926       |
| Meningioma tumor| 937       |
| No tumor        | 500       |
| Pituitary Tumor | 901       |

## Methods and Experiments

### Data Preprocessing

The data preprocessing phase involved several key steps to prepare the dataset for training our model.

First, we conducted an analysis of the dataset to understand the class distribution and ensure class balance. This analysis allowed us to identify a class imbalance, found in the part of the dataset dedicated to negative MRIs.

Next, we performed the division of the dataset into training, validation, and test sets. The training set was used to train the model, the validation set to tune the hyperparameters and monitor performance, and the test set to evaluate the generalization ability of the final model.

To address class imbalances, we applied data augmentation techniques. Initially, we focused on augmenting the minority class to increase its representation in the dataset. This involved generating additional synthetic images using transformations such as rotate, scale, and flip.

Additionally, we applied data augmentation to the entire training dataset, including all classes. This larger dataset further enriched the training data, introducing variations that could improve the model's ability to generalize and improve its robustness.

After the data augmentation process, we performed a final analysis of the dataset to ensure that class equilibrium was indeed achieved. This analysis allowed us to verify that the augmentation techniques successfully balanced the classes, ensuring that each class had a comparable number of samples to train.

### CLAHE

To improve the quality and feature representation of images in our dataset, we incorporated the Contrast Limited Adaptive Histogram Equalization (CLAHE) method as a preprocessing technique. We created a parallel dataset using CLAHE for use with the various models tested.

CLAHE (Contrast Limited Adaptive Histogram Equalization) is an image enhancement technique that aims to improve contrast by redistributing the luminance values of the image. The CLAHE algorithm consists of three main parts: tile generation, histogram equalization, and bilinear interpolation.

In the CLAHE algorithm, the input image is divided into tiles, and histogram equalization is applied to each tile independently. The excess values are redistributed to prevent over-amplification of noise, and the resulting tiles are stitched together using bilinear interpolation. This process creates an output image with improved contrast and enhanced edge definitions.

### CNN Experiments

Based on the experiment results, the best model in terms of F1 score on the test set is the one from experiment 3, achieving an F1 score of 0.92. The model demonstrates good validation loss and generalization ability.

### Pretrained Models

We used three pretrained models in our experiments: VGG16, ResNet50, and InceptionV3.

VGG16 is a deep CNN architecture consisting of 16 layers, including convolutional and fully connected layers. It has proven to be a powerful architecture for image classification tasks.

ResNet50 is another popular CNN architecture that introduces residual connections, which help address the vanishing gradient problem during training. Unfortunatelly we didn't find the right configuration of the ResNet50 to get good results.

InceptionV3 is a deep CNN architecture that uses a combination of convolutional layers with different kernel sizes to capture features at various scales. It has been widely used in image classification tasks and has achieved good results.

We experimented fine-tuning and feature extraction with these models.

### Explainability

Explainability in CNNs refers to the ability to understand and interpret the inner workings of these complex models. In the context of medical imaging, explainability becomes crucial for ensuring trust and understanding in the predictions made by CNNs.

We used explainability techniques such as intermediate activations and heatmaps to gain insights into how the models make predictions, especially in the detection of brain tumors. By visualizing the activation patterns of different layers and generating heatmaps that highlight the regions contributing to the predictions, we aimed to improve the interpretability of the models.

## Conclusion

The best performance was achieved by the VGG16 trained on the CLAHE dataset gaing a 0.93 weighted F1 on the test set.
