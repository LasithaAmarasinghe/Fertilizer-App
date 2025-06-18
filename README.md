# Fertilizer App: Potato Disease Identification 🌱🦠

The **Fertilizer App** is designed to help identify common potato leaf diseases using deep learning models. Currently, the model is trained to classify potato leaves into three categories:

- **Early Blight** 🌿
- **Late Blight** 🍂
- **Healthy** ✅

In the future, this app will be generalized to support more crop diseases and broader agricultural applications.

## Features 🎉

- **Disease Detection**: Uses a TensorFlow model to predict diseases in potato leaves from uploaded images.
- **Model Inference**: Upload an image of a potato leaf, and the model will predict if the leaf is healthy or affected by diseases like Early Blight or Late Blight.
- **Future Expansion**: The model is being generalized to support other crops and their respective diseases.

## CNN Architecture🍃

The deep learning model used is based on a **Convolutional Neural Network (CNN)**, which is highly effective for image classification tasks.

### Key Components

1. **Preprocessing Layers:**
   - **Resizing**: The images are resized to 256x256 pixels to standardize the input size for the network.
   - **Rescaling**: Image pixel values are rescaled by dividing by 255, converting the pixel values into the range [0, 1].

2. **Data Augmentation:**
   - **Random Flip**: Images are randomly flipped both horizontally and vertically to simulate various real-world conditions and improve model generalization.
   - **Random Rotation**: The images are randomly rotated by up to 20% to enhance the model’s ability to recognize potato diseases under different orientations.

3. **Convolutional Layers:**
   - **Conv2D**: Several convolutional layers with 32 and 64 filters are used to extract important features from the potato leaf images.
   - **ReLU Activation**: The ReLU activation function is applied after each convolutional layer to introduce non-linearity and help the model learn complex patterns in the images.

4. **Max-Pooling Layers:**
   - **MaxPooling2D**: Max-pooling layers are used after each convolutional layer with a pool size of 2x2 to reduce the spatial dimensions of the feature maps, retaining only the most relevant information.

5. **Fully Connected Layers:**
   - **Flatten**: After the convolutional and pooling layers, the feature maps are flattened into a 1D vector to be fed into fully connected layers.
   - **Dense Layer (64 units)**: A fully connected layer with 64 units is used to connect the features extracted by the convolutional layers to the final output.
   - **Output Layer**: The output layer has 3 units, one for each class (**Early Blight**, **Late Blight**, **Healthy**), with a **softmax activation** to output the class probabilities.

### Summary of Layers:
- **Input**: Resized 256x256x3 (RGB) image
- **Conv2D Layers**: 32 and 64 filters with ReLU activation
- **MaxPooling2D Layers**: Pooling with a 2x2 filter size
- **Dense Layers**: 64 units followed by the final softmax output layer

This architecture was implemented using **TensorFlow** and **Keras** to ensure efficient model training and inference. The model was trained on potato leaf images and uses data augmentation techniques to improve generalization and robustness.
