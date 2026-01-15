#TensorFlow Functions Used in Object Detection
tf.image.resize()
Resizes images to a fixed shape so that they can be uniformly processed by the neural network.
tf.image.decode_jpeg()
Decodes JPEG image files into tensor format for further processing by the model.
tf.image.convert_image_dtype()
Converts image data types and normalizes pixel values for better model performance.
tf.keras.layers.Conv2D()
Applies convolution operations to extract spatial and visual features from images.
tf.keras.layers.MaxPooling2D()
Reduces the size of feature maps while retaining important information.
tf.keras.applications.MobileNetV2()
A lightweight, pre-trained convolutional neural network commonly used as a backbone in object detection models.
tf.keras.Model()
Used to define, build, and customize deep learning models in TensorFlow.
tf.keras.losses.SparseCategoricalCrossentropy()
Calculates classification loss during training for multi-class problems.
tf.data.Dataset.from_tensor_slices()
Creates efficient and optimized data pipelines for loading images and labels.
tf.image.non_max_suppression()
Removes overlapping bounding boxes and keeps only the most accurate detections.
