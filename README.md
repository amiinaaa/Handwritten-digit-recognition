#Handwritten Digit Recognition
Description
This project uses machine learning and deep learning techniques for recognizing handwritten digits. The studied algorithms include K-Nearest Neighbors (KNN) and Convolutional Neural Networks (CNN). The goal is to develop a precise and efficient system for handwritten digit recognition by comparing the performance of different models.

Technologies
Python 3.12.6
TensorFlow
NumPy
Matplotlib
OpenCV
Scikit-learn
Dataset
The project uses the MNIST (Modified National Institute of Standards and Technology) dataset, which contains 70,000 grayscale images of handwritten digits, with 60,000 images for training and 10,000 for testing. Each image is 28x28 pixels, and the labels range from 0 to 9.

Installation
Clone the project repository:
bash
Copier le code
git clone https://github.com/username/repository.git
Navigate to the project directory:
bash
Copier le code
cd repository
Install the required dependencies:
bash
Copier le code
pip install -r requirements.txt
Usage
To train the KNN model:

bash
Copier le code
python knn_model.py
This script uses the KNN algorithm to classify handwritten digits and evaluates the model's performance on the test set.

To train the CNN model:

bash
Copier le code
python cnn_model.py
This script uses a Convolutional Neural Network to extract features and perform digit classification.

Test the models on custom images:

bash
Copier le code
python test_custom_images.py --image path_to_image
This script allows users to test the trained model on custom handwritten digit images.

Results
KNN: The KNN model achieved an accuracy of 98.61% on the test set, demonstrating a strong ability to classify handwritten digits for moderately sized datasets.
CNN: The CNN model provided precise classification by leveraging its ability to extract complex features from images.
Contributions
Contributions are welcome! Feel free to open an issue to discuss improvements or submit a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.
