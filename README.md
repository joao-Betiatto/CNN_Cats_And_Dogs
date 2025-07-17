# Dog vs. Cat Image Classifier

This project is a simple deep learning model built in Google Colab that classifies images as either cats or dogs. It uses the MobileNetV2 architecture with transfer learning to achieve accurate predictions while keeping training time low.

🔧 Tools & Libraries

  • Python
  
  • TensorFlow / Keras
  
  • MobileNetV2 (pre-trained on ImageNet)

📂 Dataset
  
  • The model was trained using an dataset with cats and dogs images, icluding 2000 images of cats and 2000 images of dogs for training, and 500 images of cats and 500 images of dogs for tests.

🧠 Model

  • Uses MobileNetV2 as the base (with frozen layers initially)
  
  • Adds a custom classification head for binary classification
  
  • Applies data augmentation to improve generalization

📊 Results

  • The model reaches high accuracy (typically over 90%) after just a few epochs thanks to transfer learning.

🚀 How to Run
  
  • Open the notebook in Google Colab
  
  • load the dataset.
  
  • Run the training cells.
  
  • Upload any image to test the model's prediction.
