🌱 Plant Disease Detection using Deep Learning
📌 Problem Statement

Agriculture plays a crucial role in global food security, but plant diseases can severely affect crop yield and quality. Manual detection of plant diseases is time-consuming, requires expert knowledge, and is prone to errors.
There is a need for an automated, accurate, and scalable solution to detect plant diseases early and help farmers take timely action.

💡 Solution

This project presents a Deep Learning–based Plant Disease Detection System that:

Uses Convolutional Neural Networks (CNNs) to classify plant images into healthy or diseased categories.

Provides real-time predictions through an easy-to-use web application.

Visualizes training performance (accuracy and loss graphs).

Can be deployed for farmers, researchers, or agri-tech companies to ensure sustainable crop production.

📂 Dataset

Source: PlantVillage dataset (publicly available).

Classes: Multiple crops and their respective healthy/diseased leaf images.

Size: 50,000+ images (after preprocessing & augmentation).

Images are resized and normalized for CNN input.

🛠️ Tech Stack

Language: Python

Libraries: TensorFlow / Keras, OpenCV, NumPy, Matplotlib

Framework: Streamlit / Flask (for the app)

Environment: Jupyter Notebook for training, VS Code for deployment

🔬 Model Architecture

Input Layer → Resized plant leaf image

Convolutional Layers → Feature extraction

Pooling Layers → Downsampling

Fully Connected Layers → Classification

Output Layer → Softmax activation for disease category prediction

📊 Results

Training Accuracy: ~99%

Validation Accuracy: ~97%

Loss graphs: Used to verify model convergence and avoid overfitting.

The model shows high accuracy and generalization on unseen data, making it reliable for real-world applications.

🚀 Deployment

Built an interactive web app (app.py)

Users can:

Upload a plant leaf image

Get instant prediction of the disease type

View accuracy/loss visualization of model training

Can be deployed on Streamlit Cloud / Flask server / Docker / Heroku

📈 Impact

✅ Helps farmers detect diseases early
✅ Reduces dependency on experts
✅ Supports smart farming & food security
✅ Scalable solution for AgriTech startups

👨‍💻 How to Run
# Clone repo
git clone https://github.com/yourusername/plant-disease-detection.git
cd plant-disease-detection

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

📸 Sample Output

Uploaded Image → Predicted Disease → Accuracy Graph

✨ This project demonstrates the application of AI in agriculture, combining computer vision and deep learning to solve real-world problems.
