# Heart Disease Risk Predictor

This project implements a neural network model to predict the risk of heart disease based on clinical data. It includes model training with PyTorch and an interactive web interface created with Streamlit to facilitate prediction and interpretation using SHAP explanations.

---

## 📋 Contents

- **train.py**: Code to load data, train the model, and save the trained model and scaler.  
- **model.py**: Definition of the neural network architecture `HeartDiseaseNN`.  
- **interface.py**: Streamlit app to input patient data and get predictions with explanations.  
- **heart.csv**: Original dataset with clinical data.  
- **output/**: Folder with subfolders for trained models, scalers, and plots.  

---

## 🚀 Execution

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your_username/heart-disease-predictor.git
   cd heart-disease-predictor
