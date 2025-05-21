# Heart Disease Risk Predictor

This project implements a neural network model to predict the risk of heart disease based on clinical data. It includes model training with PyTorch and an interactive web interface created with Streamlit to facilitate prediction and interpretation using SHAP explanations.

---

## 📋 Contents

- **train.py**: Code to load data, train the model, and save the trained model and scaler.  
- **model.py**: Definition of the neural network architecture `HeartDiseaseNN`.  
- **interface.py**: Streamlit app to input patient data and get predictions with explanations.  
- **heart.csv**: .  
- **output/**: Folder with subfolders for trained models, scalers, and plots.  
- **data/**: Folder with original dataset with clinical data **heart.csv** 
---

## 🚀 Execution

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your_username/heart-disease-predictor.git
   cd heart-disease-predictor
   ```

2. **Run interface.py**
   Executing ```streamlit run interface.py --server.runOnSave=false```

3. **(Optional) Run train.py**
   If you want to generate a new trained model
