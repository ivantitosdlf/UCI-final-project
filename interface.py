import streamlit as st
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import shap
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
from streamlit_shap import st_shap
shap.initjs()


# ──────────
# NOTE: Para evitar el error de torch.classes en desarrollo de Streamlit,
#        ejecuta:
#        streamlit run interface.py --server.runOnSave=false
# o bien configura la variable de entorno:
#        STREAMLIT_WATCHED_MODULES=""
# ──────────
import os
import streamlit as st

def display_saved_plots():
    st.header("📊 Heart Disease Dataset Statistics (Saved)")

    # Mostrar heatmap de correlación
    st.subheader("Feature Correlation Heatmap")
    st.image("plots/correlation_heatmap.png")

    # Mostrar distribuciones individuales
    st.header("📂 Individual Feature Distributions")

    for file in sorted(os.listdir("plots")):
        if  file.endswith("distribution.png"):
            variable_name = file.replace("dist_", "").replace(".png", "").replace("_", " ").title()
            st.subheader(variable_name)
            st.image(f"plots/{file}")

class HeartDiseaseNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return torch.softmax(x, dim=1)


# Cargar y preparar datos
def load_data(scaler,path='data/heart.csv'):
    data = pd.read_csv(path)
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    X = data.drop('HeartDisease', axis=1).values
    y = data['HeartDisease'].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.LongTensor(y_val)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)
    return data, label_encoders, X, y, X_train, X_val, y_train, y_val, train_loader, val_loader

# Cargar scaler
scaler = joblib.load('output/scalers/scaler.pkl')

data, label_encoders, X, y, X_train, *_ = load_data(scaler = scaler)
feature_names = data.columns[:-1].tolist()

# Cargar modelo
model = HeartDiseaseNN(X_train.shape[1])
model.load_state_dict(torch.load('output/trained_models/trained_HeartDiseaseNN.pth'))


feature_names = data.columns[:-1].tolist()
if 'page' not in st.session_state:
    st.session_state.page = 'predictor'

# Sidebar o botones para cambiar de pantalla
st.sidebar.title("Navigation")
if st.sidebar.button("🩺 Predictor"):
    st.session_state.page = 'predictor'
if st.sidebar.button("📈 Statistics"):
    st.session_state.page = 'statistics'


if st.session_state.page == 'predictor':
# Streamlit UI
    st.title("❤️ Heart Disease Risk Predictor")
    st.markdown("Please enter the following health information to estimate your risk of heart disease.")

    with st.form("heart_form"):
        
    
        age = st.number_input("Age", min_value=0, max_value=120, value=45)

        sex = st.selectbox("Sex", ["Male", "Female"])


        

        chest_pain = st.selectbox(
            "Type of chest pain",
            ["ATA", "NAP", "ASY", "TA"],
            help="""
            Describes the type of chest pain you may feel:

            - **ATA (Atypical Angina)**: Unusual chest discomfort not clearly related to physical activity.
            - **NAP (Non-Anginal Pain)**: Chest pain not related to the heart (e.g., from muscles or digestion).
            - **ASY (Asymptomatic)**: No chest pain at all — you might not feel any symptoms.
            - **TA (Typical Angina)**: Classic chest pain triggered by exercise or stress, likely linked to heart issues.
            """
                )


        exercise_angina = st.radio(
                    "Did you experience chest pain during exercise?",
                    ["Yes", "No"],
                    help="""
            Chest pain while exercising is called angina and may indicate that your heart is not getting enough oxygen during stress.
            """
                )
        


        resting_bp = st.number_input(
            "Resting systolic blood pressure (mm Hg)",
            min_value=50, max_value=250, value=120,
            help="""
    Your blood pressure when you're resting. 
    A normal value is around 120 mm Hg. High resting blood pressure can increase your risk of heart disease.
    """
        )

        cholesterol = st.number_input(
            "Cholesterol level (mg/dL)",
            min_value=100, max_value=600, value=200,
            help="""
    The amount of cholesterol in your blood. 
    Too much cholesterol can clog arteries and increase your risk of heart disease.
    """
        )

        fasting_bs = st.radio(
            "Fasting blood sugar > 120 mg/dL?",
            ["Yes", "No"],
            help="""
    This checks if your blood sugar level is high after not eating for 8–12 hours. 
    A value above 120 mg/dL may suggest prediabetes or diabetes, which are risk factors for heart disease.
    """
        )

        resting_ecg = st.selectbox(
            "Resting ECG result",
            ["Normal", "ST", "LVH"],
            help="""
    Results from an electrocardiogram (ECG), which measures the heart's electrical activity while at rest:

    - **Normal**: Heartbeat pattern looks healthy.
    - **ST**: Slight irregularities in heart rhythm, possibly due to temporary issues or stress.
    - **LVH (Left Ventricular Hypertrophy)**: Thickening of the heart's left wall, often from high blood pressure or overworking the heart.
    """
        )

        max_hr = st.number_input(
            "Maximum heart rate achieved",
            min_value=60, max_value=250, value=150,
            help="""
    The highest heart rate you reached during physical activity. 
    It helps doctors understand how your heart responds to exercise.
    """
        )

        

        oldpeak = st.number_input(
            "ST depression induced by exercise (Oldpeak)",
            min_value=0.0, max_value=10.0, value=1.0, step=0.1,
            help="""
    A measure from the ECG that shows how much the heart's electrical pattern changes after exercise. 
    Higher values may signal that the heart isn't getting enough blood during exertion.
    """
        )

        st_slope = st.selectbox(
            "Slope of the ST segment after exercise",
            ["Up", "Flat", "Down"],
            help="""
    Describes how the ST segment (part of an ECG reading) behaves after exercise:

    - **Up**: Normal — heart is recovering well.
    - **Flat**: May suggest the heart is under strain.
    - **Down**: Often a stronger sign of heart problems, like blocked arteries.
    """
        )

        submitted = st.form_submit_button("Predict Risk")


    if submitted:
        # Mappings
        sex_map = {"Male": 1, "Female": 0}
        chest_pain_map = {"ATA": 0, "NAP": 1, "ASY": 2, "TA": 3}
        resting_ecg_map = {"LVH": 0, "Normal": 1, "ST": 2}
        exercise_angina_map = {"No": 0, "Yes": 1}
        st_slope_map = {"Down": 0, "Flat": 1, "Up": 2}

        input_dict = {
            'Age': age,
            'Sex': sex_map[sex],
            'ChestPainType': chest_pain_map[chest_pain],
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': 1 if fasting_bs == "Yes" else 0,
            'RestingECG': resting_ecg_map[resting_ecg],
            'MaxHR': max_hr,
            'ExerciseAngina': exercise_angina_map[exercise_angina],
            'Oldpeak': oldpeak,
            'ST_Slope': st_slope_map[st_slope]
        }

        input_df = pd.DataFrame([input_dict])
        input_scaled = scaler.transform(input_df.values)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

        # Predicción
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            prob = output[0][1].item() * 100

        st.subheader("Prediction Result:")
        st.success(f"🩺 Estimated heart disease risk: **{prob:.2f}%**")

        # Semáforo de colores
        if prob < 30:
            st.markdown(
            "<h2 style='color: green;'>🟢 LOW RISK</h2>",
            unsafe_allow_html=True
            )
        elif 30 <= prob < 70:
            st.markdown(
                "<h2 style='color: orange;'>🟡 MODERATE RISK</h2>",
                unsafe_allow_html=True
            )
            st.text("Is recomended to have a visit with your doctor")
        else:
            st.markdown(
                "<h2 style='color: red;'>🔴 HIGH RISK</h2>",
                unsafe_allow_html=True
            )
            st.text("Is highly recommended to visit your doctor")



    # Definir explainer aquí antes de usarlo
        
        def model_wrapper(x_numpy):
            x_tensor = torch.from_numpy(x_numpy).float()
            with torch.no_grad():
                outputs = model(x_tensor)
            return outputs.numpy()

        # Definir explainer aquí antes de usarlo
        #explainer = shap.KernelExplainer(model_wrapper, X[:50], feature_names=feature_names)
        #shap_values = explainer.shap_values(input_scaled, nsamples=100)
        # Crear nuevo explainer compatible con waterfall plot
        explainer = shap.Explainer(model_wrapper, X[:50])
        shap_values = explainer(input_scaled)

        shap_arr = np.array(shap_values)
        # Extrae valores para la clase positiva (índice 1)
        if shap_arr.ndim == 3:
            shap_class1 = shap_arr[:, :, 1]
        else:
            shap_class1 = shap_arr

        # Contribuciones numéricas
        #contributions = dict(zip(feature_names, shap_class1[0]))
        #st.markdown("**Feature contributions (ordenado por magnitud):**")
        #for feat, val in sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True):
        #   st.write(f"- {feat}: {val:.4f}")

        # Gráfico SHAP para clase positiva
        #fig = plt.figure(figsize=(10, 4))
        #shap.summary_plot(shap_class1, input_scaled, feature_names=feature_names, plot_type="bar", show=False)
        #st.pyplot(fig)
        # SHAP devuelve una matriz de tamaño (1, num_features, num_classes)
    # Queremos la explicación del primer sample, clase 1 (enfermedad)
        single_explanation = shap.Explanation(
            values=shap_values.values[0][:, 1],
            base_values=shap_values.base_values[0][1],
            data=input_df.values[0],
            feature_names=feature_names
        )

        # Gráfico tipo Waterfall
        st.markdown("### 📉 SHAP Waterfall Plot")
        st.text("Shows how each feature pushes the risk prediction up or down.")

        fig = plt.figure(figsize=(10, 6))
        ax = plt.gca()

        # Dibujar el waterfall en el eje
        shap.plots.waterfall(single_explanation, show=False)

        # Capturar figura actual
        fig = plt.gcf()

        # Mostrar en Streamlit
        st.pyplot(fig)

        st.text("We recommend you to revise the important red features with your doctor")


    

else:
    display_saved_plots()


