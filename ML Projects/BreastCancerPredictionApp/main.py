import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def clean_data():
    df = pd.read_csv("/home/aayushmaan/PycharmProjects/BreastCancerPredictorApp/DATA/data.csv")
    data = df.drop(columns = ['Unnamed: 32','id'],axis =1)
    data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0}) # mapping M -> 1, B -> 0
    return data


def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")
    
    data = clean_data()
    
    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]
    
    input_values = {}
    
    for label , key in slider_labels:
        input_values[key] = st.sidebar.slider(
            label,
            min_value=float(data[key].min()),
            max_value=float(data[key].max()),
            value=float(data[key].mean()),
        )
    
    return input_values


def get_scaled_values(input_dict):
  data = clean_data()
  
  X = data.drop(['diagnosis'], axis=1)
  
  scaled_dict = {}
  
  for key, value in input_dict.items():
    max_val = X[key].max()
    min_val = X[key].min()
    scaled_value = (value - min_val) / (max_val - min_val)
    scaled_dict[key] = scaled_value
  
  return scaled_dict

def radar_chart(data):
    input_data = get_scaled_values(data)
    
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness','Compactness', 'Concavity', 'Concave Points','Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
            r=[
            input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
            input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
            input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']
            ],
            theta=categories,
            fill='toself',
            name='Mean Value'
    ))
    
    fig.add_trace(go.Scatterpolar(
            r=[
            input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
            input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
            input_data['concave points_se'], input_data['symmetry_se'],input_data['fractal_dimension_se']
            ],
            theta=categories,
            fill='toself',
            name='Standard Error'
    ))
    
    fig.add_trace(go.Scatterpolar(
            r=[
            input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
            input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
            input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']
            ],
            theta=categories,
            fill='toself',
            name='Worst Value'
    ))
    
    fig.update_layout(
        polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )),
        showlegend=True
    )
    
    return fig


def predictions(data):
    model = pickle.load(open("MODEL/model.pkl","rb"))
    scaler = pickle.load(open("MODEL/scaler.pkl","rb"))
    
    input_values = np.array(list(data.values())).reshape(1,-1)
    
    input_values_scaled = scaler.transform(input_values)

    prediction = model.predict(input_values_scaled)
    
    st.subheader("Cell Cluster prediction")
    st.write("Based on the provided measurements, the cell is predicted to be: ")
    
    if prediction[0] == 0:
        st.write("Benign")
    else:
        st.write("Malignant")
    
    st.write("Probability of being benign : ",model.predict_proba(input_values_scaled)[0][0])
    st.write("Probability of being malignant : ",model.predict_proba(input_values_scaled)[0][1])
    
    st.write("This app is used to assist professionals in their diagnosis. However it should NOT be used as a substitute for professionals")
    
    
def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon="👩🏻‍⚕️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    input_data = add_sidebar()
    
    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("This app predicts the likelihood of breast cancer (maglignant or benign) based on certain medical features.\nPlease connect this app to the cytology lab to help diagnose breast cancer from the tissue sample.\n")
    
    col1, col2 = st.columns([4,1])
    
    with col1:
        radar = radar_chart(input_data)
        st.plotly_chart(radar)
    
    with col2:
        predictions(input_data)


if __name__ == "__main__":
    main()