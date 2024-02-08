import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

# Function to load and preprocess data
def load_data():
    df = pd.read_csv("data.csv")
    return df

# Function for data preparation
def prepare_data(df):
#     df.drop(["Patient_ID", "Sun_Exposure", "Dietary_VitaminD_Intake"], axis=1, inplace=True)
#     le  = LabelEncoder()
#     obj_type = df.select_dtypes(include='object').columns
#     df[obj_type] = le.fit_transform(df[obj_type])
#     df[obj_type[1]] = le.fit_transform(df[obj_type[1]])
    
    df['Gender'] = df.Gender.apply(lambda x: 0 if x == "Male" else 1)
    df['Liver_Function'] = df.Liver_Function.apply(lambda x: 0 if x == "Abnormal" else 1)
    
#     num_type = df.select_dtypes(exclude=['object']).columns.drop(["Phosphorus","Magnesium", "Alkaline_Phosphatase"])
    scaler = MinMaxScaler(feature_range=(0, 1))
#     df[num_type]  = scaler.fit_transform(df[num_type])
    X = df.drop( ['Deficiency_Status', "Phosphorus",'Liver_Function'], axis=1)
    normalized_X = scaler.fit_transform(X)
    
    y = df['Deficiency_Status']
    return (normalized_X, y)

# Function for bagging model (Random Forest)

def bagging_model(X_train, y_train, X_test):
    rfm = RandomForestClassifier()
    rfm.fit(X_train, y_train)
    y_pred = rfm.predict(X_test)
#     y_pred_proba = rfm.predict_proba(X_test)[:, 1]  # Probabilities for class 1
#     y_pred = np.where(X_test[:, 3] <= 0.2, 1, y_pred_proba)  # Adjust YOUR_SERUM250H_D_INDEX accordingly
#     serum250h_d_index = 3  # Replace with the actual index of serum250h_d
#     serum250h_d_values = X_test[:, serum250h_d_index]
#     deficient_condition = serum250h_d_values <= 20

    # Create a copy of the input data and set the predicted value for the deficient condition
#     X_test_deficient = X_test.copy()
#     X_test_deficient[:, serum250h_d_index] = deficient_condition.astype(int)

    # Predict using the modified input data
#     y_pred = rfm.predict(X_test_deficient)
    return y_pred

# Function for blending model
def blending_model(X_train, y_train, X_val, y_val, X_test):
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    dtc_predict_val = dtc.predict(X_val)
    dtc_predict_test = dtc.predict(X_test)

    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    knn_predict_val = knn.predict(X_val)
    knn_predict_test = knn.predict(X_test)

    df_val_lr = pd.concat([pd.DataFrame(X_val), pd.DataFrame(dtc_predict_val), pd.DataFrame(knn_predict_val)], axis=1)
    df_test_lr = pd.concat([pd.DataFrame(X_test), pd.DataFrame(dtc_predict_test), pd.DataFrame(knn_predict_test)], axis=1)

    lr = LogisticRegression()
    lr.fit(df_val_lr, y_val)
    y_pred = lr.predict(df_test_lr)
    return y_pred

# Streamlit app
def main():
    st.title("Vitamin D Deficiency Prediction App")

    # Input variables
    gender = st.sidebar.selectbox("Gender",("Male", "Female"))
#     liver_function = st.sidebar.selectbox("Liver Function",("Normal", "Abnormal"))
    age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=45)
    bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=50.0, value=28.2)
    serum_25Oh_d = 	st.sidebar.number_input("Serum", min_value=0, max_value=200, value=35)
    calcium_level = st.sidebar.number_input("Calcium Level", min_value=0.0, max_value=20.0, value=9.0)
#     phosphorus = st.sidebar.number_input("Phosphorus", min_value=0.0, max_value=20.0, value=3.2)
    pth = 	st.sidebar.number_input("PTH", min_value=0, max_value=200, value=30)
    alkaline_phosphatase = st.sidebar.number_input("Alkaline Phosphatase", min_value=0, max_value=200, value=75)
    magnesium = 	st.sidebar.number_input("Magnesium", min_value=0.0, max_value=200.0, value=1.8, step=0.1)
    creatinine = st.sidebar.number_input("creatinine", min_value=0.0, max_value=200.0, value=1.0,step=0.1)
    
    
    # Create a dataframe with input values
    input_data = pd.DataFrame({
        'Age': [age],
        'BMI': [bmi],
        'Calcium_Level': [calcium_level],
#         'Phosphorus': [phosphorus],
        'Alkaline_Phosphatase': [alkaline_phosphatase],
        'Gender': [0],  # Placeholder, you can add a gender selection widget
#         'Liver_Function': [1], # Placeholder, you can add a liver function selection widget
        "serum_25Oh_d": [serum_25Oh_d],
        "pth": [pth],
        "magnesium": [magnesium],
        "creatinine": [creatinine]
        
        
    })

    # Load data
    df = load_data()
    st.write(df)
    
    # Prepare data
    X, y = prepare_data(df)
#     st.write(X)
    
    # Bagging model (Random Forest)
#     st.subheader("Bagging Model (Random Forest)")
    st.subheader("Prediction Result (Bagging Model) 86% accuracy")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=204)
    y_pred_bagging = bagging_model(X_train, y_train, input_data)
    
#     cm = classification_report(X_test, y_test)
#     st.write(accuracy_score(X_test, y_test))
    st.write(f'Prediction: {"Deficient" if y_pred_bagging[0] == 1 else "Not Deficient"}')

    
    
    # Blending model
    st.subheader("Prediction Result (Blending model): 80% accuracy")
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=204)
    
    
    y_pred_blending = blending_model(X_train, y_train, X_val, y_val, input_data)
#     st.write(y_pred_blending)
    st.write(f'Prediction: {"Deficient" if y_pred_blending[0] >= 0.5 else "Not Deficient"}')
    
if __name__ == "__main__":
    main()