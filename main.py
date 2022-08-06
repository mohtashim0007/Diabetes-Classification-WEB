


import streamlit as st
import sklearn
import pickle

#unpickle model for Random Forest
model = pickle.load(open('modelRF.pkl', 'rb'))
#Unpickle model for SVM
model2 = pickle.load(open('modelSVM.pkl', 'rb'))

df = pickle.load(open('df.pkl', 'rb'))

#Unpicling dataframe to show on web
test_x = pickle.load(open('test_x.pkl', 'rb'))
test_y = pickle.load(open('test_y.pkl', 'rb'))
test_y.reset_index(drop = True, inplace = True)

def isDiabetic(val):
  if val == 1:
    return 'Person is Diabetic.'
  else:
    return 'Person is NOT Diabetic.'



# WEB CODING

st.header('Diabetes Prediction Classification Model')
st.text(' ')


radio = st.sidebar.radio('Main Menu', ['Use Testing Data', 'Use Custom Data'])

#TESTING DATA RADIO BUTTON

if radio == 'Use Testing Data':
  ind = st.slider('Slide to select index value for Testing Data : ', min_value = 0, max_value = 153)
  out = isDiabetic(test_y[ind])
  st.write('Labeled Outcome for the above Data is ', out)

  st.subheader("Model Prediction")

  radio2 = st.radio('Select Model : ',['Random Forest Model','SVM Model'], key = "2" )
  if radio2 == 'Random Forest Model':
    st.write('Random Forest Predicted : ', isDiabetic(model.predict([test_x[ind]])))
    st.write('Random Forest Model Testing Data Accuracy = 99.34%')
    st.write('Random Forest Model Training Data Accuracy = 85.06%')

  else :
    st.write('SVM Predicted : ', isDiabetic(model2.predict([test_x[ind]])))
    st.write('SVM Model Testing Data Accuracy = 79.64%')
    st.write('SVM Model Training Data Accuracy = 83.12%')


elif radio == 'Use Custom Data':
  col1, col2 = st.columns([1,1])

  #NOrmalization method X_new = (X - X_min)/(X_max - X_min)
  with col1:
    Glucose = st.number_input('Glucose :',  df['Glucose'].min(), df['Glucose'].max(), value = df['Glucose'].mean(), step = 3.4)
    Glucose = (Glucose - df['Glucose'].min()) / (df['Glucose'].max() - df['Glucose'].min())
    BloodPressure = st.number_input('BloodPressure :',  df['BloodPressure'].min(), df['BloodPressure'].max(), value = df['BloodPressure'].mean(), step = 3.4)
    BloodPressure = (BloodPressure - df['BloodPressure'].min()) / (df['BloodPressure'].max() - df['BloodPressure'].min())
    SkinThickness = st.number_input('SkinThickness :',  df['SkinThickness'].min(), df['SkinThickness'].max(), value = df['SkinThickness'].mean(), step = 3.4)
    SkinThickness = (SkinThickness - df['SkinThickness'].min()) / (df['SkinThickness'].max() - df['SkinThickness'].min())
    Insulin = st.number_input('Insulin :',  df['Insulin'].min(), df['Insulin'].max(), value = df['Insulin'].mean(), step = 3.4)
    Insulin = (Insulin - df['Insulin'].min()) / (df['Insulin'].max() - df['Insulin'].min())

  with col2:
    BMI = st.number_input('BMI :',  df['BMI'].min(), df['BMI'].max(), value = df['BMI'].mean(), step = 3.4)
    BMI = (BMI - df['BMI'].min()) / (df['BMI'].max() - df['BMI'].min())
    Age = st.number_input('Age :',  df['Age'].min(), df['Age'].max(), value = 27, step = 4)
    Age = (Age - df['Age'].min()) / (df['Age'].max() - df['Age'].min())
    DiabetesPedigreeFunction = st.number_input('DiabetesPedigreeFunction :',  df['DiabetesPedigreeFunction'].min(), df['DiabetesPedigreeFunction'].max(), value = df['DiabetesPedigreeFunction'].mean() , step = 3.4)
    DiabetesPedigreeFunction = (DiabetesPedigreeFunction - df['DiabetesPedigreeFunction'].min()) / (df['DiabetesPedigreeFunction'].max() - df['DiabetesPedigreeFunction'].min())
    Pregnancies = st.number_input('Pregnancies :',  df['Pregnancies'].min(), 9, value = 0,  step = 1)
    Pregnancies = (Pregnancies - df['Pregnancies'].min()) / (df['Pregnancies'].max() - df['Pregnancies'].min())

  bt = st.button('Submit')

  if bt:
    vals = [Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]
    st.write('Random Forest Predicted : ', isDiabetic(model.predict([vals])))
    st.write('SVM Predicted : ', isDiabetic(model2.predict([vals])))
    


