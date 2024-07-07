#pip install matplotlib
#!pip install streamlit -q
#importing streamlit
import streamlit as st

#image or logo
from PIL import Image

#add photo or logo in website
img = Image.open('girl.jpg')

#tabs row title
st.set_page_config(page_title='BT CLASSIFIER')

#Importing Firebase
import firebase_admin
from firebase_admin import credentials
from firebase_admin import auth
cred = credentials.Certificate('radiant-hope-classifier-7c21d8a2f258.json')
firebase_admin.(cred)

#Creating a  login page
st.title(' Welcome to :violet[RadiantHope Classifier]')
choice = st.selectbox('Login/Signup', ['Login','Signup'])
if choice =='Login':
    email = st.text_input('Email Address')
    password = st.text_input('Password', type='password') 
    st.button('Login')
else:
    email = st.text_input('Email Address')
    password = st.text_input('Password', type='password')
    username = st.text_input('Enter Your Elegant User Name')
    if st.button('Create My Account'):
        user = auth.create_user(email = email, password = password, uid = username)
        st.success('Account Created Sucessfully')    
        st.markdown('Please Login Using Your Email and Password')     
# IMPORTING DEPENDENCIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.datasets import load_breast_cancer

#from google.colab import files
from sklearn.model_selection import train_test_split

# loading the data from sklearn
breast_cancer_dataset = load_breast_cancer()
print(breast_cancer_dataset)

# loading the data to a dataframe
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns = breast_cancer_dataset.feature_names)

#remove 20 columns
data_frame.drop(data_frame.columns[-20:], axis=1, inplace=True)

#adding the 'target' column to the dataframe
data_frame['label'] = breast_cancer_dataset.target
print(data_frame)

#CREATING SCREEN
col1,col2 = st.columns(2)
with col1:
  st.header('BREAST TUMOUR CLASSIFICATION USING DEEP LEARNING')#for Title
  st.error('_WOMEN CARE_')#second title
  st.info('Cure Sometimes, Treat Often, Comfort Always')#for paragraph
with col2:
  st.image(img)#for image

#Creating button and insert dataset in it
dingdong = st.checkbox('DataSet')
if dingdong:
  st.dataframe(data_frame)  

#"""***Building the predictive system***"""
st.header('_Please enter the Values_')
st.subheader("_Input Features_")

col3,col4,col5,col6,col7 = st.columns(5)
with col3:
  mean_radius = st.number_input("Mean Radius", value=20.0)
  mean_texture = st.number_input("Mean Texture", value=16.0)
with col4:
  mean_perimeter = st.number_input("Mean Perimeter", value=85.0)
  mean_area = st.number_input("Mean Area", value=550.0)
with col5:
  mean_smoothness = st.number_input("Mean Smoothness", value=0.1)
  mean_compactness = st.number_input("Mean Compactness", value=0.1)
with col6:
  mean_concavity = st.number_input("Mean Concavity", value=0.1)
  mean_concave_points = st.number_input("Mean Concave Points", value=0.05)
with col7:
  mean_symmetry = st.number_input("Mean Symmetry", value=0.1)
  mean_fractal_dimension = st.number_input("Mean Fractal Dimension", value=0.02)

# print the first 5 rows of the dataframe
data_frame.head()

# print last 5 rows of the dataframe
data_frame.tail()

# number of rows and columns in dataset
#data_frame.shape

# getting some information about the data
data_frame.info()

# Checking the missing values
data_frame.isnull().sum()

# Statistical measures about the data
data_frame.describe()

# Checking the distribution of target value
data_frame['label'].value_counts()

#"""1  ---->  Benign
#0  ---->  Malignant"""

# Grouping values for each label 1 & 0
data_frame.groupby('label').mean()

#"""SEPARATING THE FEATURES AND TARGET"""
X = data_frame.drop(columns= 'label', axis= 1)
Y = data_frame['label']
print(X)
print(Y)

#"""SPLITTING THE DATA INTO TRAINING DATA AND TESTING DATA"""
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

#"""***STANDARDIZE THE DATA***"""
from sklearn.preprocessing import StandardScaler

# Create an instance of the StandardScaler class.
scaler = StandardScaler()

# Fit the scaler to the training data.
X_train_std = scaler.fit_transform(X_train)

# Transform the test data using the fitted scaler.
X_test_std = scaler.transform(X_test)

# Print the shapes of the original data, the training data, and the test data.
print(X_train_std)

#"""**BUILDING THE NEURAL NETWORK**

# Importing Tensorflow and Keras

import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras

# Setting up the layers of Neural Network
model = keras.Sequential([
                          keras.layers.Flatten(input_shape=(10,)),
                          keras.layers.Dense(20, activation='relu'),
                          keras.layers.Dense(2, activation='sigmoid')
])

# compiling the neural network

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# training the Neural Network

history = model.fit(X_train_std, Y_train, validation_split=0.1, epochs=10)

#"""***VISUALIZATION OF ACCURACY AND LOSS***"""

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')

plt.legend(['training data', 'validation data'], loc = 'lower right')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')

plt.legend(['training data', 'validation data'], loc = 'upper right')

#"""***ACCURACY OF THE MODEL ON TEST DATA***"""

loss, accuracy = model.evaluate(X_test_std, Y_test)
#print(accuracy)

print(X_test_std.shape)
print(X_test_std[0])

Y_pred = model.predict(X_test_std)
print(Y_pred.shape)
print(Y_pred[0])

print(X_test_std)

print(Y_pred)

#"""**model.predict() gives me the prediction probability of each class for the data point**"""

# argmax function

my_list = [30, 20, 10]

index_of_max_value = np.argmax(my_list)
print(my_list)
print(index_of_max_value)

# Converting the prediction probability to class labels

Y_pred_labels = [np.argmax(i) for i in Y_pred]
print(Y_pred_labels)

## Put the inputs into a DataFrame
input_data = pd.DataFrame({
    "mean radius": [mean_radius],
    "mean texture": [mean_texture],
    "mean perimeter": [mean_perimeter],
    "mean area": [mean_area],
    "mean smoothness": [mean_smoothness],
    "mean compactness": [mean_compactness],
    "mean concavity": [mean_concavity],
    "mean concave points": [mean_concave_points],
    "mean symmetry": [mean_symmetry],
    "mean fractal dimension": [mean_fractal_dimension]
})

# change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the numpy array as we are predicting for one data point
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input_data
input_data_std = scaler.transform(input_data_reshaped)

prediction = model.predict(input_data_std)
print(prediction)

prediction_label = [np.argmax(prediction)]

Predict = st.button('CLASSIFY')
if Predict:
  if(prediction_label[0] == 0):
    st.error(f"The predicted tumor type is: MALIGNANT. You have to take more care about your Breast Health ")
  else:
    st.success(f"The predicted tumor type is: BENIGN. watch the video below to improve your Breast Health ")

#For Video Screening
st.header('_FIVE WAYS TO REDUCE YOUR RISK OF BREAST CANCER_')
bullet_points = [
    "--->  Maintain a Healthy Body Weight",
    "--->  Maintain an Active Lifestyle ",
    "--->  Limit Your Alcohol ",
    "--->  Breastfeed if possible ",
    "--->  Weigh the Risks and Benefits of Hormone Therapy for Menopause Symptoms ",
    "------------------------------------------------------------------------------",
    "***You should be aware of any changes in your breasts***",
    "***When you turn 50 get a mammogram for every 2 years until you're 74***",
    "------------------------------------------------------------------------------"
]
st.write(bullet_points)
video_file = open('Five_Ways_to_Reduce_your_Risk_for_Breast_Cancer(1080p).mp4','rb')
video_bytes = video_file.read()
st.video(video_bytes)
