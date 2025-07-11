# deep-learning
Credit Risk Prediction and Customer Segmentation
Credit Risk Prediction and Customer Segmentation
This project aims to build a deep learning model to predict credit risk and perform customer segmentation using KMeans clustering.

Dataset
The dataset used in this project is cs-training.csv, which is included in the archive (1).zip file. The dataset contains various features related to credit risk and customer information.

Project Structure
The notebook is structured as follows:

Data Extraction: Extracts the dataset from the zip file.
Data Loading and Preprocessing: Loads the data into a pandas DataFrame, renames columns, handles missing values, and scales the features.
Credit Risk Prediction Model: Builds and trains a deep learning model using TensorFlow/Keras for credit risk prediction.
Customer Segmentation: Applies KMeans clustering to segment customers based on their features.
Model and Scaler Saving: Saves the trained deep learning model, the scaler used for feature scaling, and the KMeans model.
Flask App (Optional): Includes code to create a simple Flask app for serving the models.

Getting Started
Clone the repository.
Open the notebook in Google Colab or a local Jupyter environment.
Run the cells sequentially to reproduce the analysis and models.
If you want to run the Flask app, you will need to install Flask and pyngrok (pip install flask pyngrok) and set up your ngrok authtoken.

Files
archive (1).zip: Zipped dataset.
cs-training.csv: Training data (extracted from zip).
Data Dictionary.xls: Data dictionary for the dataset.
credit_dl_model.h5: Saved deep learning model.
scaler.pkl: Saved StandardScaler object.
kmeans.pkl: Saved KMeans model.
credit_data_segmented.csv: Dataset with added segment labels.
Dataset----> https://drive.google.com/file/d/1WKEhlDG6AyjvbMvGo_F47FgRlqJn3bH3/view?usp=sharing

Dependencies
pandas
numpy
scikit-learn
tensorflow/keras
matplotlib
seaborn
joblib
flask (optional, for serving)
pyngrok (optional, for serving)

Usage
Run the notebook to train the models and perform segmentation.
Use the saved models (credit_dl_model.h5, scaler.pkl, kmeans.pkl) for making predictions on new data or deploying the models.
The credit_data_segmented.csv file contains the original data with an added 'Segment' column.
Author
[M.venkata snehith]
