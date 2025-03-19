# Neural Network-based Room Occupancy Detection

## Overview
This project implements a neural network (NN) classifier to predict the number of occupants in a room based on sensor data collected from an IoT-based system. The dataset includes sensor readings such as temperature, light, CO2 levels, and motion data, used to classify the number of people in the room.

## Features
- **Binary Classification**: Detects if there are more than two persons in the room.
- **Multi-Class Classification**: Predicts the number of people in the room (0, 1, 2, or 3).
- **IoT Sensor Data Processing**: Utilizes temperature, light, CO2, and motion data to make predictions.
- **Data Preprocessing**: Handles missing values, noise, and imbalances in the dataset.
- **Neural Network Training**: Implements an MLP model to optimize the classifier’s performance.
- **Evaluation**: Uses confusion matrices, precision, recall, and F1-scores to assess model accuracy.

## Dataset
The dataset, **Lab6Dataset.csv**, consists of over 10,000 records taken over several days, with each record containing:
- Date and time
- Sensor data: temperature, light intensity, CO2 levels, PIR motion data
- The number of occupants (0-3)

### Data Features:
- `Si_Temp`: Temperature (°C)
- `Si_Light`: Light intensity (Lux)
- `CO2`: CO2 levels (PPM)
- `PIRi`: PIR sensor boolean (motion detected or not)
- `Number of persons`: The actual number of persons in the room (0, 1, 2, 3)

## Installation & Setup

### Prerequisites
- Python 3.x
- Required libraries:  
  ```sh
  pip install numpy pandas scikit-learn tensorflow

### Clone Repository
  ```sh
  git clone <repository-url>
  cd Neural-Network-Occupancy-Detection
  ```

## Running the model
  ```sh
  python main.py
  ```

## How it works
- Data Collection: Data is gathered from IoT sensors monitoring temperature, light, CO2, and motion every 30 seconds.
- Preprocessing: The dataset is cleaned and features are extracted.
- Model Training: A neural network is trained to predict the number of occupants using sensor data.
- Prediction: The trained model predicts room occupancy from new sensor data.

