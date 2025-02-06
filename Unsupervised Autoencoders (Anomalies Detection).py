# chatgpt : example Autoencoders when work as a data scientist in bank using pandas 

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# Sample financial data (e.g., transactions)
data = {
    'transaction_amount': [200, 150, 300, 250, 220, 800, 120, 350, 400, 950],
    'account_balance': [1200, 1500, 1300, 1400, 1350, 2000, 1150, 1500, 1600, 1800],
    'time_of_transaction': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}

# Convert to pandas DataFrame
df = pd.DataFrame(data)

# Standardize the features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)


#=============================================Define the Autoencoder Model

# Define the autoencoder architecture
input_dim = df_scaled.shape[1]  # Number of features
encoding_dim = 2  # Number of dimensions for the encoded representation

# Define the input layer
input_layer = Input(shape=(input_dim,))

# Encoder layer(s)
encoded = Dense(encoding_dim, activation='relu')(input_layer)

# Decoder layer(s)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

# Build the autoencoder model
autoencoder = Model(input_layer, decoded)

# Compile the model
autoencoder.compile(optimizer=Adam(), loss='mean_squared_error')

# Print model summary
autoencoder.summary()


#=============================================Train the Autoencoder

# Train the model (use data itself as both input and target for reconstruction)
autoencoder.fit(df_scaled, df_scaled, epochs=100, batch_size=2, shuffle=True, validation_data=(df_scaled, df_scaled))


#=============================================Use the Autoencoder for Anomaly Detection

# Get the reconstructed data
reconstructed = autoencoder.predict(df_scaled)

# Calculate the reconstruction error (Mean Squared Error)
mse = np.mean(np.power(df_scaled - reconstructed, 2), axis=1)

# Set a threshold to classify anomalies (e.g., top 10% errors are anomalies)
threshold = np.percentile(mse, 90)

# Classify anomalies
anomalies = mse > threshold

# Output anomalies
print("Anomalies detected:", anomalies)


#=============================================Visualize the results

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.plot(mse, label='Reconstruction Error')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.title('Reconstruction Error vs Threshold')
plt.xlabel('Data Points')
plt.ylabel('Reconstruction Error')
plt.legend()
plt.show()

#Explanation of the Approach
# 1.Preprocessing: Data is normalized using StandardScaler to ensure that all features contribute equally to the model training.
# 2.Autoencoder: The autoencoder model is designed with a simple encoder-decoder architecture, using Dense layers for the compression and decompression of features.
# 3.Anomaly Detection: After training the model, the reconstruction error for each data point is calculated. Points with a high error are likely to be anomalies (e.g., fraudulent transactions).
# 4.Thresholding: You set a threshold (e.g., top 10% of errors) to classify transactions as anomalies.

#This type of model can be useful for detecting unusual customer behavior, fraudulent transactions, or identifying patterns that differ from normal financial activity.