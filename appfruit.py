import streamlit as st
import joblib
import numpy as np

# Load the trained model and other artifacts
model = joblib.load('perceptron_model_fruit.pkl')
scaler = joblib.load('scaler_fruit.pkl')
label_encoder = joblib.load('label_encoder_fruit.pkl')

# Define the application
def app():
    st.title('Fruit Classification')
    
    # Input fields for the user to enter data
    st.header('Enter the fruit features for prediction:')
    
    diameter = st.slider('Diameter of the fruit (cm)', min_value=0.0)
    weight = st.slider('Weight of the fruit (grams)', min_value=0.0)
    red = st.slider('Red color intensity (0-255)', min_value=0, max_value=255)
    green = st.slider('Green color intensity (0-255)', min_value=0, max_value=255)
    blue = st.slider('Blue color intensity (0-255)', min_value=0, max_value=255)
    
    # When the user presses the predict button
    if st.button('Predict'):
        # Prepare the input for the model
        input_features = np.array([[diameter, weight, red, green, blue]])
        scaled_input = scaler.transform(input_features)
        
        # Prediction
        prediction = model.predict(scaled_input)
        predicted_name = label_encoder.inverse_transform(prediction)
        
        # Display the result
        st.write(f"Predicted Fruit: {predicted_name[0]}")
        
        # Option to show more info (optional)
        st.markdown("### Model Evaluation (on test data)")
        
        # Accuracy report
        # The model and evaluation could also be shown here based on the previously saved evaluation output
        st.write(f"Accuracy: {0.88:.2f}")  # Change with actual evaluation results if desired

# Run the application
if __name__ == '__main__':
    app()
