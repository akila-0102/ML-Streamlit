import os
import streamlit as st
import pickle
import numpy as np

# Get the absolute path to the pickle file
pickle_file_path = os.path.join(os.getcwd(), '203.pkl')

# Check if the file exists
if os.path.exists(pickle_file_path):
    with open(pickle_file_path, 'rb') as f:
        model = pickle.load(f)
else:
    st.error("Error: pickle file '203.pkl' not found.")
    st.stop()

def predict_rings(Sex, Length, Diameter, Height, WholeweightShuckedweight, Visceraweight, Shellweight, Age):
    input_data = np.array([[Sex, Length, Diameter, Height, WholeweightShuckedweight, Visceraweight, Shellweight, Age]]).astype(np.float32)  # Adjust data type if needed
    print("Input data:", input_data)  # Debugging: Print input data
    print("Input data shape:", input_data.shape)  # Debugging: Print input data shape
    prediction = model.predict(input_data.reshape(1,-1))
    # pred = '{0:.{1}f}'.format(prediction, 2)
    return prediction


def main():
    st.title("Streamlit")
    html_temp = """
    <div style="background-color:black ;padding:10px">
    <h2 style="color:white;text-align:center;"> Abalone Age Prediction</h2>
    </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)

    Sex = st.number_input("Sex")
    Length = st.number_input("Length")
    Diameter = st.number_input("Diameter")
    Height = st.number_input("Height")
    WholeweightShuckedweight = st.number_input("WholeweightShuckedweight")
    Visceraweight = st.number_input("Visceraweight")
    Shellweight = st.number_input("Shellweight")
    Age = st.number_input("Age")
    
    safe_html = """
       <div style="background-color:grey ;padding:10px">
       <h2 style="color:white;text-align:center;">The abalones are rings</h2>
       </div>
    """

    danger_html = """
       <div style="background-color:grey ;padding:10px">
       <h2 style="color:white;text-align:center;">The abalones are not rings</h2>
       </div>
    """

    if st.button("Predict"):
        output = predict_rings(Sex, Length, Diameter, Height, WholeweightShuckedweight, Visceraweight, Shellweight, Age)
        st.success("The predicted age of the abalone is {}".format(output))

        if output > 0.5:
            st.markdown(danger_html, unsafe_allow_html=True)
        else:
            st.markdown(safe_html, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
