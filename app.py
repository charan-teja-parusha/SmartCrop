from flask import Flask, render_template, request, jsonify
import joblib
import sklearn
import numpy as np

app = Flask(__name__)

# load the pre-trained model and scaler
model = joblib.load('./model.joblib')

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('home.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    print("REQUESTJSON")
    if request.method == 'POST':
        try:
            # retrieve the slider values from the form
            feature1 = float(request.form['slider1'])
            feature2 = float(request.form['slider2'])
            feature3 = float(request.form['slider3'])
            feature4 = float(request.form['slider4'])
            feature5 = float(request.form['slider5'])
            feature6 = float(request.form['slider6'])
            feature7 = float(request.form['slider7'])
            print(feature1)
            # make a prediction using the pre-trained model
            features_list=[feature1,feature2,feature3,feature4,feature5,feature6,feature7]
            features = np.array(features_list)
            features_reshaped_array = features.reshape((1,-1))
            prediction = model.predict(features_reshaped_array)
            return jsonify(str(prediction[0]))
        except Exception as e:
            # Debugging code to print out error message
            print(f"Error: {str(e)}")
            return jsonify(str(e))
    else:
        # render the template with empty sliders
        return render_template('home.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
