import pickle
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

def load_artifacts(path):
    """
    Load trained model and fitted vectorizer

    Args:
        path (str): path to model artifacts

    Returns:
        Tuple(Vectorizer, Model): Tuple containing vectorizer and model.
    """
    with open(path, 'rb') as f_in:
        artifact = pickle.load(f_in)
    return artifact


def predict_grape_quality(dv, model, grape_info):
    """
    Generate prediction

    Args:
        dv (Vectorizer): fitted dict vectorizer
        model (Model object): trained model
        grape_info (dict): Dictionary with grape features

    Returns:
        float: class of grape_info
    """
    X = dv.transform([grape_info])  
    y_pred = model.predict(X)
    return y_pred


# Home route with form input
@app.route('/')
def home():
    return render_template('form.html')


@app.route('/predict', methods=['POST'])
def grapify():
    # Load model artifacts
    dv, model = load_artifacts("./artifacts/model.bin")
    if request.is_json:
        data_dict = request.get_json()
        prediction = predict_grape_quality(dv, model, data_dict)[0]
        result = {'predicted_class': prediction}
        return jsonify(result), 200
    else:
        # Extract data from the form
        sugar_content_brix = float(request.form['sugar_content_brix'])
        acidity_ph = float(request.form['acidity_ph'])
        cluster_weight_g = float(request.form['cluster_weight_g'])
        berry_size_mm = float(request.form['berry_size_mm'])
        sun_exposure_hours = float(request.form['sun_exposure_hours'])
        soil_moisture_percent = float(request.form['soil_moisture_percent'])
        rainfall_mm = float(request.form['rainfall_mm'])
        harvest_day = int(request.form['harvest_day'])
        harvest_month = int(request.form['harvest_month'])
        variety = request.form['variety']
        region = request.form['region']          

        # Simulate a prediction
        data_dict = {
                    'sugar_content_brix': sugar_content_brix,
                    'acidity_ph': acidity_ph,
                    'cluster_weight_g': cluster_weight_g,
                    'berry_size_mm': berry_size_mm,
                    'sun_exposure_hours': sun_exposure_hours,
                    'soil_moisture_percent': soil_moisture_percent,
                    'rainfall_mm': rainfall_mm,
                    'harvest_day': harvest_day,
                    'harvest_month': harvest_month,
                    'variety': variety,
                    'region': region,
                }
        prediction = predict_grape_quality(dv, model, data_dict)[0]

        return render_template('form.html', prediction=prediction), 200

if __name__ == '__main__':
    print(app.instance_path)
    app.run(debug=True, host='0.0.0.0', port=8787)
