import pickle
from flask import Flask, request, jsonify


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


dv, model = load_artifacts("./artifacts/model.bin")
app = Flask('capstone_project')


@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()
    prediction = predict_grape_quality(dv, model, client)

    
    result = {
        'predicted_class': str(prediction),
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8787)