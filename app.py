from flask import Flask, request, render_template, send_from_directory
from flask_cors import CORS
from ml import preprocess, predict

app = Flask(__name__)
CORS(app)

@app.route('/assets/<path:path>')
def send_js(path):
    return send_from_directory('static/assets', path)


@app.route('/', methods = ['POST', 'GET'])
def index():
    if request.method == 'POST':
        prediction_dict = preprocess.transform(request.data)
        
        scaled = preprocess.scaler([list(prediction_dict.values())])

        return preprocess.dump({
            'message': predict.do_predict(scaled).tolist()[0], 
            'review': predict.do_cause(prediction_dict)
        })
    else:
        return render_template('index.html')

@app.route('/status', methods = ['GET']) 
def status(): 
    return preprocess.dump({'accuracy': 95.02}); 

if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)