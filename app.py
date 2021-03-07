from flask import Flask, request, render_template, send_from_directory
from flask_cors import CORS
from ml import preprocess, predict
from storage import drive

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

        image_name = predict.do_cause(prediction_dict)

        return preprocess.dump({
            'message': predict.do_predict(scaled).tolist()[0], 
            'review': image_name, 

        })
    else:
        return render_template('index.html')

@app.route('/status', methods = ['GET']) 
def status(): 
    return preprocess.dump({'accuracy': 96.7}); 

@app.route('/getMedia', methods = ['POST'])
def get_media():
   
    file_dict = preprocess.parse(request.data)
    print(file_dict)
    file = file_dict['imageName']

    return preprocess.dump({
        'link': drive.get_file_link(file)
    })


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)