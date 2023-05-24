import json
from flask import Flask, jsonify, request, flash
from datetime import datetime
import helper_file 
import logging


logging.basicConfig(filename='NLP-log.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

app = Flask(__name__)

@app.route('/')
def index():
    app.logger.info('Info level log')
    return jsonify({'Member Names': 'Shabnam Harjeet Aman TingTing Deepesh'})

@app.route('/input', methods = ['POST', 'GET'])
def input_func():
    #take the input from get or post method
    if request.method == 'GET':
        if 'file' not in request.files:
            flash('No file part')
        else:
            file = request.files['file']
            # save file in local directory
        #     file.save(file.filename)

        output,time_list,model_name,hyperparams = helper_file.input_output(file)
        # app.logger.info('Info level log')
        # app.logger.info(f'Info Level log: {jsonify({"output":output,"time_list":time_list,"model_name":model_name})}')
        log = logging.getLogger("log-output-size")

        log.info(f'Info Level log: {jsonify({"output":output,"time_list":time_list,"model_name":model_name})}')

        log = logging.getLogger("Input--output")
        log.info(f'"output": {output}')

        log = logging.getLogger("Model-Name")
        log.info(f'{model_name}')

        log = logging.getLogger("Time-Taken")
        log.info(f'{time_list}')

        log = logging.getLogger("Hyperparameters")
        log.info(f'{hyperparams}')

        return jsonify ({"output":output,"time taken":time_list,"model_name":model_name})

    else:
        return "data not accepted"

if __name__ == "__main__":
    # Launch the Flask dev server
    app.run(host="127.0.0.1", debug=True)

# app.run()