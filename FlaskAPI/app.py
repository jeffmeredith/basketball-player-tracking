import flask
from flask import Flask, jsonify, request
import json
import numpy as np
import cv2
import io
import onnxruntime as ort
from ultralytics import YOLO
import os


player_model = YOLO('models/player_detect_model.onnx', task='detect')
bball_model = YOLO('models/bball_detect_model.onnx', task='detect')
court_model = YOLO('models/court_obb_model.onnx', task='obb')
session = ort.InferenceSession('models/position_model.onnx')


app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['image']
    in_memory = io.BytesIO(file.read())
    img = cv2.imdecode(np.frombuffer(in_memory.getvalue(), np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'Unable to load image'}), 400
    
    player_results = player_model(img)
    player_results = player_results[0].boxes.xywh
    if player_results.shape[0] == 0:
        player_results = None

    bball_results = bball_model(img)
    bball_results = bball_results[0].boxes.xywh

    default_court = np.array([[0.508744, 0.580394, 0.798212, 0.452258, 1.333351]])
    court_results = court_model(img)
    court_results = court_results[0].obb.xywhr
    if court_results.shape[0] == 0:
        court_results = default_court
    
    inputs_to_final_model = []

    # Combine each output from output1 with output3
    if bball_results.shape[0] > 0:
        for i in range(bball_results.shape[0]):
            combined_input = np.concatenate((bball_results[i], np.squeeze(court_results, axis=0)), axis=0)
            inputs_to_final_model.append(combined_input)

    # Combine each output from output2 with output3
    if player_results is not None:
        for i in range(player_results.shape[0]):
            combined_input = np.concatenate((player_results[i], np.squeeze(court_results, axis=0)), axis=0)
            inputs_to_final_model.append(combined_input)

    # Stack the inputs for batch prediction
    if inputs_to_final_model:
        final_inputs = np.stack(inputs_to_final_model)
    else:
        final_inputs = np.empty((0, 9))  # In case there are no valid inputs
    
    if final_inputs.shape[0] > 0:
        inputs = {session.get_inputs()[0].name: final_inputs}
        final_outputs = session.run(None, inputs)[0].tolist()
    else:
        final_outputs = None

    return jsonify({'prediction': final_outputs, 'bballs': bball_results.shape[0]})