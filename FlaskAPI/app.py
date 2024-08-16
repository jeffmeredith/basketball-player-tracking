import flask
from flask import Flask, jsonify, request
import json
import numpy as np
import cv2
import io
from ultralytics import YOLO
import torch
import torch.nn as nn

player_model = YOLO('models/player_detect_model.pt', task='detect')
bball_model = YOLO('models/bball_detect_model.pt', task='detect')
court_model = YOLO('models/court_obb_model.onnx', task='obb')

class BasketballNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(BasketballNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

position_model = BasketballNet(input_size=9, output_size=3)
position_model.load_state_dict(torch.load('models/position_model.pth'))
position_model.eval()

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

    default_court = torch.tensor([[0.508744, 0.580394, 0.798212, 0.452258, 1.333351]])
    court_results = court_model(img)
    court_results = court_results[0].obb.xywhr
    if court_results.shape[0] == 0:
        court_results = default_court
    
    inputs_to_final_model = []

    # Combine each output from output1 with output3
    if bball_results.shape[0] > 0:
        for i in range(bball_results.shape[0]):
            combined_input = torch.cat((bball_results[i], court_results.squeeze(0)), dim=0)  # (4 + 5 = 9 floats)
            inputs_to_final_model.append(combined_input)

    # Combine each output from output2 with output3
    if player_results is not None:
        for i in range(player_results.shape[0]):
            combined_input = torch.cat((player_results[i], court_results.squeeze(0)), dim=0)  # (4 + 5 = 9 floats)
            inputs_to_final_model.append(combined_input)

    # Stack the inputs for batch prediction
    if inputs_to_final_model:
        final_inputs = torch.stack(inputs_to_final_model)
    else:
        final_inputs = torch.empty(0, 9)  # In case there are no valid inputs
    
    if final_inputs.shape[0] > 0:
        with torch.no_grad():
            final_outputs = position_model(final_inputs)
            final_outputs = final_outputs.tolist()
    else:
        final_outputs = None

    return jsonify({'prediction': final_outputs, 'bballs': bball_results.shape[0]})
