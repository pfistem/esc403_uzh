# Project: App Skin Cancer Classification
# Description: This file contains the code to train the CNN model for skin cancer classification.
# Author: Roman Stadler, Carolyne Huang, Rahel Eberle and Manuel Pfister
# Date: 2023-05-01
# License: MIT License
# Version: 1.0
# ======================================================================================================================

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import numpy as np
from PIL import Image
import io
import base64
from tensorflow.keras.models import load_model
import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html
from dash.exceptions import PreventUpdate
from dash import dash_table

# ======================================================================================================================

# Load the trained CNN model
model_path = 'www/model/skin_cnn_model.h5'
model = load_model(model_path)

# Load the label mapping
label_mapping_path = 'www/label_mapping.json'
with open(label_mapping_path, 'r') as f:
    label_mapping = json.load(f)

image_size = 64

# Create the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


# Define the app layout
app.layout = dbc.Container([
    html.Link(href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.0/css/all.min.css", rel="stylesheet"),
    dbc.Row([
        dbc.Col([
            html.Img(src='https://www.uzh.ch/docroot/logos/uzh_logo_d_pos.svg', style={'width': '200px', 'display': 'block', 'margin-bottom': '20px'})
        ], width=2, className='text-center'),
        dbc.Col([
            html.H1('Skin Lesion Classifier | University of Zurich', className='mt-5 mb-5 text-center')
        ], width=10)
    ]),
    dbc.Row([
        dbc.Col([
            dash_table.DataTable(
                id='lesion-class-table',
                columns=[
                    {'name': 'Lesion Type', 'id': 'lesion-type'},
                    {'name': 'Abbreviation', 'id': 'abbreviation'}
                ],
                data=[
                    {'lesion-type': 'Actinic keratoses and intraepithelial Carcinoma / Bowen’s disease', 'abbreviation': 'akiec'},
                    {'lesion-type': 'Basal cell carcinoma', 'abbreviation': 'bcc'},
                    {'lesion-type': 'Benign keratosis-like lesions', 'abbreviation': 'bkl'},
                    {'lesion-type': 'Dermatofibroma', 'abbreviation': 'df'},
                    {'lesion-type': 'Melanoma', 'abbreviation': 'mel'},
                    {'lesion-type': 'Melanocytic nevi', 'abbreviation': 'nv'},
                    {'lesion-type': 'Vascular lesions', 'abbreviation': 'vasc'},
                    {'lesion-type': 'Unknown', 'abbreviation': 'others'}
                    
                ],
                style_cell={
                    'textAlign': 'left'
                },
                style_header={
                    'fontWeight': 'bold'
                },
            ),
        ], width=8, className='mx-auto mb-5')
    ]),
        
    dbc.Row([
        dbc.Col([
            dcc.Upload(
                id='upload-image',
                children=html.Div(['Drag and drop or click to select an image in JPG format.']),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                accept='.jpg'
            ),
            html.Div(id='output-image-upload', className='text-center'),
            html.Div(id='output-image-preview', className='text-center'),  # Image preview
            html.Div(id='output-prediction', className='text-center mt-5')
    ], width=8, className='mx-auto')
    ]),

    dbc.Row([
        dbc.Col([
            html.P(
                "Disclaimer: This tool is provided for informational purposes only and is not intended as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition or skin issue.",
                className='text-center mt-5 mb-5',
                style={
                    'font-size': '0.9rem',
                    'color': 'red',
                    'padding-left': '20px',
                    'padding-right': '20px'
                }
            ),
        ])
    ]),
])

# Define a function to process the uploaded image and make a prediction
def process_image(image):
    # Resize and convert the image to an array
    image = image.resize((image_size, image_size))
    image = np.array(image)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    # Make a prediction
    prediction = model.predict(image)
    # Get the predicted class and probability
    predicted_class = np.argmax(prediction)
    predicted_probability = np.max(prediction)
    predicted_label = list(label_mapping.keys())[list(label_mapping.values()).index(predicted_class)]
    # Return all predictions
    predicted_probabilities = prediction[0]
    return predicted_label, predicted_probability, predicted_probabilities

@app.callback(
    Output('output-image-preview', 'children'),
    Input('upload-image', 'contents')
)
def update_image_preview(image_contents):
    if image_contents is not None:
        img_html = html.Img(src=image_contents, style={'max-width': '100%', 'max-height': '300px'})
        return img_html

@app.callback(
    Output("output-prediction", "children"),
    Input("upload-image", "contents"),
)
def make_prediction(upload_contents):
    ctx = dash.callback_context
    image_data = None

    if not ctx.triggered:
        raise PreventUpdate
    else:
        input_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if input_id == "upload-image":
        image_data = upload_contents

    if image_data is not None:
        content_type, content_string = image_data.split(",")
        decoded = base64.b64decode(content_string)
        image = Image.open(io.BytesIO(decoded))
        predicted_label, predicted_probability, predicted_probabilities = process_image(image)
        return html.Div(
            [
                html.H3("Predicted Class: {}".format(predicted_label)),
                html.H3("Probability: {:.2f}%".format(predicted_probability * 100)),
                html.Ul([
                    html.Li("{}: {:.2f}%".format(label, prob*100), style={'display': 'inline', 'padding-right': '10px'}) for label, prob in zip(label_mapping.keys(), predicted_probabilities)
                ], style={'list-style-type': 'none'})
            ]
        )
    raise PreventUpdate

#Run the app
server = app.server
if __name__ == "__main__":
    app.run_server(debug=False, port=8000, host='0.0.0.0')