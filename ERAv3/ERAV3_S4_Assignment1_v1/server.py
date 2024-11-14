from flask import Flask, render_template, jsonify
import json
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/progress')
def get_progress():
    try:
        with open('static/progress.json', 'r') as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify({'train_losses': [], 'test_accuracies': []})

@app.route('/predictions')
def get_predictions():
    try:
        with open('static/predictions.json', 'r') as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify([])

if __name__ == '__main__':
    app.run(debug=True) 