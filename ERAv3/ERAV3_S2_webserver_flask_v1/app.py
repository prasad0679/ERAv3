from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.json.get('message')
    # Simulate AI processing
    ai_response = f"AI says: {user_input[::-1]}"  # Reverse the user's message
    return jsonify({'response': ai_response})

if __name__ == '__main__':
    app.run(debug=True)
