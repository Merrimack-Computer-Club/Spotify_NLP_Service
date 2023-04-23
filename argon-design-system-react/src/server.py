from flask import Flask, request
app = Flask(__name__)

selected_inputs = []

@app.route('/api/save_input', methods=['POST'])
def save_input():
    data = request.get_json()
    selected_inputs.append(data)
    return 'Input saved successfully!'

if __name__ == '__main__':
    app.run()