from imports import *

app = Flask(__name__)
CORS(app)


@app.route("/api/save_input", methods=["POST"])
def save_input():
    data = request.get_json()
    # Do something with data
    return jsonify({"Server side received data": data}), 200


if __name__ == "__main__":
    app.run(debug=True)
