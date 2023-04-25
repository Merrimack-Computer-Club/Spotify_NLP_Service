from imports import *

app = Flask(__name__)
CORS(app)

@app.route("/api/save_input", methods=["POST"])
def save_input():
    data = request.get_json()
    # Do something with data
    return jsonify({"Server side received data": data}), 200


if __name__ == "__main__":
   print("Spotify Emotions Server Started \nPress Ctrl+C to stop the server \nServing..")
   serve(app, host="0.0.0.0", port=8080)
