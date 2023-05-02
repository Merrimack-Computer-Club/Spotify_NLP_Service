from imports import *
import webscraping

app = Flask(__name__)
CORS(app)

@app.route("/api/save_input", methods=["POST"])
def save_input():
    data = request.get_json()
    # Do something with data
    return jsonify({"Server side received data": data}), 200

@app.route("/api/emotions/post/list", methods=["POST"])
def get_Emotions_For_A_List_Of_Songs():
    data = request.get_json()

    classified = webscraping.read_top_songs(data)

    # Get all the musixmatch song information
    for i in range(1):
        webscraping.scrape_song(classified[i])

    return {}, 200#{'response': map(lambda song: song.toJson(), classified) }), 200

if __name__ == "__main__":
   print("Spotify Emotions Server Started \nPress Ctrl+C to stop the server \nServing..")
   serve(app, host="0.0.0.0", port=8021)
