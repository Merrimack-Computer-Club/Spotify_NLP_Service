from imports import *
import webscraping
import Graph_Code

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
    for val in classified:
        webscraping.scrape_song(val)
        #print(classified[i].lyrics)

    # Run Each song through the model
    

    # Construct a DataFrame for emotions from the songs
    emotions = [song.emotions for song in classified]
    df = pd.DataFrame({
        'emotions': emotions
    }) 

    # Send the Dataframe to the Graph_Code function -> Base64 encoded image
    b64encoded_string = Graph_Code.construct_Song_Emotions_Graph(emotions)

    return {'base64_encoded_gimage': b64encoded_string}, 200#{'response': map(lambda song: song.toJson(), classified) }), 200

if __name__ == "__main__":
   print("Spotify Emotions Server Started \nPress Ctrl+C to stop the server \nServing..")
   serve(app, host="0.0.0.0", port=8080)
