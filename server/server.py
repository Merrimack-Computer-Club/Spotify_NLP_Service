from imports import *
import webscraping
from model import Model
import Graph_Code

# Load the model on the 3 GoEmotions training sets.
df = pd.concat(
    map(pd.read_csv, ['data/goemotions_1.csv'""",
                      'data/goemotions_2.csv',
                      'data/goemotions_3.csv'"""]), ignore_index=True)
model = Model(df)

# Start the server
app = Flask(__name__)
CORS(app)

@app.route("/api/emotions/post/list", methods=["POST"])
def get_Emotions_For_A_List_Of_Songs():
    data = request.get_json()

    classified = webscraping.read_top_songs(data)

    # Get all the musixmatch song information
    webscraping.scrape_song(classified[0])
    #for song in classified:
        #webscraping.scrape_song(song)
        #print(classified[i].lyrics)

    # Run Each song through the model
    model.eval(classified[0].lyrics)

    # Construct a DataFrame for emotions from the songs
    emotions = [song.emotions for song in classified]
    df = pd.DataFrame({
        'emotions': emotions
    }) 

    # Send the Dataframe to the Graph_Code function -> Base64 encoded image
    b64encoded_string = Graph_Code.construct_Song_Emotions_Graph(emotions)

    return {'base64_encoded_gimage': b64encoded_string}, 200#{'response': map(lambda song: song.toJson(), classified) }), 200

def construct_Data_Frame_from_Song(song):
    df = pd.DataFrame({
        "text": song.lyics,
        "admiration": [0 for x in range(len(song.lyrics))],
        "amusement": [0 for x in range(len(song.lyrics))],
        "anger": [0 for x in range(len(song.lyrics))],
        "annoyance": [0 for x in range(len(song.lyrics))],
        "approval": [0 for x in range(len(song.lyrics))],
        "caring": [0 for x in range(len(song.lyrics))],
        "confusion": [0 for x in range(len(song.lyrics))],
        "curiosity": [0 for x in range(len(song.lyrics))],
        "desire": [0 for x in range(len(song.lyrics))],
        "disappointment": [0 for x in range(len(song.lyrics))],
        "disapproval": [0 for x in range(len(song.lyrics))],
        "disgust": [0 for x in range(len(song.lyrics))],
        "embarrassment": [0 for x in range(len(song.lyrics))],
        "excitement": [0 for x in range(len(song.lyrics))],
        "fear": [0 for x in range(len(song.lyrics))],
        "gratitude": [0 for x in range(len(song.lyrics))],
        "grief": [0 for x in range(len(song.lyrics))],
        "joy": [0 for x in range(len(song.lyrics))],
        "love": [0 for x in range(len(song.lyrics))],
        "nervousness": [0 for x in range(len(song.lyrics))],
        "optimism": [0 for x in range(len(song.lyrics))],
        "pride": [0 for x in range(len(song.lyrics))],
        "realization": [0 for x in range(len(song.lyrics))],
        "relief": [0 for x in range(len(song.lyrics))],
        "remorse": [0 for x in range(len(song.lyrics))],
        "sadness": [0 for x in range(len(song.lyrics))],
        "surprise": [0 for x in range(len(song.lyrics))],
        "neutral": [0 for x in range(len(song.lyrics))],
    })
    return df # Return the dataframe of song-lyrics.

if __name__ == "__main__":
   print("Spotify Emotions Server Started \nPress Ctrl+C to stop the server \nServing..")
   serve(app, host="10.0.0.231", port=8080)
