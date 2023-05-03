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

if __name__ == "__main__":
   print("Spotify Emotions Server Started \nPress Ctrl+C to stop the server \nServing..")
   serve(app, host="10.0.0.231", port=8080)
