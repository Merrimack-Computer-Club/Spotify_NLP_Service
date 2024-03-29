from imports import *
import webscraping
from model import Model
from model import BertClassifier
from model import loadModel
import Graph_Code

# Load in the config file.
config = toml.load('config.toml')

# Load the model on the 3 GoEmotions training sets.
df = pd.concat(
    map(pd.read_csv, [
                      #'data/testemotions_3.csv'
                      #'data/testemotions_2.csv'
                      #'data/testemotions_1.csv'
                      #'data/goemotions_1.csv'#,
                      #'data/goemotions_2.csv',
                      'data/goemotions_3.csv'
                      ]),
                      ignore_index=True)

model = None
if(config['server']['load_model']):
    model = loadModel(config['server']['model_path'])
else:
    model = Model(df, train=True)

print("Model Initialized")

# Start the server
app = Flask(__name__)
CORS(app)

@app.route("/api/emotions/post/list", methods=["POST"])
def get_Emotions_For_A_List_Of_Songs():
    data = request.get_json()

    classified = webscraping.read_top_songs(data)

    # Remove songs that are not none
    classified = [song for song in classified if song is not None]

    # Get all the musixmatch song information
    for song in classified:
        webscraping.scrape_song(song)


    # Run Each song through the model
        # Build dataframe from the song lyrics to pass to BERT model
    songs = [x.lyrics for x in classified if x.lyrics != None and len(x.lyrics) > 0]
    print(songs)
    songs_df = construct_Data_Frame_from_Song(songs)
    #print(songs_df)
    
    # Evaluate the song. Returns probabilities of the emotions for each sentence
    probs = model.eval(songs_df)
    #print(probs)
    
    def emotions_occurences(probs):
        labels = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", 
            "confusion", "curiosity", "desire", "disappointment", "disapproval", 
            "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
            "joy", "love", "nervousness", "optimism", "pride", "realization", 
            "relief", "remorse", "sadness", "surprise", "neutral"]
        occurrences = []
        for arr in probs:
            emos = sorted(list(zip(labels, arr)), key = lambda x: x[1], reverse=True)
            occurrences.append(emos[0][0])
        return occurrences


    #emotional_value = avg_emotions(probs)
    emotional_value = emotions_occurences(probs)
    print(len(emotional_value))
        # Check if there were no valid lyrics that tokenized
    if(len(emotional_value) == 0):
        print("Errored. No lyrics to tokenize.")
        return {'error': 'No lyrics tokenizable by the model. Please listen to more music with lyrics!!!'}

    # Construct a DataFrame for emotions from the songs
    df = pd.DataFrame(emotional_value, columns=['emotion'])

    print("Response sent to client.")
    return {'probabilities': df.to_dict(orient="records")}, 200#{'response': map(lambda song: song.toJson(), classified) }), 200

'''Construct a data frame of songs'''
def construct_Data_Frame_from_Song(song):
    cols = ["text", "admiration", "amusement", "anger", "annoyance", "approval", "caring",
            "confusion", "curiosity", "desire", "disappointment", "disapproval",
            "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
            "joy", "love", "nervousness", "optimism", "pride", "realization",
            "relief", "remorse", "sadness", "surprise", "neutral"]

    df = pd.DataFrame(columns=cols)

    emotions = [x for x in cols if x != "text"]

    df["text"] = song

    df[emotions] = 0

    return df

if __name__ == "__main__":
   print("Spotify Emotions Server Started \nPress Ctrl+C to stop the server \nServing..")
   serve(app, host="127.0.0.1", port=8080)