from imports import *
import webscraping
from model import Model
import Graph_Code

# Load the model on the 3 GoEmotions training sets.
df = pd.concat(
    map(pd.read_csv, [
                      'data/testemotions_2.csv'
                      #'data/testemotions_1.csv'
                      #'data/goemotions_1.csv'#,
                      #'data/goemotions_2.csv',
                      #'data/goemotions_3.csv'
                      ]),
                      ignore_index=True)
model = Model(df)

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
    songs_df = pd.concat(map(construct_Data_Frame_from_Song, classified), ignore_index=True)
    print(songs_df.head())
        # Evaluate the song. Returns probabilities of the emotions for each sentence
    probs = model.eval(songs_df)

    """    # Average the probabilities across all sentences and turn it into one list
    def avg_emotions(probs):
        labels = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", 
            "confusion", "curiosity", "desire", "disappointment", "disapproval", 
            "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
            "joy", "love", "nervousness", "optimism", "pride", "realization", 
            "relief", "remorse", "sadness", "surprise", "neutral"]
        
        avg_probs = []
        # Run through each of the 28 emotions...
        for i in range(len(probs[0])):
            total_val = 0
            for val in probs: # For all sentences
                total_val = total_val + val[i]
            avg_probs.append(total_val / len(avg_probs)) # Get the average, and append it (keeps order)

        return list(zip(labels, avg_probs))"""
    
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

    # Send the Dataframe to the Graph_Code function -> Base64 encoded image
    b64encoded_string = Graph_Code.construct_Song_Emotions_Graph(df)

    print("Sent graph over to user.")

    return {'base64_encoded_gimage': b64encoded_string}, 200#{'response': map(lambda song: song.toJson(), classified) }), 200

def construct_Data_Frame_from_Song(song):
    cols = ["text", "admiration", "amusement", "anger", "annoyance", "approval", "caring", 
            "confusion", "curiosity", "desire", "disappointment", "disapproval", 
            "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
            "joy", "love", "nervousness", "optimism", "pride", "realization", 
            "relief", "remorse", "sadness", "surprise", "neutral"]

    df = pd.DataFrame(columns=cols)

    emotions = [x for x in cols if x != "text"]

    df["text"] = song.lyrics

    df[emotions] = 0

    return df # Return the dataframe of song-lyrics.

if __name__ == "__main__":
   print("Spotify Emotions Server Started \nPress Ctrl+C to stop the server \nServing..")
   serve(app, host="127.0.0.1", port=8080)

