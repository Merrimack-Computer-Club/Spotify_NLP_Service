from imports import *
from bert_utils import create_tokenizer_from_hub_module, tokenize_sentences
from transformers import BertTokenizer, BertModel

# Load the CSV data
df = pd.read_csv('D:/Documents/College Folder/Artificial Intelligence/Spotify_NLP_Service/server/data/testemotions_1.csv')

# Define the emotion labels
emotion_labels = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"]

# Initialize the pre-trained BERT tokenizer and model
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define a function to tokenize the input and prepare it for fine-tuning
def tokenize_inputs(data):
    input_ids = []          # Holds tensor information for the word in context
    attention_masks = []    # Holds binary data on what's important (1 for tensor data, 0 for filler)
    labels = []             # Holds binary data on emotions (1 for has emotion, 0 for doesn't)
    
    # Iterate through each sentence and emotion labels associated with that sentence for each row.
    for sentence, emotions in zip(data['text'], data[emotion_labels].values):
        # Tokenize the input sentence
        # Tokenizer understanding credit: https://towardsdatascience.com/how-to-train-a-bert-model-from-scratch-72cfce554fc6
        encoded_dict = tokenizer.encode_plus(
                            sentence,                      
                            add_special_tokens = True, 
                            max_length = 128,
                            pad_to_max_length = True,
                            truncation=True,
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        
        # Add the encoded sentence to the list
        input_ids.append(encoded_dict['input_ids'])
        
        # Add the attention mask for the encoded sentence to the list
        attention_masks.append(encoded_dict['attention_mask'])
        
        # Add the labels to the list
        labels.append(emotions)
    
    # Convert the lists to tensors
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    
    # Return the tokenized inputs and labels
    return input_ids, attention_masks, labels

# Tokenize the inputs
input_ids, attention_masks, labels = tokenize_inputs(df)

# Print the shape of the tokenized inputs and labels
print("Input IDs shape: ", input_ids.shape)
print("Attention Masks shape: ", attention_masks.shape)
print("Labels shape: ", labels.shape)

print("Input IDs: ", input_ids[input_ids.shape[0] - 1])
print("Attention Masks: ", attention_masks[input_ids.shape[0] - 1])
print("Labels: ", labels[input_ids.shape[0] - 1])

print(df['text'][input_ids.shape[0] - 1])
print(emotion_labels)


"""
This class represents a Model.
To run make a new model class then call construct_model to make a new model.



class Model:
    model = None

    # Constructor, creates the model
    def __init__(self):
        self.model = None

    # Returns a new BERT model using the emotion_set
    def construct_model(self):
        train_sentences = [
            "I am feeling happy today",
            "The news made me sad",
            "I am tired of working all day",
            "The customer service was so bad that it made me angry",
            "I am feeling violent and I want to hit something",
            "My friend was so helpful to me when I was in need"
        ]

        train_labels = [
            "happy",
            "sad",
            "tired",
            "angry",
            "violent",
            "helpful"
        ]

        val_sentences = [
            "I am feeling so happy right now",
            "The movie was really sad",
            "I am exhausted from the long day",
            "The rude behavior of the driver made me angry",
            "I feel like I could explode with rage",
            "My teacher was very helpful and patient with me"
        ]

        val_labels = [
            "happy",
            "sad",
            "tired",
            "angry",
            "violent",
            "helpful"
        ]
        # Import the training & validation csv
        #df = pd.read_csv('data/goemotions_1.csv')

        # Definine example training & testing data

        # Define the input and output data
        #train_sentences = [data[0] for data in train_data]
        #train_labels = [data[1] for data in train_data]

        #val_sentences = [data[0] for data in val_data]
        #val_labels = [data[1] for data in val_data]

        # Load the BERT module
        module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3"
        bert_layer = hub.KerasLayer(module_url, trainable=True)

        max_seq_length = 128
        # Define the model architecture
        input_word_ids = tf.keras.layers.Input(shape=(max_seq_length), dtype=tf.int32, name="input_word_ids")
        input_mask = tf.keras.layers.Input(shape=(max_seq_length), dtype=tf.int32, name="input_mask")
        segment_ids = tf.keras.layers.Input(shape=(max_seq_length), dtype=tf.int32, name="segment_ids")

        pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

        drop = tf.keras.layers.Dropout(0.4)(pooled_output)

        output = tf.keras.layers.Dense(6, activation='softmax')(drop)

        model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=output)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5), loss='categorical_crossentropy', metrics=['accuracy'])

        # Tokenize the input sentences
        tokenizer = create_tokenizer_from_hub_module(bert_layer)
        train_input_ids = tokenize_sentences(tokenizer, train_sentences, max_seq_length)
        val_input_ids = tokenize_sentences(tokenizer, val_sentences, max_seq_length)

        # Convert the labels to one-hot encoding
        train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=6)
        val_labels = tf.keras.utils.to_categorical(val_labels, num_classes=6)

        # Train the model
        history = model.fit(
            x=[train_input_ids['input_word_ids'], train_input_ids['input_mask'], train_input_ids['segment_ids']],
            y=train_labels,
            validation_data=([val_input_ids['input_word_ids'], val_input_ids['input_mask'], val_input_ids['segment_ids']], val_labels),
            batch_size=32,
            epochs=3
        )

        # Use the model to predict the emotions of some example sentences
        test_sentences = ['I feel happy today', 'I am tired of this work', 'I am sad to hear this news', 'The violent movie made me scared']

        test_input_ids = tokenize_sentences(tokenizer, test_sentences, max_seq_length)
        test_predictions = model.predict([test_input_ids['input_word_ids'], test_input_ids['input_mask'], test_input_ids['segment_ids']])

        test_predictions_labels = np.argmax(test_predictions, axis=1)
        emotions = ['happy', 'sad', 'tired', 'angry', 'violent', 'helpful']
        test_emotions = [emotions[prediction] for prediction in test_predictions_labels]

        print(test_emotions)

        # Return the model
        return None
    
model = Model()
model.construct_model()

"""