from imports import *
from bert_utils import create_tokenizer_from_hub_module, tokenize_sentences
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split

# Class built for our BERT Classifier.
# Source: https://skimai.com/fine-tuning-bert-for-sentiment-analysis/
class BertClassifier(nn.Module):

    # Constructor, creates the classifier
    def __init__(self, freeze_bert = False):
        """
        @param  bert_model: BertModel object
        @param  freeze_bert (bool): Set `False` to fine-tune the BERT model (according to the resource)
        @param  classifier: torch.nn.Module classifier
            #    Note: torch.nn.Module is used to help train / build        #
            #       neural networks, so we will build on BERT with this     #
        """
        super(BertClassifier, self).__init__()
        # Specify output size of BERT model (768), hidden size of our classifier (50), and number of labels (28)
        D_in, H, D_out = 768, 50, 28


        # Instantiate the pre-trained BERT tokenizer and model
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

        # Instantiate one-layer feed-forward classifer
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H), # Applies linear transformation
            nn.ReLu(),
            nn.Linear(H, D_out)
        )
        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert_model.parameters():
                param.requires_grad = False

    # Define a feed-forward function to compute the logits
    # Logits = Output of logistic regression function (done through NN) (between 0 and 1)
    def forward(self, ids, mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask information
        @return   logits (torch.Tensor): an output tensor
        """
        # Feed input to BERT
        outputs = self.bert_model(input_ids=ids, attention_mask=mask)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits
    
class Model():
    # Constructor, creates the classifier
    def __init__(self, bert_classifier):
        """
        @param  bert_classifier: BertClassifier object
        @param  tokenizer: BertTokenizer object
        """
        
        # Instantiate the classifier
        self.bert_classifier = bert_classifier
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Initializing the model, optimizer, and learning rate scheduler for training
    # Source: https://skimai.com/fine-tuning-bert-for-sentiment-analysis/
    def initialize_model(epochs=3):
        # Initialize the classifier model, the optimizer, and the learning rate scheduler
        classifier = BertClassifier(freeze_bert=False) #Instantiate the model
        # Try to use GPU (cuda). Otherwise, we will have to use CPU
        device = None
        if torch.cuda.is_available():       
            device = torch.device("cuda")
            print(f'There are {torch.cuda.device_count()} GPU(s) available.')
            print('Device name:', torch.cuda.get_device_name(0))
        else:
            print('No GPU available, using the CPU instead.')
            device = torch.device("cpu")

        # Assign the model to hardware
        classifier.to(device)

        # Create an AdamW optimizer
        optimizer = AdamW(classifier.parameters(),
                        lr=5e-5,   # Best learning rate (lr) described. Also default
                        eps=1e-8)  # Default epsilon value
        
        # Total number of training steps for the lr scheduler
        total_steps = len(train_data_loader) * epochs

        # Set up learning rate scheduler for our model
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0, #Default
                                                    num_training_steps=total_steps)
        
        return classifier, optimizer, scheduler

    
    # Define a function to tokenize the input and prepare it for fine-tuning
    def tokenize_inputs(self, data):
        input_ids = []          # Holds tensor information for the word in context
        attention_masks = []    # Holds binary data on what's important (1 for tensor data, 0 for filler)
        labels = []             # Holds binary data on emotions (1 for has emotion, 0 for doesn't)
        
        # Iterate through each sentence and emotion labels associated with that sentence for each row.
        for sentence, emotions in zip(data['text'], data[emotion_labels].values):
            # Tokenize the input sentence
            # Tokenizer understanding credit: https://towardsdatascience.com/how-to-train-a-bert-model-from-scratch-72cfce554fc6
            encoded_dict = self.tokenizer.encode_plus(
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

def to_data_loader(input_ids, attention_masks, labels):
    # Place data in a 
        # Place data in a PyTorch DataLoader (faster training / less resources)
    data = TensorDataset(input_ids, attention_masks, labels)
    data_sampler = RandomSampler(data)
    data_loader = DataLoader(data, sampler=data_sampler, batch_size=32)
    return data_loader


###                 Main Code                   ###


# Load the CSV data (expected to run from folder: "./Artificial Intelligence/Spotify_NLP_Service")
df = pd.read_csv('./server/data/testemotions_1.csv')

df_train = df.sample(frac=0.8, random_state=200)
df_test = df.drop(df_train.index)

# Define the emotion labels
emotion_labels = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", 
                  "confusion", "curiosity", "desire", "disappointment", "disapproval", 
                  "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
                  "joy", "love", "nervousness", "optimism", "pride", "realization", 
                  "relief", "remorse", "sadness", "surprise", "neutral"]

# Step 1: Tokenize inputs so BERT can read it. 
    # Tokenize the inputs
train_input_ids, train_attention_masks, train_labels = model.tokenize_inputs(df_train)
test_input_ids, test_attention_masks, test_labels = model.tokenize_inputs(df_test)


# Step 2: Fine-tune BERT.
# Resource(s): https://www.youtube.com/watch?v=mw7ay38--ak
train_data_loader = to_data_loader(train_input_ids, train_attention_masks, train_labels)


###                 End Main Code                   ###


"""
# Print the shape of the tokenized inputs and labels
print("Input IDs shape: ", input_ids.shape)
print("Attention Masks shape: ", attention_masks.shape)
print("Labels shape: ", labels.shape)

print("Input IDs: ", input_ids[input_ids.shape[0] - 1])
print("Attention Masks: ", attention_masks[input_ids.shape[0] - 1])
print("Labels: ", labels[input_ids.shape[0] - 1])

print(df['text'][input_ids.shape[0] - 1])
print(emotion_labels)



This class represents a Model.
To run make a new model class then call construct_model to make a new model.

"""



#"""