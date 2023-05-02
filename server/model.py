from imports import *
from bert_utils import create_tokenizer_from_hub_module, tokenize_sentences
from transformers import BertTokenizer, BertModel

# Class built for our BERT Classifier (called Model).
# Resource(s): https://skimai.com/fine-tuning-bert-for-sentiment-analysis/
class Model(nn.Module):

    # Constructor, creates the model
    def __init__(self, freeze_bert = False):
        """
        @param  model: BertModel object
        @param  tokenizer: BertTokenizer object
        @param  freeze_bert (bool): Set `False` to fine-tune the BERT model (according to the resource)
        @param  classifier: torch.nn.Module classifier
            #    Note: torch.nn.Module is used to help train / build        #
            #       neural networks, so we will build on BERT with this     #
        """
        super(Model, self).__init__()
        # Specify output size of BERT model (768), hidden size of our classifier (50), and number of labels (28)
        D_in, H, D_out = 768, 50, 28


        # Instantiate the pre-trained BERT tokenizer and model
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Instantiate one-layer feed-forward classifer
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H), # Applies linear transformation
            nn.ReLu(),
            nn.Linear(H, D_out)
        )
        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
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
        outputs = self.bert(input_ids=ids, attention_mask=mask)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits
    
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


    
model = Model()

def to_data_loader(input_ids, attention_masks, labels):
    # Place data in a 
        # Place data in a PyTorch DataLoader (faster training / less resources)
    data = TensorDataset(input_ids, attention_masks, labels)
    data_sampler = RandomSampler(data)
    data_loader = DataLoader(data, sampler=data_sampler, batch_size=32)
    return data_loader

def fine_tune(model, data_loader):
    # Freeze the BERT layers to start (that way we can start fine-tuning)
    for param in model.parameters():
        param.requires_grad = False


# Load the CSV data (expected to run from folder: "./Artificial Intelligence/Spotify_NLP_Service")
df = pd.read_csv('./server/data/testemotions_1.csv')

# Define the emotion labels
emotion_labels = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", 
                  "confusion", "curiosity", "desire", "disappointment", "disapproval", 
                  "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
                  "joy", "love", "nervousness", "optimism", "pride", "realization", 
                  "relief", "remorse", "sadness", "surprise", "neutral"]

# Step 1: Tokenize inputs so BERT can read it. 
    # Tokenize the inputs
input_ids, attention_masks, labels = model.tokenize_inputs(df)

# Step 2: Fine-tune BERT.
# Resource(s): https://www.youtube.com/watch?v=mw7ay38--ak
data_loader = to_data_loader(input_ids, attention_masks, labels)
fine_tune(model, data_loader)


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