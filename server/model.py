import random
import time
from imports import *
from bert_utils import create_tokenizer_from_hub_module, tokenize_sentences
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split

# Class built for our BERT Classifier.
# Source: https://skimai.com/fine-tuning-bert-for-sentiment-analysis/
class BertClassifier(nn.Module):

    # Constructor, creates the classifier
    def __init__(self, freeze_bert = True):
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
            nn.ReLU(),
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
    
class Model:
    # Constructor, creates the classifier
    def __init__(self, file_path):
        """
        @param  bert_classifier: BertClassifier object
        @param  tokenizer: BertTokenizer object
        """
    # TRAINING DATA
        # Load the CSV data (expected to run from folder: "./Artificial Intelligence/Spotify_NLP_Service")
        df = pd.read_csv(file_path)

        df_train = df.sample(frac=0.8, random_state=200)
        df_test = df.drop(df_train.index)

        # Define the emotion labels
        self.emotion_labels = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", 
                        "confusion", "curiosity", "desire", "disappointment", "disapproval", 
                        "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
                        "joy", "love", "nervousness", "optimism", "pride", "realization", 
                        "relief", "remorse", "sadness", "surprise", "neutral"]

        # Step 1: Tokenize inputs so BERT can read it.
            # Initialize the tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            # Tokenize the inputs
        train_input_ids, train_attention_masks, train_labels = self.tokenize_inputs(df_train)
        test_input_ids, test_attention_masks, test_labels = self.tokenize_inputs(df_test)


        # Step 2: Load the data into a DataLoader object
            # Resource(s): https://www.youtube.com/watch?v=mw7ay38--ak
        train_data_loader = self.to_data_loader(train_input_ids, train_attention_masks, train_labels)
        test_data_loader = self.to_data_loader(test_input_ids, test_attention_masks, test_labels)

        # Initialize the model for training purposes
        classifier, optimizer, scheduler = self.initialize_model(train_data_loader)
        
        # Instantiate the classifier, optimizer, scheduler, and tokenizer
        self.bert_classifier = classifier
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train(train_data_loader=train_data_loader)
        self.probs = self.eval(df_test)
        print("Init Eval:")
        print(self.probs)

    # Initializing the model, optimizer, and learning rate scheduler for training
        # Source: https://skimai.com/fine-tuning-bert-for-sentiment-analysis/
    def initialize_model(self, train_data_loader, epochs=2):
        # Initialize the classifier model, the optimizer, and the learning rate scheduler
        classifier = BertClassifier(freeze_bert=False) #Instantiate the model
        # Try to use GPU (cuda). Otherwise, we will have to use CPU
        self.device = None
        if torch.cuda.is_available():       
            self.device = torch.device("cuda")
            print(f'There are {torch.cuda.device_count()} GPU(s) available.')
            print('Device name:', torch.cuda.get_device_name(0))
        else:
            print('No GPU available, using the CPU instead.')
            self.device = torch.device("cpu")

        # Assign the model to hardware
        classifier.to(self.device)

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
        for sentence, emotions in zip(data['text'], data[self.emotion_labels].values):
            # Tokenize the input sentence
            # Tokenizer understanding credit: https://towardsdatascience.com/how-to-train-a-bert-model-from-scratch-72cfce554fc6
            encoded_dict = self.tokenizer.encode_plus(
                                sentence,                      
                                add_special_tokens = True, 
                                max_length = 128,
                                padding = 'max_length',
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
    
    def to_data_loader(self, input_ids, attention_masks, labels):
        # Place data in a 
            # Place data in a PyTorch DataLoader (faster training / less resources)
        data = TensorDataset(input_ids, attention_masks, labels)
        data_sampler = RandomSampler(data)
        data_loader = DataLoader(data, sampler=data_sampler, batch_size=32)
        return data_loader
    
    def train(self, train_data_loader, val_dataloader=None, epochs=2, evaluation=False):
        loss_fn = nn.CrossEntropyLoss()

        """Train the BertClassifier model."""
        # Start training loop
        print("Start training...\n")
        for epoch_i in range(epochs):
            # =======================================
            #               Training
            # =======================================
            # Print the header of the result table
            print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
            print("-"*70)

            # Measure the elapsed time of each epoch
            t0_epoch, t0_batch = time.time(), time.time()

            # Reset tracking variables at the beginning of each epoch
            total_loss, batch_loss, batch_counts = 0, 0, 0

            # Put the model into the training mode
            self.bert_classifier.train()

            # For each batch of training data...
            for step, batch in enumerate(train_data_loader):
                batch_counts +=1
                # Load batch to GPU
                b_input_ids, b_attn_mask, b_labels = tuple(t.to(self.device) for t in batch)

                # Zero out any previously calculated gradients
                self.bert_classifier.zero_grad()

                # Perform a forward pass. This will return logits.
                logits = self.bert_classifier(b_input_ids, b_attn_mask)

                # Compute loss and accumulate the loss values
                loss = loss_fn(logits, b_labels.float())
                batch_loss += loss.item()
                total_loss += loss.item()

                # Perform a backward pass to calculate gradients
                loss.backward()

                # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
                torch.nn.utils.clip_grad_norm_(self.bert_classifier.parameters(), 1.0)

                # Update parameters and the learning rate
                self.optimizer.step()
                self.scheduler.step()

                # Print the loss values and time elapsed for every 20 batches
                if (step % 20 == 0 and step != 0) or (step == len(train_data_loader) - 1):
                    # Calculate time elapsed for 20 batches
                    time_elapsed = time.time() - t0_batch

                    # Print training results
                    print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                    # Reset batch tracking variables
                    batch_loss, batch_counts = 0, 0
                    t0_batch = time.time()

            # Calculate the average loss over the entire training data
            avg_train_loss = total_loss / len(train_data_loader)

            print("-"*70)
            """# =======================================
            #               Evaluation
            # =======================================
            if evaluation == True:
                # After the completion of each training epoch, measure the model's performance
                # on our validation set.
                val_loss, val_accuracy = evaluate(model, val_dataloader)

                # Print performance over the entire training data
                time_elapsed = time.time() - t0_epoch
                
                print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
                print("-"*70)
            print("\n")"""
        
        print("Training complete!")
    
    def eval(self, df):
        """Perform a forward pass on the trained BERT model to predict probabilities
        on the test set.
        """
        # Tokenize inputs from the df
        input_ids, attention_masks, labels = self.tokenize_inputs(df)

        # Convert to a data loader
        dataloader = self.to_data_loader(input_ids, attention_masks, labels)

        # Put the model into the evaluation mode. The dropout layers are disabled during
        # the test time.
        self.bert_classifier.eval()

        all_logits = []

        # For each batch in our test set...
        for batch in dataloader:
            # Load batch to GPU
            b_input_ids, b_attn_mask = tuple(t.to(self.device) for t in batch)[:2]

            # Compute logits
            with torch.no_grad():
                logits = self.bert_classifier(b_input_ids, b_attn_mask)
            all_logits.append(logits)
        
        # Concatenate logits from each batch
        all_logits = torch.cat(all_logits, dim=0)

        # Apply softmax to calculate probabilities
        probs = F.softmax(all_logits, dim=1).cpu().numpy()

        return probs




###                 Main Code                   ###

# Source: https://skimai.com/fine-tuning-bert-for-sentiment-analysis/
def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

set_seed() #Sets predetermined 'randomization' for repeated trials

# Build the model 
model = Model('D:/Documents/College Folder/Artificial Intelligence/Spotify_NLP_Service/server/data/testemotions_2.csv')

# Get test data and evaluate the model.
myDF = pd.read_csv('D:/Documents/College Folder/Artificial Intelligence/Spotify_NLP_Service/server/data/runtest.csv')

probs = model.eval(myDF)

print("Sentence Eval:")
print(probs)


###                 End Main Code                   ###
