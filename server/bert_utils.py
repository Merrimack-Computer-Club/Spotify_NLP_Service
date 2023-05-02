from imports import * 

def create_tokenizer_from_hub_module(bert_layer):
    """Creates a tokenizer from a BERT module."""
    tokenizer = hub.KerasLayer(bert_layer.resolved_object.vocab_file.asset_path.numpy())
    return tokenizer

def tokenize_sentences(tokenizer, sentences, max_seq_length):
    """Tokenizes a list of sentences and converts them into input features for the BERT model."""
    input_ids_all, input_mask_all, segment_ids_all = [], [], []
    for sentence in sentences:
        tokens = tokenizer.tokenize(sentence)
        tokens = tokens[:max_seq_length-2]
        input_tokens = ['[CLS]'] + tokens + ['[SEP]']
        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)
        padding_length = max_seq_length - len(input_ids)
        input_ids += [0] * padding_length
        input_mask += [0] * padding_length
        segment_ids += [0] * padding_length
        input_ids_all.append(input_ids)
        input_mask_all.append(input_mask)
        segment_ids_all.append(segment_ids)
    return {
        'input_word_ids': np.array(input_ids_all),
        'input_mask': np.array(input_mask_all),
        'segment_ids': np.array(segment_ids_all)
    }