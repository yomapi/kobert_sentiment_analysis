import torch
from transformers import BertModel
import gluonnlp as nlp
from kobert_tokenizer import KoBERTTokenizer


tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")
bertmodel = BertModel.from_pretrained("skt/kobert-base-v1", return_dict=False)
vocab = nlp.vocab.BERTVocab.from_sentencepiece(
    tokenizer.vocab_file, padding_token="[PAD]"
)
tok = tokenizer.tokenize

# Setting parameters
max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
log_interval = 200
learning_rate = 5e-5
# device = torch.device("cuda:0")
device = torch.device("cpu")
