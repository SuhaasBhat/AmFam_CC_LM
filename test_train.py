from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer, LineByLineTextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers import RobertaTokenizerFast, RobertaForMaskedLM, RobertaConfig
import process_txt_data as ptd
#import torch

tokenizer = GPT2Tokenizer.from_pretrained("gpt2", pad_token = '<pad>')
bertTokenizer = RobertaTokenizerFast.from_pretrained("distilbert-base-uncased")

path = "amfamdata1.txt"

dataset = ptd.AmfamDataset(tokenizer=tokenizer, file_path = path, block_size = 128)

data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = False)
bert_data_collator = DataCollatorForLanguageModeling(tokenizer = bertTokenizer, mlm = True, mlm_probability = 0.15)

#config = GPT2Config()
model = GPT2LMHeadModel.from_pretrained('gpt2')
#device = torch.device('cuda')

#bertConfig = RobertaConfig(
#    vocab_size=52_000,
#    max_position_embeddings=514,
#    num_attention_heads=12,
#    num_hidden_layers=6,
#    type_vocab_size=1,
#)

##mode ;

bertModel = RobertaForMaskedLM.from_pretrained('distilbert-base-uncased')
#bertModel = RobertaForMaskedLM(config = bertConfig)



training_args = TrainingArguments(
    output_dir = "./TrainerAmFamBERTv2",
    overwrite_output_dir = True,
    num_train_epochs = 1,
    per_device_train_batch_size = 32,
    save_steps = 1_000,
    save_total_limit = 10)

trainer = Trainer(
    model = bertModel,
    args = training_args,
    data_collator = bert_data_collator,
    train_dataset = dataset,
    prediction_loss_only = True
)

trainer.train()

trainer.save_model("./TrainerAmFamBERTv2")

##I really want to use pytorch for this stuff, the Trainer does not make me happy
