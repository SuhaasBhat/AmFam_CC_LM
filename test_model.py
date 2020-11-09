from transformers import pipeline, RobertaTokenizerFast
#import test_train as tt

bertTokenizer = RobertaTokenizerFast.from_pretrained("distilbert-base-uncased")
fill_mask = pipeline("fill-mask", model = "./TrainerAmFamBERT/checkpoint-10000", tokenizer = bertTokenizer)

print(fill_mask("My favorite insurance policy is <mask>"))

#config = GPT2Config.from_pretrained("./TrainerAmFamBERT", return_dict = True)
#config.is_decoder = True

#model = GPT2Config.from_pretrained("./TrainerAmFamBERT", config = config)

#text_gen = pipeline("text-generation", model = model, tokenizer = bertTokenizer)
#print(text_gen("American Family"))
