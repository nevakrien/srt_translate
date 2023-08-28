import argparse
import srt
from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast
import torch
from collections import deque

from tqdm import tqdm

model_name = 'facebook/nllb-200-3.3B'
SEP_TOKEN=2

def get_model_args(src_lang,tgt_lang):
	model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
	tokenizer = NllbTokenizerFast.from_pretrained(model_name, src_lang=src_lang, tgt_lang=tgt_lang)
	src_tok=tokenizer.lang_code_to_id[src_lang]
	tgt_tok=tokenizer.lang_code_to_id[tgt_lang]
	return{'model':model,'tokenizer':tokenizer,'src_tok':src_tok,'tgt_tok':tgt_tok}

def translate_text(src_text:str,tgt_text:str,model,tokenizer, src_tok, tgt_tok,max_length=1000,level=1):
	# Encode the source text, translate it, and then decode the translation
	inputs = tokenizer(src_text, return_tensors="pt")
	tgt_tokens=tokenizer.encode(tgt_text,add_special_tokens=False)
	#may need to add sep here
	tgt_tokens=torch.LongTensor([[SEP_TOKEN,tgt_tok]+tgt_tokens])

	translation = model.generate(**inputs, decoder_input_ids=tgt_tokens,
		max_length=max_length,top_p=0.75,temperature=1.4,num_beams=level,penalty_alpha=0.4
		,length_penalty=0.7,min_new_tokens=3)#,repetition_penalty=1.5,no_repeat_ngram_size=5)
	
	translated_text = tokenizer.decode(translation[0][tgt_tokens.shape[1]:], skip_special_tokens=True)
	return translated_text

def main():
	# Define the command-line arguments
	parser = argparse.ArgumentParser(description="Translate an SRT file.")
	parser.add_argument("--input_srt",type=str, required=True, help="Path to the input SRT file.")
	parser.add_argument("--output_srt",type=str, required=True, help="Path to the output SRT file.")
	#eng_Latn
	parser.add_argument("--src_lang", type=str,default="eng_Latn", help="Source language code.")
	parser.add_argument("--tgt_lang", type=str,required=True, help="Target language code.")
	
	parser.add_argument("--history", type=int,default=1, help="number of sentances in the context window")
	parser.add_argument("--level", type=int,default=1, help="depth level of the beam search")
	parser.add_argument("--max_length", type=int,default=1000, help="max token length of a line")

	parser.add_argument("--print", type=int,default=0, help="if passed as 1 will print to the command line as well")
	

	args = parser.parse_args()
	assert(args.history>=0)

	print('loading file')
	with open(args.input_srt, "r", encoding="utf-8") as f:
		subtitles = list(srt.parse(f))

	print('loading model and tokenizer')

	model_args=get_model_args(args.src_lang,args.tgt_lang) 

	print('translating')
	
	q=deque(maxlen=args.history+1)
	outputed=deque(maxlen=args.history)

	for subtitle in tqdm(subtitles):
		q.append(subtitle.content)
		src_text='\n'.join(q)
		tgt_text=''.join(outputed)
		subtitle.content = translate_text(src_text,tgt_text
			,max_length=args.max_length,level=args.level,
			**model_args)

		outputed.append(subtitle.content)
		if(args.print):
			print(subtitle.content+"\n\n")

	print('writing output to file')
	with open(args.output_srt, "w", encoding="utf-8") as f:
		f.write(srt.compose(subtitles))

	print('done')

if __name__ == "__main__":
	main()
