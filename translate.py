import argparse
import srt
from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast

model_name = 'facebook/nllb-200-3.3B'
model = None
tokenizer = None

def init_models(src_lang=None,tgt_lang=None):
	global model, tokenizer
	# Load the translation model and tokenizer
	model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
	if src_lang==None:
		tokenizer = NllbTokenizerFast.from_pretrained(model_name)
	else: 
		tokenizer = NllbTokenizerFast.from_pretrained(model_name, src_lang=src_lang, tgt_lang=tgt_lang)

def translate_text(text, src_lang, tgt_lang,max_length=1000,level=1):
	# Encode the source text, translate it, and then decode the translation
	inputs = tokenizer.encode(text, return_tensors="pt")
	translation = model.generate(inputs, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
		max_length=max_length,top_k=level,num_beams=level,penalty_alpha=0.4)
	translated_text = tokenizer.decode(translation[0], skip_special_tokens=True)
	return translated_text

def main():
	# Define the command-line arguments
	parser = argparse.ArgumentParser(description="Translate an SRT file.")
	parser.add_argument("--input_srt",type=str, required=True, help="Path to the input SRT file.")
	parser.add_argument("--output_srt",type=str, required=True, help="Path to the output SRT file.")
	#eng_Latn
	parser.add_argument("--src_lang", type=str,default=None, help="Source language code.")
	parser.add_argument("--tgt_lang", type=str,required=True, help="Target language code.")
	
	parser.add_argument("--level", type=int,default=1, help="increase this to get better results")
	parser.add_argument("--max_length", type=int,default=1000, help="max token length of a line")
	

	args = parser.parse_args()

	print('loading file')
	with open(args.input_srt, "r", encoding="utf-8") as f:
		subtitles = list(srt.parse(f))

	print('loading model and tokenizer')
	global model, tokenizer
	if model is None or tokenizer is None:
		init_models(args.src_lang,args.tgt_lang) 

	print('translating')
	for subtitle in subtitles:
		subtitle.content = translate_text(subtitle.content, args.src_lang, args.tgt_lang,args.max_length)

	print('writing output to file')
	with open(args.output_srt, "w", encoding="utf-8") as f:
		f.write(srt.compose(subtitles))

	print('done')

if __name__ == "__main__":
	main()
