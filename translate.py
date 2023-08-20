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

def translate_text(text, src_lang, tgt_lang):
	# Encode the source text, translate it, and then decode the translation
	inputs = tokenizer.encode(text, return_tensors="pt")
	translation = model.generate(inputs, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang])
	translated_text = tokenizer.decode(translation[0], skip_special_tokens=True)
	return translated_text

def main():
	# Define the command-line arguments
	parser = argparse.ArgumentParser(description="Translate an SRT file.")
	parser.add_argument("--input_srt", required=True, help="Path to the input SRT file.")
	parser.add_argument("--output_srt", required=True, help="Path to the output SRT file.")
	#eng_Latn
	parser.add_argument("--src_lang", default=None, help="Source language code.")
	parser.add_argument("--tgt_lang", required=True, help="Target language code.")
	args = parser.parse_args()

	# Read and parse the SRT file
	with open(args.input_srt, "r", encoding="utf-8") as f:
		subtitles = list(srt.parse(f))

	global model, tokenizer
	if model is None or tokenizer is None:
		init_models(args.src_lang,args.tgt_lang) 

	# Translate the content of each subtitle
	for subtitle in subtitles:
		subtitle.content = translate_text(subtitle.content, args.src_lang, args.tgt_lang)

	# Write the translated subtitles to a new SRT file
	with open(args.output_srt, "w", encoding="utf-8") as f:
		f.write(srt.compose(subtitles))

if __name__ == "__main__":
	main()
