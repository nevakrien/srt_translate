# srt_translate

srt_translate is a Python application that uses the Hugging Face Transformers library to translate SRT (SubRip Text) files from one language to another. This repository contains the source code and instructions for setting up and running the translation application.

## Requirements

* Python 3.8 or higher
* Transformers library
* `srt` library

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/nevakrien/srt_translate.git
    cd srt_translate
    ```

2. Install the required Python packages:

   Using pip:

    ```sh
    pip install -r requirements.txt
    ```

   Using conda:

    ```sh
    conda create --name srt_translate --file requirements.txt
    conda activate srt_translate
    ```

## Usage

To translate an SRT file, use the following command:

```sh
python translate.py --input_srt <path_to_input_srt> --output_srt <path_to_output_srt> --tgt_lang <target_language>
```
 
all available language codes are listed in the `langs.txt` file. If you need to find a particular language code, you can use the following command: 
```sh 
cat langs.txt | grep heb 
```
which should print: "heb_Hebr"