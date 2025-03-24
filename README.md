# MSDZip
A general-purpose lossless compressor via Stepwise-parallel and Learning-based Prediction

# News
This work has been accepted by The ACM Web Conference (WWW '25).

# Running
## Regular
```
# Compression
python compress.py <file> <file>.mz
# Decompression
python decompress.py <file>.mz <file>.mz.out
```

## Stepwise-parallel
```
# Compression
bash sp-compress.sh <file> <file>.mz
# Decompression
bash sp-decompress.sh <file>.mz <file>.mz.out
```

# Dataset
| ID  | Name           | Type          | Size (Byte)   | Link                                                                                   |
|:---:|:--------------:|:-------------:|:-------------:|:--------------------------------------------------------------------------------------:|
| D1  | Enwik8         | text          | 100000000     | https://mattmahoney.net/dc/enwik8.zip                                                  |
| D2  | Text8          | text          | 100000000     | https://mattmahoney.net/dc/text8.zip                                                   |
| D3  | Enwik9         | text          | 1000000000    | https://mattmahoney.net/dc/enwik9.zip                                                  |
| D4  | Book           | text          | 1000000000    | https://storage.googleapis.com/huggingface-nlp/datasets/bookcorpus/bookcorpus.tar.bz2  |
| D5  | Silesia        | heterogeneous | 211938580     | https://sun.aei.polsl.pl//~sdeor/corpus/silesia.zip                                    |
| D6  | Backup         | heterogeneous | 1000000000    | https://drive.google.com/file/d/18qvfbeeOwD1Fejq9XtgAJwYoXjSV8UaC/view?usp=sharing     |
| D7  | CLIC           | image         | 243158876     | https://www.compression.cc/tasks/                                                      |
| D8  | ImageTest      | image         | 470611702     | http://imagecompression.info/test_images/rgb8bit.zip                                   |
| D9  | GoogleSpeech   | audio         | 327759206     | http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz                       |
| D10 | LJSpeech       | audio         | 293847664     | https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2                             |
| D11 | DNACorpus      | genome        | 685597124     | https://sweet.ua.pt/pratas/datasets/DNACorpus.zip                                      |
| D12 | GenoSeq        | genome        | 1926041160    | https://www.ncbi.nlm.nih.gov/sra/ERR7091247                                            |

# Contact Us
Email: mahd@nbjl.nankai.edu.cn
