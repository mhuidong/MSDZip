# MSDZip
An general-purpose lossless compressor via Stepwise-parallel and Learning-based Prediction

# Running
## Regular
```
# Compression
python compress.py <file> <file>.mz
# Decompression
python decompress.py <file>.mz <output>.mz.out
```

## Stepwise-parallel
```
# Compression
bash sp-compress.sh <file> <file>.mz
# Decompression
bash sp-decompress.sh <file>.mz <file>.mz.out
```

# Dataset
| ID  | Name       | Type          | Size (Byte) | Description                                                         | Link                                                                                   |
|-----|------------|---------------|-------------|---------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| D1  | Enwik8     | text          | 100000000   | First $2^8$ bytes of the English Wikipedia dump on 2006             | https://mattmahoney.net/dc/enwik8.zip                                                  |
| D2  | Text8      | text          | 100000000   | First $2^8$ bytes of the English Wikipedia (only text) dump on 2006 | https://mattmahoney.net/dc/text8.zip                                                   |
| D3  | Enwik9     | text          | 1000000000  | First $2^9$ bytes of the English Wikipedia dump on 2006             | https://mattmahoney.net/dc/enwik9.zip                                                  |
| D4  | Book       | text          | 1000000000  | First $2^9$ bytes of BookCorpus                                     | https://storage.googleapis.com/huggingface-nlp/datasets/bookcorpus/bookcorpus.tar.bz2  |
| D5  | Silesia    | heterogeneous | 211938580   | A heterogeneous corpus of 12 documents with various data types      | https://sun.aei.polsl.pl//~sdeor/corpus/silesia.zip                                    |
| D6  | Backup     | heterogeneous | 1000000000  | $2^9$ bytes random extract from the disk backup of TRACE            | https://drive.google.com/file/d/18qvfbeeOwD1Fejq9XtgAJwYoXjSV8UaC/view?usp=sharing     |
| D7  | CLIC       | image         | 243158876   | The classical image compression benchmark of the 6th CLIC 2024      | https://www.compression.cc/tasks/                                                      |
| D8  | ImageTest  | image         | 470611702   | A new 8-bit benchmark dataset for image compression evaluation      | http://imagecompression.info/test_images/rgb8bit.zip                                   |
| D9  | GoogleSpeech    | audio         | 327759206   | First 10,000 audio files of the Google Speech Commands Dataset      | http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz                       |
| D10 | LJSpeech   | audio         | 293847664   | First 10,000 audio files of  the LJ Speech Dataset                  | https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2                             |
| D11 | DNACorpus  | genome        | 685597124   | A corpus of DNA sequences from 15 different species                 | https://sweet.ua.pt/pratas/datasets/DNACorpus.zip                                      |
| D12 | GenoSeq | genome        | 1926041160  | A collection of genomics sequencing dataset with FastQ format       | https://www.ncbi.nlm.nih.gov/sra/ERR7091247                                            |


