# E2E-SpokenQA
This project aims to develope the code for end-to-end spoken question answering without any labeled transcription and ASR system.

### Prerequisite
- Spoken SQuAD dataset, including: 
    - `{split}-answer-span.csv`: question-answer pair and the answer span time
    - `{split}_audios`: directory thats contains the audio questions and contexts
    - `{split}-v1.1.json`: original text-SQuAD 1.1 dataset
### Running step
1. Preprocess Spoken SQuAD to link question ID to question audio file
`hash2question.py`
2. Specify the configuration in yaml file
`config.yaml`
3. Run training by `sh train.sh`, it simply run the `main.py` file. There are some arguments you need to specify:
    - `--config`: Path to experiment config file
    - `--name`: Experiment name for logging
    - `--upstream`: Specify the upstream variant according to torch.hub.list, for more details, referring to [s3prl/upstream](https://github.com/s3prl/s3prl/tree/master/s3prl/upstream) 

(More updates are comming soon)