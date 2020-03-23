## Seq2Seq

Based on the script [`run_ner.py`](https://github.com/huggingface/transformers/blob/master/examples/ner/run_ner.py) for Pytorch 

This example fine-tunes a transformer model to format date strings from long form to short form.
This is obviously not a production use case,
but it demonstrates the workflow for how to train a seq2seq model on arbitrary data.
Details and results provided by [@mgoldey](http://github.com/mgoldey).

### Environment

Pip requirements are stored in requirements.txt in this example.

```bash
python3 -m pip install -r requirements.txt
```

### Data preparation using Faker

Fake data can be made by running
```bash
python3 data_generator.py
```

By default, this uses a comparatively small training set of 10K examples and dev and test sets of 1K examples. 
You can change these values through command line flags.
```text
FLAGS
    --train_size=TRAIN_SIZE
    --dev_size=DEV_SIZE
    --test_size=TEST_SIZE
```

The training data format is given in paired input/output examples, as follows:
```text
input: twentieth march 1996
output: 20/03/1996
input: 15 aug 1998
output: 15/08/1998
input: third oct 1972
output: 03/10/1972
input: 15th august 1987
output: 15/08/1987
input: sixteenth oct 04
output: 16/10/2004
```

Let's define some variables that we need for further pre-processing steps and training the model:

```bash
# set env vars
export MAX_LENGTH=16

# Model types taken from https://huggingface.co/transformers/pretrained_models.html
export MODEL_TYPE=gpt2
# Specific pre-trained model used is `gpt2`
export MODEL_NAME=gpt2
export NUM_EPOCHS=1
export OUTPUT_DIR="date_${MODEL_TYPE}"
export BATCH_SIZE=64
export SAVE_STEPS=750
export SEED=1
```

Note that here we use `GPT2` since it allows for output such as `3/23/2020` without changing the tokenizer.

### Run seq2seq for on this toy problem

To start training, just run:

```bash
python3 run_seq2seq.py \
  --data_dir ./ \
  --model_type $MODEL_TYPE \
  --model_name_or_path $MODEL_NAME \
  --output_dir $OUTPUT_DIR \
  --max_seq_length $MAX_LENGTH \
  --num_train_epochs $NUM_EPOCHS \
  --per_gpu_train_batch_size $BATCH_SIZE \
  --save_steps $SAVE_STEPS \
  --seed $SEED \
  --do_train \
  --do_eval \
  --do_predict
```

If your GPU supports half-precision training, just add the `--fp16` flag. 

After training, the model will be both evaluated on development and test datasets.


#### Evaluation

Evaluation on development dataset outputs the following for our example:
```bash
03/23/2020 16:12:05 - INFO - __main__ -   ***** Eval results  *****
03/23/2020 16:12:05 - INFO - __main__ -     f1 = 0.9405
03/23/2020 16:12:05 - INFO - __main__ -     loss = 0.18552476764907913
03/23/2020 16:12:05 - INFO - __main__ -     precision = 0.9405
03/23/2020 16:12:05 - INFO - __main__ -     recall = 0.9405
```

On the test dataset the following results could be achieved:
```bash
03/23/2020 16:12:25 - INFO - __main__ -   ***** Eval results  *****
03/23/2020 16:12:25 - INFO - __main__ -     f1 = 0.9396875
03/23/2020 16:12:25 - INFO - __main__ -     loss = 0.18529377321875284
03/23/2020 16:12:25 - INFO - __main__ -     precision = 0.9396875
03/23/2020 16:12:25 - INFO - __main__ -     recall = 0.9396875
```

### Missing features
- No test of Tensorflow version
- PreTrainedEncoderDecoder needs greater test coverage
- BERT models appear to completely fail for this task
- Max length should be enforced
- Attention could be used to improve the loss function calculation
