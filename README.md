# BISS

train whisper2diff

```
python ./models/speech_trainer.py --max_steps 1000000 --fp16 --batch_size 32 --learning_rate 2e-4 --num_gpus 8 'results'
```

train semantic text encoder

```
python models/text_decoder/train_EM.py --subject 94 --session \_single_f
```

semantic decoding

```
python ./models/text_decoder/decoding.py --subject 4 --experiment perceived --word_info_path 'data/text_decoding/test_token.csv' --logname 'text_gpt_chahua_decode.log'
```
