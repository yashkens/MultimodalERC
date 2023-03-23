### Multimodal model with concatenation fusion

```python3 multimodal_concat/train.py -n <wandb project name> --lr <learning rate> --batch_size <batch_size> --seed <random seed>```

Notes: 
* Each modality features are represented by **last hidden state** of a modality-specific model
* Text features are taken from [CLS] token
* Video features are pooled from 16 frames into 1
* Audio features are **WIP**
* The model consists of two linear layers for now
