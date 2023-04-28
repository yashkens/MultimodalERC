## Libraries

1. os
2. random
3. typing
4. argparse


5. numpy
6. pandas
7. scipy
8. sklearn
9. cv2 - 4.7.0.68 - устанавливать как opencv-python


10. torch - 1.13.1
11. transformers - 4.28.0.dev0
12. accelerate - 0.18.0


13. wandb

## Data (CREMA-D)

1. CSV files (содержит название эмоции и путь к видео), лежат в директории

`/cephfs/home/mikhaylova/multimodal_emo_reco/CREMA-D/CSV/`

**Размер:** 208K

2. Video files, format: `flv`, лежат в директории

`/cephfs/home/mikhaylova/multimodal_emo_reco/CREMA-D/VideoFlash/`

**Размер**: 2.3 G

3. Audio files, format: `wav`, лежат в директории

`/cephfs/home/mikhaylova/AudioWAV-CREMA/`

**Размер:** 580 M

## Few words about model

Мы используем предобученный персивер, обученный изначально на датасете Kinetics-700. 

Модель отдает предсказание, что происходит на видео, а также декодированные видео и аудио. Метки классификатора представляют собой простые дискрипции действий, вроде *играть на скрипке*, *смотреть в глаза*, *кататься на сёрфе* и пр. Всего меток - 700 штук.

На вход модели подается словарь вида

<code>{
    'image': tensor(размер_батча, кол-во_фреймов_в_видео, кол-во_каналов, высота, ширина), - представление видео
    'audio': tensor(размер_батча, 1, длина_аудио, кол-во_каналов), - представление аудио
    'label': tensor((кол-во_меток_класса,)) - one-hot вектор с лейблами
}</code>

На выходе мы получаем похожий словарь со всеми модальностями.

Важно также отметить, что при получении результата мы отдаем модели не только такого рода словарь (см. выше), но и словарь с индексами для сабсэмпла данных модели. 

<code>subsampling = {
    "image": torch.arange(размер_части_видео * индекс_части, размер_части_видео * (индекс_части + 1)),
    "audio": torch.arange(размер_части_аудио * индекс_части, размер_части_аудио * (индекс_части + 1)),
    "label": None,
}</code>

Sources:
1. <a href='https://github.com/huggingface/transformers/blob/v4.27.2/src/transformers/models/perceiver/modeling_perceiver.py#L3355'>How the model works Git-HuggingFace</a>
2. <a href='https://huggingface.co/deepmind/multimodal-perceiver'>Its HuggingFace page</a>
3. <a href='https://github.com/huggingface/blog/blob/main/perceiver.md#perceiver-for-multimodal-autoencoding'>Few words about model training</a>
4. <a href='https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Perceiver/Perceiver_for_Multimodal_Autoencoding.ipynb'>Nice tutorial on topic</a>

## Fine-tuning preparation

1. Модель состоит из следующих частей:
    - input_preprocessor
    - output_postprocessor
    - embeddings
    - encoder
    - decoder
    
2. Нам нужно поменять первые две части

**2.1.** input_preprocessor - предобрабатывает key/value вектор для модели, отдает на выходе вектора размером 1\*704 для каждой модальности

Внутри есть энкодинг аудио и видео модальностей - его не трогаем. Меняем 

а) словари label2id и id2label в конфиге модели, вслед за ними меняется config.num_labels,

б) паддинг для вектора с лейблом,

в) значение минимального паддинга в модели

**2.2.** output_postprocessor - декодирует выход модельки. Меняем

а) слой классификатора для лейбла

<code>    (label): PerceiverClassificationPostprocessor(
      (classifier): Sequential(
        (0): Linear(in_features=512, out_features=256, bias=True)
        (1): Dropout(p=0.3, inplace=True)
        (2): Linear(in_features=256, out_features=6, bias=True)
      )</code>

## Previous training hyperparams

1. batch_size = 2
2. epochs = 10
3. learning rate = 0.01

<a href='https://wandb.ai/annette-mikhaylova/crema-perceiver?workspace='>Wandb link</a>
