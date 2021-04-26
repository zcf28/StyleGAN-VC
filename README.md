## [StyleGAN-VC]

This is a pytorch implementation of  one-shot Voice Conversion

The converted voice examples are in `stylegan/samples` and `stylegan/results` directory

.

## [Dependencies]
- Python 3.6+
- pytorch 1.5
- librosa 
- pyworld 
- soundfile


## [Usage]

### dataset

Download the VCC2020 dataset to the **dataset** directory 

You can download from 

```
https://github.com/nii-yamagishilab/VCC2020-database
```

The downloaded zip files are extracted to `vcc2020_database_training_source/source`. and `vcc2020_database_evaluation\vcc2020_database_evaluation`.

1. **training set:** In the paper, the author choose **four speakers** from `vcc2020_database_training_source/source`. So we  move the corresponding folder(eg. SEF1,SEF2,SEM1,SEM2 ) to `dataset/vcc2020/speakers`. (Each speaker contains 70 sentences )
2. **testing set** In the paper, the author choose **four speakers** from `vcc2020_database_evaluation\vcc2020_database_evaluation`. So we  move the corresponding folder(eg. SEF1,SEF2,SEM1,SEM2 ) to `dataset/vcc2020/speakers_test`. (Each speaker contains 25 sentences)

The data directory now looks like this:

```
dataset/vcc2020
├── speakers  (training set)
│   ├── SEF1
│   ├── SEF2
│   ├── SEM1
│   └── SEM2
├── speakers_test (testing set)
│   ├── SEF1
│   ├── SEF2
│   ├── SEM1
│   └── SEM2
```

### Preprocess

Extract features (mcep, f0, ap) from each speech clip.  The features are stored as npy files. We also calculate the statistical characteristics for each speaker.

```
python preprocess.py
```

This process may take minutes !

Finally, you will find `dataset/processed_256` and `dataset/etc_256`, They recorded mcep and f0


### Train

```
python main.py
```

### Convert

```
python main.py --mode test --test_iters 200000 --src_speaker SEM1 --trg_speaker "['SEM2','SEF1','SEF2']"
```

### Note

if you want realize one-shot VC, You should save the unseen speakers f0 parameter in the  `dataset/etc_256` as you did in the previous **Preprocess**, then you take unseen speakers in `dataset/vcc2020/speakers_test`

## [Model structure]

[](https://github.com/zcf28/StyleGAN-VC/blob/master/fig/generator.jpg)

## [Generator structure]

[](https://github.com/zcf28/StyleGAN-VC/blob/master/fig/model.jpg)



## [Reference]

[tensorflow StarGAN-VC code](https://github.com/hujinsen/StarGAN-Voice-Conversion)

[StarGAN code](https://github.com/taki0112/StarGAN-Tensorflow)

[CycleGAN-VC code](https://github.com/leimao/Voice_Converter_CycleGAN)


[pytorch-StarGAN-VC code](https://github.com/liusongxiang/StarGAN-Voice-Conversion)

[StarGAN-VC paper](https://arxiv.org/abs/1806.02169)

[StarGAN paper](https://arxiv.org/abs/1806.02169)

[CycleGAN paper](https://arxiv.org/abs/1703.10593v4)

---

If you feel this repo is good, please  **star**  ! 

Your encouragement is my biggest motivation!
