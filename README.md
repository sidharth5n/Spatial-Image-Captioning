# Spatial-Image-Captioning
This is a PyTorch implementation of the Spatial Image Captioning. This repository is largely based on code from [Object Relation Transformer](https://github.com/yahoo/object_relation_transformer) which in turn is based on [Self-Critical Sequence Training](https://github.com/ruotianluo/self-critical.pytorch).

## Requirements
- The code is based on Python 3.7. List of dependencies can be found in [environment.yml]().
- The [cider](https://github.com/ruotianluo/cider) repo for evaluation code for CIDEr metric. Clone the repo into the `Spatial-Image-Captioning` folder.
- The [coco-caption](https://github.com/stevehuanghe/coco-caption-py3) library, which is used for generating different evaluation metrics. To set it up, clone the repo into the `Spatial-Image-Captioning` folder. Make sure to keep the cloned repo folder name as `coco-caption` and also to run the `get_stanford_models.sh` script from within that repo.

## Dataset Preparation

### Captions
Download the [preprocessed COCO captions](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) from Karpathy's homepage. Extract `dataset_coco.json` from the .zip file and copy it in to `data/`. This file provides preprocessed captions and also standard train-val-test splits.

Then run :
```
python scripts/prepro_labels.py --input_json data/dataset_coco.json --output_json data/cocotalk.json --output_h5 data/cocotalk
```
`prepro_labels.py` will map all words that occur <= 5 times to a special `UNK` token, and create a vocabulary for all the remaining words. The image information and vocabulary are dumped into `data/cocotalk.json` and discretized caption data are dumped into `data/cocotalk_label.h5`.

Next run :
```
python scripts/prepro_ngrams.py --input_json data/dataset_coco.json --dict_json data/cocotalk.json --output_pkl data/coco-train --split train
```

### Images
Download COCO images from the MS-COCO [website](http://mscoco.org/dataset/#download). We need 2014 training images and 2014 validation images. You should put the train2014 and val2014 folders in the same directory, denoted as `IMAGE_ROOT`

```
mkdir IMAGE_ROOT
pushd IMAGE_ROOT
wget http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip
popd
wget https://msvocds.blob.core.windows.net/images/262993_z.jpg
mv 262993_z.jpg IMAGE_ROOT/train2014/COCO_train2014_000000167126.jpg
```

The last two commands are needed to address an issue with a corrupted image in the MSCOCO dataset (see [here](https://github.com/karpathy/neuraltalk2/issues/4)). The prepro script will fail otherwise.

### Top Down Features
For models using attention maps from a lower convolutional layer or using global visual information. Download the file `resnet101.pth` from [here](https://drive.google.com/drive/folders/0B7fNdx_jAqhtbVYzOURMdDNHSGM). Copy the weights to `data/`.

Then run:
```
python scripts/prepro_feats.py --input_json data/dataset_coco.json --output_dir data/cocotalk --images_root IMAGE_ROOT
```

### Bottom Up Features
Download the pre-extracted features from [here](https://github.com/peteanderson80/bottom-up-attention). For the paper, the adaptive features were used.

```
mkdir data/bu_data
cd data/bu_data
wget https://imagecaption.blob.core.windows.net/imagecaption/trainval.zip
unzip trainval.zip
```
The .zip file is around 22 GB. Then return to the base directory and run:
```
python scripts/make_bu_data.py --output_dir data/cocobu
```

This will create `data/cocobu_fc`, `data/cocobu_att` and `data/cocobu_box`.

### Grid Features
To extract the horizontal and vertical spatial weights, run:
```
python prepro_areas.py --input_dir data/cocobu_box --output_dir data/cocobu_box_vec --num_grids 144
```

## Training

### Cross-Entropy
```
python train.py --id spatial_transformer_bu --caption_model spatial_pe --input_json data/cocotalk.json --input_img_feat_dir data/cocobu_att --input_grid_vec_dir data/cocobu_box_vec --input_label_h5 data/cocotalk_label.h5 --checkpoint_path checkpoints/spatial_transformer_bu --noamopt --noamopt_warmup 10000 --batch_size 15 --learning_rate 1e-3 --num_layers 6 --input_encoding_size 512 --ff_size 2048 --img_feat_size 2048 --save_checkpoint_every 1000 --val_images_use 5000 --perform_validation_every 2500 --max_epochs 30 --num_grids 144 --enc_pos_embedding 1 --enc_pos_type gxg --cross_attn 'xlinear'
```
To resume training, you can specify `--start_from` option to be the checkpoint path.

For RNN based models, scheduled sampling and learning rate decay can be used. For more options, see [opts.py]().

### Self-Critical Sequence Training
After training using cross-entropy loss, additional self-critical training produces signficant gains in CIDEr-D score.

First, copy the model from the pretrained model using cross entropy. (It's not mandatory to copy the model, just for back-up)
```
bash scripts/copy_model.sh spatial_transformer_bu spatial_transformer_bu_rl
```

```
python train.py --id spatial_transformer_bu_rl --caption_model spatial_pe --input_json data/cocotalk.json --input_img_feat_dir data/cocobu_att --input_grid_vec_dir data/cocobu_box_vec --input_label_h5 data/cocotalk_label.h5 --checkpoint_path checkpoints/spatial_transformer_bu_rl --batch_size 9 --learning_rate 1e-3 --num_layers 6 --input_encoding_size 512 --ff_size 2048 --img_feat_size 2048 --save_checkpoint_every 1000 --val_images_use 5000 --perform_validation_every 4000 --self_critical_after 30 --max_epochs 60 --num_grids 144 --enc_pos_embedding 1 --enc_pos_type gxg --cross_attn 'xlinear' --start_from checkpoints/spatial_transformer_bu_rl
```

## Evaluation
To evaluate the cross-entropy model, run:
```
python eval.py --dump_images 0 --num_images 5000 --model checkpoints/spatial_transformer_bu/model.pth --infos_path checkpoints/spatial_transformer_bu/infos_spatial_transformer_bu.pkl --image_root IMAGE_ROOT --input_json data/cocotalk.json --input_label_h5 data/cocotalk_label.h5 --input_img_feat_dir data/cocobu_att --input_grid_vec_dir data/cocobu_box_vec --language_eval 1
```

To evaluate cross-entropy + RL model, run:
```
python eval.py --dump_images 0 --num_images 5000 --model checkpoints/spatial_transformer_bu_rl/model.pth --infos_path checkpoints/spatial_transformer_bu_rl/infos_spatial_transformer_bu_rl.pkl --image_root IMAGE_ROOT --input_json data/cocotalk.json --input_label_h5 data/cocotalk_label.h5  --input_img_feat_dir data/cocobu_att --input_grid_vec_dir data/cocobu_box_vec --language_eval 1
```
