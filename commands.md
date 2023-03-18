## Sample commands -- for running inside docker compose

### 1. For generating mask

```cmd
python bin/gen_mask_dataset.py "./configs/data_gen/random_medium_256.yaml" "./data/bgimages2paste/" "./output/auto_gen_mask"
```

## 2. Prediction

```cmd
python3 bin/predict.py model.path=$(pwd)/big-lama indir=$(pwd)/data/pan-with-mask outdir=$(pwd)/output/object_removal_result
```