# meta_sift_artifacts

1. Create Download dataset form this link and put it on ./dataset

2. Use    pip install -r requirements.txt    to install required packages.

3. Run the quick_start.ipynb!


# For evaluation in GTSRB

For Badnets:
python main.py

For Narcissus:
python main.py --corruption_type narcissus --corruption_ratio 0.1

For targeted label filpping:
python main.py --corruption_type targeted_label_filpping --corruption_ratio 0.5

# Plug in the meta_sift function?
```python
class Args:
    num_classes = 43
    tar_lab = 38
    repeat_rounds = 5
    res_epochs = 1
    warmup_epochs = 1
    batch_size = 128
    num_workers = 16
    v_lr = 0.0005
    meta_lr = 0.1
    top_k = 15
    go_lr = 1e-1
    num_act = 4
    momentum = 0.9
    nesterov = True
    random_seed = 0
args=Args()
clean_idx = meta_sift(args, dataset)
```
Change the parameter in args and change the dataset into your poisoned dataset!, it will return about 1000 clean sample indices. You can use Subset to create the subset dataset.