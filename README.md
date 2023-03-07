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