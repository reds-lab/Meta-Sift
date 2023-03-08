# Meta-Sift: A PyTorch Implementation
This is the code implementation for USENIX'23 paper [Meta-Sift: How to Sift Out a Clean Subset in the Presence of Data Poisoning?](https://arxiv.org/abs/2210.06516)
This github repo focuses on road-mapping the three main claims we developed in the “Meta-Sift” paper:
1. Defense performance is sensitive to the purity of the
base set.
2. Both existing automated methods and human inspec-
tion fail to identify a clean subset with high enough
precision.
3. Our proposed solution, Meta-Sift, can successfully obtain a stable subset in many different poison situations.

# A quick start for implementation
The file `quick_start.ipynb` file contains a simple implementation to verify our three points of view, the following part is guidance on how to run it:

1. Download dataset `gtsrb_dataset.h5` form this [link](https://drive.google.com/file/d/1SKYMwrnjEyFjjc7UWTdAyAjFI_demNtD/view?usp=sharing) and put it on './dataset' folder.

2. Use `pip install -r requirements.txt` to install required packages.

3. Run the `quick_start.ipynb`, and more detailed comments are in the file!

# More evaluation results for Meta-Sift?

In order to prove that our method can work in many poisoning methods, we choose `targeted label filpping`, `Narcissus` and `Badnets` as the representative of three different types of attacks. Here is the start command: 

For targeted label filpping from class 2 to class 38:  
`python main.py --corruption_type targeted_label_filpping --corruption_ratio 0.5`  

For Badnets in class 38:  
`python main.py`  

For Narcissus in class 38:  
`python main.py --corruption_type narcissus --corruption_ratio 0.1`  


# Human inspection experiment

Another important claim in out paper is that humans cannot filter a clean base set, and the labeling tools and dataset used in the experiments can be found at `./human_exp` folder.

