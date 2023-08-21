# unsupervised-flower-image-segmentation

Installation steps:
```
git clone https://github.com/Ignacio-Ibarra/unsupervised-flower-image-segmentation.git

cd unsupervised-flower-image-segmentation

python3 -m venv <venv_name>

source venv_name/bin/activate

pip install --upgrade pip

pip install -r requirements.txt

```
Download flowers image data from [here](https://www.kaggle.com/datasets/olgabelitskaya/flower-color-images?resource=download) and store them in an `input` folder, resulting in the following directory structure: 

```
.
├── README.md
├── input
│   ├── flower_images
│   │   ├── 0001.png
│   │   ├── 0002.png
│   │   ├── 0003.png
|   |   ...
│   │   └── flower_labels.csv
│   └── flowers
│       ├── 00_001.png
│       ├── 00_002.png
│       ├── 00_003.png
|       ...
├── notebook.ipynb
└── requirements.txt

```