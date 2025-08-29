# Context-aware Biometrics - Facial landmark detection and Expressions recognition
Project undertaken as part of my Masters thesis

## To try it:
This requires git-lfs (large file storage to be installed). Installation: https://git-lfs.com. Once installed:
`git lfs clone https://github.com/buratino91/thesis_project.git && cd thesis_project/`


### Create virtual env to install dependencies
```bash
python -m venv thesis_project
source thesis_project/bin/activate
```
### Windows
`thesis_project\Scripts\activate`

### Install dependencies
`pip install -r requirements.txt`

## How to load the trained weights and run it
open `main.py`. The model is already loaded using `trained_model = keras.models.load_model()` with the trained weights.
If you run `main.py`, it uses the test dataset for evaluation. Feel free to input any other images for prediction.

