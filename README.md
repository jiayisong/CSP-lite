# CSP-lite: Real-Time And Efficient Keypoint-Based Pedestrian Detection
## Prepare Dataset
Download the dataset from the official website and organize the file structure in the following form. 
### caltech
```text
caltech
├── anno_train10x_alignedby_RotatedFilters
    ├── set00_V000_I00002.txt
    ├── set00_V000_I00005.txt
    ├── set00_V000_I00008.txt
    ├── '''
├── anno_test_1xnew
    ├── set06_V000_I00029.jpg.txt
    ├── set06_V000_I00059.jpg.txt
    ├── '''
├── frame
    ├── set00_V000_I00002.jpg
    ├── set00_V000_I00005.jpg
    ├── set00_V000_I00008.jpg
    ├── '''
```
### citypersons
```text
cityperson
├── gtBboxCityPersons
    ├── train
        ├── aachen
           ├── aachen_000000_000019_gtBboxCityPersons.json
           ├── '''
        ├── '''
    ├── val
        ├── frankfurt
           ├── frankfurt_000000_000294_gtBboxCityPersons.json
           ├── '''
        ├── '''
├── leftImg8bit
    ├── train
        ├── aachen
           ├── aachen_000000_000019_leftImg8bit.png
           ├── '''
        ├── '''
    ├── val
        ├── frankfurt
           ├── frankfurt_000000_000294_leftImg8bit.png
           ├── '''
        ├── '''
```
## Prepare Environment
* Python 3.7
* numpy 1.21.1
* opencv-python 4.5.3.56
* thop 0.0.31.post2005241907
* torch 1.8.0+cu111
* torchvision 0.9.0+cu111
## Prepare Code
Download or clone the repository code and modify caltech_root in config_caltech.py and cityperson_root in config_cityperson.py.
## Run Code
# Train
```python
python3 train.py
```
# Test
```python
python3 test.py
```
# Test FPS
```python
python3 test_FPS.py
```
