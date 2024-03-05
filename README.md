# CSP-lite: Real-Time And Efficient Keypoint-Based Pedestrian Detection
## Prepare Dataset
Download the dataset from the official website and organize the file structure in the following form. 
### caltech
https://drive.google.com/file/d/15WftklRDGdWz2-9DojhIfnHh4XAKv0Xv/view?usp=sharing
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
* Python=3.7
* numpy=1.21.1
* opencv-python=4.5.3.56
* thop=0.0.31.post2005241907
* torch=1.8.0+cu111
* torchvision=0.9.0+cu111
## Prepare Code
Download or clone the repository code and modify '*caltech_root*' in [config_caltech.py](config_caltech.py#L10) and '*cityperson_root*' in [config_cityperson.py](config_cityperson.py#L11).
## Run Code
### Train
```python
python3 train.py --dataset caltech --gpu 0 --run_number 1 --NNL --neck mulbn --neck_depth 4 --loss_weight 1 1 1
```
Accepts the following parameters：

- `--dataset`: cityperson or caltech
- `--gpu`: which GPU to use
- `--run_number`: Experiment number
- `--NNL`: NNL in the paper
- `--neck`: 'no', 'res' or 'mulbn'
- `--neck_depth`: Number of neck modules
- `loss_weight`: loss weight
### Test
```python
python3 test.py --dataset caltech --gpu 0 --run_number 1 --NNL --neck mulbn --neck_depth 4
```
Parameters are consistent with training.
### Test FPS
```python
python3 test_FPS.py
```
## Running Result
|  Dataset   |  Scale | Reasonable  | Heavy | Partial |Bare| Small | Medium | Large| All | Time(ms) | Model |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|------|------|------|
|  Caltech   |  480*640   |    4.1 | 13.5 | - |-| 5.3 |  - |-| 49.4  | 6.3    |  [checkpoint](https://drive.google.com/file/d/1kl9CWQz6OJBd08Gw8mTznXY24dFo2ZBv/view?usp=sharing) |
|  Caltech(Cityperson pre-training)   |  480*640   |    3.4 | 9.7 | - |-| 4.6 |  - |-| 46.4  | 6.3    |  [checkpoint](https://drive.google.com/file/d/1AqClnRgJ2CuH1noUM83pq2GQIOn-g_DL/view?usp=sharing) |
|   Cityperson  | 1024*2048    |  11.0 | 48.8 | 10.3 | 7.3 | 15.4 | 4.6 | 5.2 |  -| 35.7  |  [checkpoint](https://drive.google.com/file/d/1PwQIa_wtJepoKIyZh3KP7r2AhEbKJN6S/view?usp=sharing) |
|   Cityperson  |  1312*2624   |  10.2 | 46.6 | 8.8 | 6.6 | 13.5 | 2.4 | 5.3 |  -| 50.9   |     |
