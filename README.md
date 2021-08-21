## A BRITISH SIGN LANGUAGE INTERPRETER

> The aim of the project is to create live British Sign Language interpreter using AI and ML. It's a web application that is using your camera, to predict shown by you signs and display them as a text.

## Table of contents

- [Models](#screenshots)
- [Technologies](#technologies)
- [Setup](#setup)
- [Status](#status)
- [Contact](#contact)

## Tensorboard

To use Tensorboard for every model statistics,
go to chosen model directory and run:

```
$ tensorboard --logdir "choose the model"/logs/
```

## Technologies

Backend:

- Python version: 3.6.6
- TensorFlow version: 2.3.1
- Keras version: 2.4.3
- OpenCV-python version: 4.3.0.36
- Flask version: 1.1.2
- Rest of the libraries saved in requirements.txt

## Setup

### To run the backend:

First of all make sure that you have Python version 3.6.6 installed on your machine.
Prepare virtual environment, for example with:

- pyenv (https://github.com/pyenv/pyenv-virtualenv)
- conda (https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html)
- venv (https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)

After creating and running your virtual environment make sure that your 'pip' package installer is working. Check its version for a test and if needed update it:

```
$ pip --version
$ pip install --upgrade pip
```

Now time to install all the requirements like TensorFlow, Keras etc. :

```
$ cd ../project folder/backend
$ pip install -r requirements.txt
```

If all the requirements has been installed time to run our application!

```
$ python index.py
```


## Generating images

To take a specified number of pictures use take_pictures() function placed in utilities file.
Choose your number of pictures starting from and ending with. Specify your camera, usually number
0 will be you first and default one. If you have more devices pick higher numbers:

```
.
.
#take_pictures(start_name, end_name, camera_num)

take_pictures(0, 50, 0)
```

## Model evaluation

Go to backend directory and use evaluation.py file. Specify testing dataset path and
model path and run the function. The method will print the scores and save Confusion Matrix in default directory

## Status

Project is: _in progress_

## Contact

Created by Stanislaw Polichnowski
