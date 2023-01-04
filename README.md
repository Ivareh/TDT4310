# TDT 4310 - Intelligent Text Analytics and Language Understanding - Spring 2023

This is a space for the labs of TDT4310, including the tasks and supplementary material.

Main points:
1. Labs released every second Friday (see dates below)
2. You get an initial week to work with (and possibly finish) the lab
3. Lab session takes place. I will do my best to assist as many as I can. Note that this is an open session, where I encourage you to talk to your peers and discuss in groups.
4. Deadline is 1 week after the lab session.

For any other questions that may not be suitable for the lab sessions, contact us through the emails provided on blackboard.

## `Curriculum 📚`
This year, the course curriculum is mostly based around the brand new book by *Ekaterina Kochmar* -- Getting Started with Natural Language Processing. For purchasing instructions (+ discount), visit the course page on Blackboard.

<p align="center">
    <img src="assets/kochmar.png" width=150>
</p>

We'll also use the NLTK library. As usual, more information on textbooks and other resources can be found on blackboard.

## `Labs 📝`
Labs will be found in this main directory as markdown files (e.g. [LAB_1.md](LAB_1.md)).

Here's a brief overview of the labs:

| Lab | Published | Deadline | Topic | Smart keyboard features | Libraries | Chapters |
| - | - | - | - | - | - | - |
| 1 | Jan. 13 | Jan. 27 | Basic text processing, introduction to word vectors and language modeling | Next-word prediction | NLTK | 2, 3 |
| 2 | Jan. 27 | Feb. 10 | Part-of-speech tagging, stemming/lemmatization, TF-IDF | Utilize part-of-speech for predictions | NLTK | 4, 5, 6 |
| 3 | Feb. 10 | Feb. 24 | Wordnet and SentiWordNet, dependency parsing, POS chunking | Sentiment-based and multi-word predictions | spaCy, Scikit-learn | 7, 8 |
| 4 | Feb. 24 | Mar. 10 | Unsupervised topic modeling and named entities | Topic-aware predictions | Scikit-learn, Gensim | 9, 10 |
| Project | Mar. 10 | inf | - | - | - | - |
|

From this point on, the focus will be shifted towards your projects. This will involve machine learning and deep learning techniques, building upon the tasks you've completed in the labs.

The level of difficulty will increase over time, as can be seen from the topics. Some of you will definitely feel the need to get more applications of modern state-of-the-art approaches, but note that you're free to explore anything in the project later in the course!

## `The main theme: building a smart keyboard 💻`
Throughout the labs, alongside a few questions and simple tasks, you will be implementing a **smart keyboard**, mostly from scratch. I have prepared a full-stack application, where your task is to build a system that suggests the next word(s).
The frontend application is built using [**React**](https://reactjs.org/) and is accessible directly through [npm](https://www.npmjs.com) or through the supplied (precompiled) Electron application. The former will give you more flexibility when running the systems locally.

**The backend, which you will be working with**, is built using [**Flask**](https://palletsprojects.com/p/flask/). This course assumes fairly good knowledge of programming in general, and you are expected to be able to debug potential issues with the system yourself. I will, of course, do my best to aid you in this process. A screenshot of the application is shown below.

<p align="center">
    <img src="assets/electronapp.png" width=400>
</p>

The frontend application (running on your PC) is also set up to be exposed on your local network, allowing access from any other device connected to it. Below is an example with it accessed from a phone:

<p align="center">
    <img src="assets/lab_phone.png" width=250>
</p>

## `Setup 🛠`
You are required to run a few commands to get the system up and running, both for the frontend and the backend.

### `Backend (Python/Flask server) 🐍`
I suggest you download all packages in NLTK so you can experiment freely without caring about downloading separate packages later on. Data is stored in `nltk_data` in your user folder. The total size is around 3 GB. If you're low on space, look up how to download the packages you require.
- `pip install -r requirements.txt`
- `python -m nltk.downloader all`
### `Frontend (React application) ⚛`
The frontend application can be run in two ways:
1. Using the precompiled Electron application (recommended)
2. Using npm (more flexible, but requires some setup, as described below)

- install `npm`:
    - Linux:
        - `sudo apt install npm`
    - MacOS:
        - `brew install npm`
            - If you don't have brew, install it first: https://brew.sh/
    - Windows:
        - https://www.npmjs.com/get-npm
        - With chocolatey: `choco install npm`
- `cd labs_frontend`
- `npm install`
- `npm start`


## `Grading 👨‍🏫`
The labs are not graded, but pass/fail. You need to pass all labs to be eligible for the exam. The criteria will be explained further below.

## `Questions and help 🙋‍♂️`
Try to keep questions regarding labs to the lab session. Other questions may be posted on Blackboard.

Some info on the libraries used:
- NLTK has an extensive collection of both explanations and sample usages:
    - https://www.nltk.org/api/nltk.html and https://www.nltk.org/howto.html
- Spacy: https://spacy.io/api
- Scikit-learn: https://scikit-learn.org/stable/modules/classes.html
- Gensim: https://radimrehurek.com/gensim/apiref.html
