# 🌴 Welcome to Level 1 of the 2024 EM/Dev Assessment! 🌴
This is the first level of the Vanderbilt Data Science dev assignment! In this level, you will be choosing a topic you are interested in and cleaning a dataset so that it can be used for later analysis. The data can be about anything: a social issue, a topic in your major field of study, etc.

<br><br> 
Some datasets and their focus questions:
- [YouTube Channel Niche](https://www.kaggle.com/datasets/nelgiriyewithana/global-youtube-statistics-2023), using this dataset, create a model to predict the category or niche of YouTube channels. This is a <b>classification</b> problem.
- [Pet's Facial Expression Image Dataset](https://www.kaggle.com/datasets/anshtanwar/pets-facial-expression-dataset), using this dataset, create a model to classify images of animals into different categories. This is a <b>classification</b> problem.
- [Customer Personality Analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis), using this dataset, create a model to predict whether a customer has made at least one complaint in the last 2 years. This is a <b>classification</b> problem.
- [ECG Heartbeat Categorization Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat?select=ptbdb_abnormal.csv), using this dataset, create a model to cluster ECG heartbeats into different categories. This is a <b>clustering</b> problem.
- [Pokemon with stats](https://www.kaggle.com/datasets/abcsds/pokemon), using this dataset, create a model to cluster Pokemon into different categories. This is a <b>clustering</b> problem.

Some of these datasets are silly and some of them are serious, some of them make sense in context of their focus question and some of them don't. The point of this level is to get you thinking about the features of the dataset you choose and how they relate to the focus question you want to answer. <b>We want you to choose your own dataset and focus question to explore for your coding challenge.</b>


## Objective:
Your task is to get this data loaded and cleaned in your notebook. This means that you should choose what (if any) libraries you are using, and how you are going to clean the data. You should also keep track of your answers to these questions:
- What is the dataset you chose? (include a link to the dataset)
- What is the focus question you are trying to answer with this dataset? Include whether this is a classification, clustring, or other type of problem.
- What is your plan for cleaning this dataset? (you don't have to go into too much detail, just a general idea of what you are going to do) 
- Any extra information you'd like to include

## What are we looking for?
Ideally, use Python for for your coding challenge!<br>
We strongly encourage you to use a dataset that is not from Kaggle, and to come up with your own focus question.<br>
Complex problems such as clustering and neural network are evaluated favorably!<br>

## Tips and Tricks
  **What libraries should I use?**
  We recommend [NumPy](https://numpy.org/), [pandas](https://pandas.pydata.org/), and [scikit-learn](https://scikit-learn.org/stable/). Ultimately, the libraries you use are up to you, but these are the ones we recommend.

  **This section is super short??**
  Yup! The first step is always the hardest part but in terms of the actual code/submission, this is the shortest portion of the entire coding challenge. Don't worry if you feel like you're missing something, this is just the beginning!

  **How do I clean a data set?**
  Unless you want to agonize through hundreds of hours scrolling down an Excel spreadsheet, data is cleaned
  using programs specifically made to filter through keywords in formatted datasets.
  [This article](https://www.tableau.com/learn/articles/what-is-data-cleaning)
  explains what cleaning data is and steps that can be taken, and
  [this article](https://towardsdatascience.com/so-youve-got-a-dataset-here-s-how-you-clean-it-5d0b04a2ed86)
  has many specific cleaning programs that can be done in python. Google is your friend!
  How data is cleaned depends on the format and type of the dataset, but python has very extensive and intuitive support in
  data management, making it ideal for data science.

  **More cleaning tips:**
  Not all features are created equally! Some features are important in answering a focus question (i.e. the number of eye-shapes in an image when trying to classify humans and aliens), and some aren't (i.e. the favourite color of the subject in an image when trying to classify humans and aliens). It is up to you to decide what is and isn't important! Additionally, you might find additional features not in a dataset, i.e. BMI, to be useful in your focus question.