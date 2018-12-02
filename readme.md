# Sentiment Analysis on Movie Reviews
## Background
Sentiment Analysis is also known as ​ Opinion Mining ​ is a field within Natural Language Processing
(NLP) that builds systems that try to identify and extract opinions within the text​ (Bo and Lillian, 2008). Currently, sentiment analysis has become a topic with great interests and developments since it has many practical applications, including social media monitoring, marketing analysis,
customer service, and product analytics. With the advantage of growing data size in most areas, a large number of texts expressing opinions and attitudes are available in review sites, blogs, and social media.

## PROJECT DETAILS
The objective of this project is to apply different machine learning and deep learning methods in the task of sentiment analysis of movie reviews. Specifically, there are multiple movie review websites such as Rotten Tomatoes, IMDB, and Flixster, where users can rate based on their own
feelings. Our sentiment analysis project aims at using raw movie review text with associated labels from IMDB to classify phrases on a scale of five classes: negative, somewhat negative, somewhat positive, positive (We use numerical value 0, 1, 2, 3 to represent them respectively). The challenging part of the task is dealing with obstacles like sentence
negation, abbreviation, language ambiguity, and metaphors.

## Data Preprocessing
The first part of the project is data-preprocessing. Since the original dataset contains only raw text with its label, we need to transform the shape of natural language and remove noises in order to better fit our model. Therefore, the data preprocessing contains the following steps:

> Tokenization & Segmentation  
Noise Removal (Remove stop words)   
Lemmaziation & Normalization

Besides, for the purpose of contrast with traditional NLP classification methods, it is also essential tovectorize phrases (TF-IDF, Word Embedding) in order to learn the correlation between texts rather than treating words as discrete symbols like the *bag of words* model.

After integrating the dataset, we have 29379, 22627, 22988 and 25995 samples for each of our four classes. We split our dataset into training set and testing set in the ratio of 4 to 1. 