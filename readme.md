# Sentiment Analysis on Movie Reviews
University of Delaware   
CISC 882 Natural Language Proccessing   
Final Project  
## Background
Sentiment Analysis is also known as ​Opinion Mining​ is a field within Natural Language Processing (NLP) that builds systems that try to identify and extract opinions within the text​ (Bo and Lillian, 2008). Currently, sentiment analysis has become a topic with great interests and developments since it has many practical applications, including social media monitoring, marketing analysis, customer service, and product analytics. With the advantage of growing data size in most areas, a large number of texts expressing opinions and attitudes are available in review sites, blogs, and social media.

## Project Details
The objective of this project is to apply different machine learning and deep learning methods in the task of sentiment analysis of movie reviews. Specifically, there are multiple movie review websites such as Rotten Tomatoes, IMDB, and Flixster, where users can rate based on their own feelings. Our sentiment analysis project aims at using raw movie review text with associated labels from IMDB to classify phrases on a scale of five classes: negative, somewhat negative, somewhat positive, positive (We use numerical value 0, 1, 2, 3 to represent them respectively). The challenging part of the task is dealing with obstacles like sentence negation, abbreviation, language ambiguity, and metaphors.

### Data Preprocessing
The first part of the project is data-preprocessing. Since the original dataset contains only raw text with its label, we need to transform the shape of natural language and remove noises in order to better fit our model. Therefore, the data preprocessing part contains the following steps:

> Tokenization & Segmentation  
Noise Removal (Remove stop words)   
Lemmaziation & Normalization

Besides, for the purpose of contrasting with traditional NLP classification methods, it is also essential to vectorize phrases (TF-IDF, Word Embedding) in order to learn the correlation between texts rather than treating words as discrete symbols like the *bag of words* model.   

#### Tokenization & Segmentation  
With the advance of the famous Python NLP library - [NLTK](https://www.nltk.org/), we are able to tokenize and segment our movie reviews.   
Specifically, Tokenizers divide strings into lists of substrings. For example, tokenizers can be used to find the words and punctuation in a string (from our dataset):   
> 'One would expect cast although thornton really tries can't really blame writers shame!'   
> ['One', 'would', 'expect', 'cast', 'although', 'thornton', 'really', 'tries', 'can', ''', 't', 'really', ...]

#### Noise Removal
Once we have parsed the text from our movie review dataset, the challenge is to make sense of this raw data. Text cleansing is loosely used for most of the cleaning to be done on text, depending on the data source, parsing performance, external noise and so on.   

Since there is no existing package to effectively remove stop words from a sentence, we have to design specific patterns to clean our data for our task. For example, a single word 'a' is normally treated as a stop word in NLTK's stop word's list, but in our movie reviews, people usually give comments like this:   
>'I would give this movie a big A'   

The lower case 'a' can be removed because it's trivial in sentiment analysis, but the upper case 'A' is really important in our sentiment analysis task. Besides, the encoding format of characters in movie reviews is inconsistent. It's often to see ‘ instead of ' in the data sample. 

Therefore, we designed multiple regular expressions to extract useful information and effectively remove irrelevant stop words.  

#### Feature Selection
Feature selection plays a really important role in machine learning and natural language processing, which serves two main purposes. First, it makes training and applying a classifier more efficient by decreasing the size of the effective vocabulary. Secondly, feature selection often increases classification accuracy by eliminating noise features. In this task, we have tried three feature extraction methods: *bag of words*, *TF-IDF* and *Word Embedding*. We will introduce these three methods here briefly.  
> **Bag of Words**  
> A bag of words model is the most straightforward way to represent text in NLP. In this model, a text (such as a sentence or a document) is represented as the bag (multiset) of its words, disregarding grammar and even word order but keeping multiplicity.    
       
> **TF-IDF**  
> TF-IDF stands for term frequency-inverse document frequency, and the TF-IDF weight is a weight often used in information retrieval and text mining.    
> TF-IDF is made up with two parts: *term frequency* and *inverse document frequency*. The number of times a term occurs in a document is called *term frequency* (TF), and the *inverse document frequency factor* (IDF) is incorporated which diminishes the weight of terms that occur very frequently in the document set and increases the weight of terms that occur rarely.    

> **Word Embedding**  
> *Word Embedding* is currently one of the most popular representations of document vocabulary. A Word Embedding format generally tries to map a word using a dictionary to a vector. [Word2vec](https://en.wikipedia.org/wiki/Word2vec) is a great example of using Word Embedding, which is a two-layer neural net that processes text. Its input is a text corpus and its output is a set of vectors: feature vectors for words in that corpus. While Word2vec is not a deep neural network, it turns text into a numerical form that deep nets can understand.   
> In this task, I mainly used pre-trained [word vector](https://nlp.stanford.edu/projects/glove/) (6B tokens, 400K vocab) from Stanford GloVe, which is mainly trained on English Gigaword and Wikipedia 2014.    

## Models
### Baseline Model
For the baseline model, we chose Naive Bayes Model as we used in the previous assignment. In addition, we also used SVM (with RBF kernel) for the sentiment classification task. The comparison of models will be introduced in the **Statistics** section. 

### Machine Learning Models
#### Random Forest Resemble Model   
xxx  
#### k Nearest Neighbor
xxx  
#### Logistic Regression  
xxx  

### Deep Learning Model
For deep learning model, I used TextCNN as the primary model. In the area of text classification, RNN has its natural advantage because the human text has a sequential and consistent feature. However, Yoon Kim (2014) proposed that CNN can also be used in the text classification. The following graph demonstrates the structure of the CNN model:       

[![F3W8sI.jpg](https://s1.ax1x.com/2018/12/08/F3W8sI.jpg)](https://imgchr.com/i/F3W8sI)

Suppose we have a sentence waiting to be classified, and the sentence is made up with multiple $n$-dimensional word vectors. Suppose the length of the sentence is $m$, which implies that the input is a $m*n$ matrix. For this sentiment analysis task, we conduct a convolution on an input matrix. For text data, instead of horizontally sliding our filter, we only move our filter vertically, which is similar to N-Gram extracting the local correlations between word and word. I have used three strides (3, 4, 5), and we have two filters for each stride. Finally, we can get 6 vectors after the convolution process. We apply max-pooling for each vector and concatenate them into a final vector.    

After getting the feature vector from the Text-CNN model, we add a RNN model in order to better capture the sequential features of the human text. We have tried different RNN models, including traditional RNN, Recurrent Neural Network(GRU) and Long Short-term Memory(LSTM), and we finally chose GRU because it has less parameters and training cost. The following is the final CRNN model.   
![22](https://s1.ax1x.com/2018/12/08/F3WFz9.png)

## Statistics

## References
Kim, Yoon. "Convolutional neural networks for sentence classification." *arXiv preprint arXiv:1408.5882 (2014)*.   

Zhang, Ye, and Byron Wallace. "A sensitivity analysis of (and practitioners' guide to) convolutional neural networks for sentence classification." *arXiv preprint arXiv:1510.03820 (2015)*.    

Ma, Xuezhe, and Eduard Hovy. "End-to-end sequence labeling via bi-directional lstm-cnns-crf." *arXiv preprint arXiv:1603.01354 (2016)*.    

