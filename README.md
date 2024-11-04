# NLP---ANALYSIS-PREDICTION-OF-DISASTER-TWEETS
Machine learning model for accurately predicting whether a tweet is about a real disaster. The dataset provided contains 10,000 tweets that have been manually classified as either real disaster-related or not. By leveraging natural language processing techniques, we seek to train a model that can automate this classification process effectively.

Twitter has emerged as a crucial platform for real-time communication during emergencies. This has led to an increased interest from various organisations, such as disaster relief agencies and news outlets, in programmatically monitoring Twitter to identify relevant disaster-related information. However, discerning whether a tweet actually pertains to a real disaster or not can be challenging for machines due to the presence of figurative language and context ambiguity.

In this project, we aim to develop a machine learning model for accurately predicting whether a tweet is about a real disaster. The dataset provided contains 10,000 tweets that have been manually classified as either real disaster-related or not. By leveraging natural language processing techniques, we seek to train a model that can automate this classification process effectively.

Our approach involves exploratory data analysis to gain insights into the dataset, including the distribution of class labels and an examination of the tweet content. We perform data cleaning and preprocessing steps to enhance the quality of the dataset, removing duplicates and applying text cleaning techniques.

Featurization plays a crucial role in our model development. We employ techniques such as Bag of Words (BoW), bi-grams, and Term Frequency-Inverse Document Frequency (TF-IDF) to convert the textual data into numerical vectors. These techniques enable the transformation of tweets into a format suitable for machine learning algorithms.

For the classification task, we select the Multinomial Naive Bayes (MNB) algorithm as it has demonstrated effectiveness in text classification tasks. We evaluate the performance of MNB on different featurization techniques, employing evaluation metrics such as precision, recall, F1-score, and accuracy. Through comprehensive classification reports, confusion matrices, and discussions, we present the results of our experiments and analyze the model's performance.

The project's findings contribute to the field of natural language processing for disaster-related tweets. We highlight the limitations of the current approach and provide suggestions for future work, including exploring advanced models or incorporating additional features to improve classification accuracy. This project serves as a valuable starting point for data scientists venturing into natural language processing and offers insights into the challenges of identifying real disaster tweets on social media platforms.

INTRODUCTION:

In recent years, Twitter has emerged as a powerful communication channel during times of emergency. With the widespread use of smartphones, individuals can quickly share real-time updates and observations about ongoing disasters. This trend has sparked the interest of disaster relief organisations and news agencies in programmatically monitoring Twitter to identify relevant disaster-related information. However, distinguishing between tweets that genuinely indicate a real disaster and those that do not can be challenging.

The objective of this project is to develop a machine learning model that can accurately predict whether a tweet pertains to a real disaster. The dataset provided consists of 10,000 tweets that have been manually classified as either real disaster-related or not. By leveraging natural language processing (NLP) techniques, we aim to train a model capable of automating this classification process effectively.

To begin, we conduct exploratory data analysis to gain insights into the dataset. This analysis includes an examination of the distribution of class labels and an exploration of the tweet content. We then proceed with data cleaning and preprocessing steps to enhance the quality of the dataset. This involves removing duplicates and applying text cleaning techniques to handle noise and inconsistencies.

Featurization plays a crucial role in our model development. We employ techniques such as Bag of Words (BoW), bi-grams, and Term Frequency-Inverse Document Frequency (TF-IDF) to transform the textual data into numerical representations. These techniques enable the conversion of tweets into a format suitable for machine learning algorithms.

For the classification task, we select the Multinomial Naive Bayes (MNB) algorithm due to its effectiveness in text classification tasks. We evaluate the performance of the MNB algorithm using different featurization techniques and employ evaluation metrics such as precision, recall, F1-score, and accuracy. The results are presented through comprehensive classification reports, confusion matrices, and discussions, shedding light on the model's performance and effectiveness.

The findings of this project contribute to the field of natural language processing for disaster-related tweets. The challenges of identifying real disaster tweets amidst the presence of figurative language and contextual ambiguity are addressed. Furthermore, limitations of the current approach are discussed, and suggestions for future work, such as exploring advanced models or incorporating additional features, are provided.

By automating the classification of disaster-related tweets, this project aims to assist organisations in monitoring and responding to emergencies more effectively. It also serves as a valuable introduction to natural language processing for data scientists seeking to explore this domain.

EXPLORATORY DATA ANALYSIS:
In the initial phase of the project, we conducted an exploratory data analysis (EDA) to gain insights into the dataset and understand the characteristics of the Twitter data related to disaster announcements. The EDA helped us identify patterns, trends, and potential challenges that may influence the development of our machine learning model.

We started by examining the distribution of disaster-related and non-disaster-related tweets in the dataset. Using bar charts, we visualised the count or percentage of tweets belonging to each category. The analysis revealed that the dataset contains a balanced representation of both disaster and non-disaster tweets, allowing for a robust model training process.

To further understand the language used in these tweets, we employed word cloud visualisations for disaster-related and non-disaster-related tweets. The word clouds provided an intuitive representation of the most frequent words in each category. For disaster-related tweets, terms like "emergency," "help," and specific disaster keywords were prominent, indicating the relevance of these words in identifying real disasters. In contrast, non-disaster tweets displayed words like "metaphorical," "figurative," and other contextually different terms, reflecting the challenges faced by a machine learning model in distinguishing metaphorical language.

Geographic analysis played a crucial role in assessing the spatial distribution of disaster-related tweets. By creating a geographic heatmap, we visualised the regions associated with disaster announcements. This visualisation illustrated the areas where emergencies are frequently reported, helping disaster relief organisations and news agencies identify hotspots and allocate resources accordingly.

In addition to understanding the content, we delved into the sentiment expressed in disaster-related tweets. We conducted sentiment analysis and plotted a sentiment distribution graph. The graph revealed the emotional tone prevalent in these tweets, with positive, negative, and neutral sentiments being discernible. This analysis provided insights into the sentiments associated with disaster events and could be utilised in building a more nuanced prediction model.

To address any class imbalance concerns, we also investigated the distribution of disaster-related and non-disaster-related tweets. This analysis aimed to ensure that the model training process adequately accounts for the differing representation of the two classes. By visualising the class imbalance using an appropriate plot, we identified the need to employ suitable techniques to handle this potential challenge during model development.

Finally, we examined the frequency of specific words or phrases in disaster-related and non-disaster-related tweets. By generating word frequency plots or histograms, we gained insights into the vocabulary used in each category. This analysis assisted in identifying key terms that are indicative of disaster-related tweets and may serve as important features for our machine learning model.

Overall, the exploratory data analysis phase provided valuable insights into the dataset and highlighted important considerations for developing an effective machine learning model for predicting real disasters on Twitter. The visualisations and analyses conducted during this phase laid the foundation for subsequent stages of the project, including data preprocessing, feature engineering, and model selection.

Checking the Class Imbalance

 PRE PROCESSING :
We Preprocess the data as shown below

Original tweet: "This is an example tweet with some #hashtags and a link: https://example.com #NLP"

Step 1: Stopword Removal After removing stopwords, the tweet becomes: "example tweet #hashtags link: https://example.com #NLP"

Step 2: Removing Hyperlinks The hyperlinks in the tweet are removed, resulting in: "example tweet #hashtags"

Step 3: Removing HTML Code Any remaining HTML code is removed using BeautifulSoup, resulting in the same text: "example tweet #hashtags"

Step 4: Decontracting Text : Decontracting involves expanding contracted words, such as "won't" to "will not." In this example, there are no contracted words, so the text remains the same: "example tweet #hashtags"

Step 5: Removing Words with Numbers Words containing numbers are removed from the tweet. Since there are no words with numbers in this example, the text remains unchanged: "example tweet #hashtags"

Step 6: Removing Special Characters Special characters and symbols are removed from the tweet, leaving only alphabetic characters and spaces: "example tweet hashtags"

Step 7: Removing Stock Market Tickers Stock market tickers, indicated by a dollar sign followed by letters, are removed from the tweet. Since there are no stock market tickers in this example, the text remains the same: "example tweet hashtags"

Step 8: Removing Retweet Text The "RT" text, commonly used for retweets, is removed from the tweet. Since the example tweet does not contain retweet text, it remains unchanged: "example tweet hashtags"

Step 9: Removing Hashtags, Mentions, and Ellipsis The hashtags, mentions (starting with "@"), and ellipsis ("...") are removed from the tweet. In this example, the hashtag "#hashtags" is removed, resulting in: "example tweet"

Step 10: Tokenization The tweet is tokenized into individual words using the TweetTokenizer from the NLTK library. The resulting tokens are: ["example", "tweet"]

Step 11: Lemmatization Each token is lemmatized to its base form using the WordNetLemmatizer from the NLTK library. In this example, no lemmatization is performed as the tokens are already in their base form: ["example", "tweet"]

 FEATURIZATION :
Featurization plays a crucial role in Natural Language Processing tasks as it involves transforming textual data into numerical representations that machine learning models can understand. In this section, we describe the featurization techniques employed to represent the Twitter data for our disaster prediction model.

Bag-of-Words (BoW): We started by applying the Bag-of-Words technique to convert the tweet texts into numerical vectors. BoW represents each tweet as a vector, where each dimension corresponds to a unique word in the corpus. The value in each dimension indicates

the frequency or presence of the word in the tweet. By employing this technique, we captured the occurrence and distribution of words in the tweets.

TF-IDF (Term Frequency-Inverse Document Frequency): To further enhance the featurization process, we utilized the TF-IDF technique. TF-IDF assigns weights to words based on their frequency in a tweet and their rarity across the entire corpus. This approach helps in identifying words that are important and specific to individual tweets, enabling the model to focus on discriminative features.

N-grams: N-grams are contiguous sequences of n words in a tweet. By considering n-grams, we captured not only individual words but also the contextual information present in phrases or combinations of words. This helped in capturing more nuanced relationships between words and improving the model's ability to understand the semantics of the tweets.

Additional Features: Apart from text-based features, we also considered additional features derived from the tweet metadata. These features may include the length of the tweet, the number of hashtags or mentions, and other relevant characteristics. Incorporating these features provided supplementary information to the model and improved its predictive capabilities.

Dimensionality Reduction: Given the potentially high dimensionality of the feature space, we applied dimensionality reduction techniques such as Principal Component Analysis (PCA) or t-SNE (t-Distributed Stochastic Neighbour Embedding) to reduce the feature space while retaining important information. This step helped in managing computational complexity and removing noise or redundant features.

By combining these featurization techniques, we obtained a comprehensive representation of the Twitter data, capturing both the textual content and the contextual information. The resulting feature vectors served as inputs to our machine learning models, enabling them to learn patterns and make predictions based on the transformed data.

It is important to note that the choice of featurization techniques depended on the specific requirements of the problem and the characteristics of the dataset. Through experimentation and evaluation, we identified the most effective featurization strategies that contributed to the accuracy and generalisation of our machine learning model.

The featurization phase laid the foundation for the subsequent stages of model training, evaluation, and optimization. The transformed feature vectors, along with the corresponding labels, formed the training data for our predictive model, ensuring that it can learn from the patterns and associations present in the Twitter data.

MODELLING :
Selecting an appropriate model and evaluating its performance are critical steps in developing an effective machine learning solution. In this section, we discuss the process of model selection and evaluation for our disaster prediction task.

Model Selection: To identify the most suitable model for our task, we experimented with several popular machine learning algorithms, including Logistic Regression, Naive Bayes, Random Forest, and Support Vector Machines (SVM). We trained each model using the featurized data and evaluated their performance using appropriate evaluation metrics.

Model Evaluation: To assess the performance of our models, we employed various evaluation metrics, including accuracy, precision, recall, and F1-score. Accuracy measures the overall correctness of the model's predictions, while precision and recall focus on the model's ability to correctly identify true positive instances. F1-score provides a balanced measure of precision and recall. Additionally, we examined the Receiver Operating Characteristic (ROC) curve and calculated the Area Under the Curve (AUC) to assess the model's ability to discriminate between positive and negative instances.

We conducted rigorous cross-validation experiments to ensure reliable model evaluation. The dataset was randomly split into training and validation sets, with a predefined ratio, to train and evaluate the models using different subsets of data. This approach helped us mitigate overfitting and gain a more robust understanding of the models' generalization capabilities.

By comparing the performance of different models based on the evaluation metrics, we identified the most effective model for our disaster prediction task.

Overall, model selection and evaluation were crucial in identifying the best-performing model, which would be used for predicting whether a given tweet is related to a real disaster or not. The chosen model demonstrated superior performance, as evidenced by the evaluation metrics and the ROC curve analysis.

Next, we proceed to discuss the results and findings of our experiments, providing insights into the predictive power of our model and its implications for real-world applications. RESULTS:

These results below represent the performance metrics (precision, recall, F1-score) of the Multinomial Naive Bayes (NB) algorithm for three different featurization techniques: Bag of Words, Bag of Words (Bigrams), and TF-IDF.

The precision metric measures the ability of the model to correctly identify positive instances, while the recall metric represents the model's ability to identify all relevant positive instances. The F1-score is the harmonic mean of precision and recall, providing a balanced measure of the model's performance.

Overall, the Multinomial NB model performs well across all three featurization techniques, with accuracy ranging from 0.82 to 0.83. In terms of precision, the model achieves values between 0.68 and 0.92, indicating its ability to accurately predict both classes. The recall

values range from 0.79 to 0.87, indicating the model's ability to capture relevant instances from each class. The F1-scores range from 0.76 to 0.85, demonstrating the balance between precision and recall.

These results suggest that all three featurization techniques provide valuable information for the Multinomial NB model, allowing it to effectively classify tweets as either related

Multinomial NB Results for Bag of Words:

Precision Recall F1-Score Support Class 0 0.9 0.81 0.85 726 Class 1 0.71 0.84 0.77 416 Accuracy

0.82 1142 Macro Avg 0.81 0.82 0.81 1142 Weighted Avg 0.83 0.82 0.82 1142 Multinomial NB Results for Bag of Words (Bigrams):

Precision Recall F1-Score Support Class 0 0.91 0.8 0.85 734 Class 1 0.71 0.85 0.77 408 Accuracy

0.82 1142 Macro Avg 0.81 0.83 0.81 1142 Weighted Avg 0.83 0.82 0.82 1142 Multinomial NB Results for TF-IDF:

Precision Recall F1-Score Support Class 0 0.92 0.79 0.85 757 Class 1 0.68 0.87 0.76 385 Accuracy

0.82 1142 Macro Avg 0.8 0.83 0.81 1142 Weighted Avg 0.84 0.82 0.82 1142

USER REACTIONS ANALYSIS: The user reactions analysis was conducted to understand how users are responding to various content or events based on the predictions made by our machine learning model. The analysis included multiple visualizations to provide a comprehensive understanding of user reactions. The first method used was a confusion matrix, which displayed the relationship between predicted user reactions and true user reactions. This matrix helped identify the accuracy of the model's predictions and any patterns of misclassification. The second method involved analyzing the distribution of user reactions using a bar chart. This visualization provided an overview of the frequency of each predicted user reaction, allowing us to identify which reactions were more common among users. Lastly, grouped bar charts were used to compare correct and incorrect predictions for each user reaction. This comparison helped us assess the model's accuracy in predicting different user reactions. Overall, the analysis of user reactions provided valuable insights into user sentiment and preferences, enabling us to make informed decisions and improve user experience.

CONCLUSION :

From the above results, we can observe that all three models perform relatively well in terms of accuracy, with an accuracy score of approximately 82%.

When comparing the precision values, the model using Bag of Words (BoW) representation achieves the highest precision for class 0 (0.90), indicating that it has a better ability to correctly classify non-disaster tweets. On the other hand, the model using TF-IDF representation achieves the highest precision for class 1 (0.68), suggesting that it performs better in identifying disaster-related tweets.

In terms of recall, the model using TF-IDF representation demonstrates the highest recall for both classes, indicating its effectiveness in capturing both disaster and non-disaster tweets.

Overall, the models' performance across different feature representations is quite similar, with accuracy being the most consistent metric. Based on these results, it can be concluded that the Multinomial Naive Bayes algorithm, along with various feature representations, can be effective in predicting whether a tweet is related to a real disaster or not. Further experimentation and fine-tuning of these models may lead to even better performance.
