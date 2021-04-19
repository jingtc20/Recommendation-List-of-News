# Recommendation Lists of News
make classification of 509,236 news using KMeans and build the recommendation list with the most votes in each category

***
## Results
- The classification of the news with n_clusters = 6:
  <img src='pic/classification plot.png' width='400'/>

- The recommendation list with the most votes in each category:
  <img src='pic/recommendation plot.png' width='400'/>

## Main Steps 
- Run `Data_Analysis.ipynb` to analyze the data
  * Library used: **pandas, numpy, nltk, sklearn**
  * The shape of the data is (509236, 8)
  * Add a new column called `cleaned` to store the processed title
     - convert title to all lowercase letters
     - remove the punctuation and digits
     - remove stop words and do lemmatization
  * Plot two word clouds to show the most frequently used words in titles under 18 and above 18.
  * Download the new data called `data.csv`
  
- Run `optimal_k.py` to find the optimal k in KMeans clustering based on a small dataset.  
  * Library used: **pandas, numpy, gensim, sklearn**
  * It takes a long time to calculate silhouette score in a big dataset, so I find the optimal k based on the first 10,000 rows of the data. For the silhouette score, the best value is 1 and the worst value is -1
  * The result shows that the best interval for k is [4, 10], and the silhouette score is in the [0.471, 0.517]
  * The optimal k is set as 6 considering the dataset is big 
  
- Run `result.py` to make classification and build the recommendation list
  * Library used: **pandas, numpy, gensim, sklearn, matplotlib**
  * Calculate the TFIDF of each word in the cleaned title, and remove unimportant words if its TFIDF is lower than specfic value (0.5). Convert each remaining word to a 100 dimensional vector using word embedding, take an average of all the vectors, and add a new column called `Word2Vec` to store it 
  * Make the classification using KMeans and df['Word2Vec']
  * Reduce the demension of word2vec to 2D array using PCA, and plot the figure
   
