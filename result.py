import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
import time



# parameters setting
RECOM_NUM = 5    # the number of recommended titles in each label
MIN_K = 5        # min k of k_means cluster
MAX_K = 15       # max k of k_means cluster
MIN_TFIDF = 0.5  # remove the word in title if its tfidf is lower than this value
OPTIMAL_K = 7    # the optimal k is decided by the highest silhouette_score


df = pd.read_csv('data.csv')
# df = pd.read_csv('data.csv', nrows = 10000)
df['cleaned'] = df['cleaned'].fillna("")
print("Shape of the data: ", df.shape)


def get_word2vec_model(corpus):
    model = Word2Vec(corpus, size=100, min_count=1, workers=4, sg = 1, seed = 1, window = 5)
    model.save("word_2_vec.model")


def word_to_vec(model, cleaned_title, dict_idf, vocabulary):
    if cleaned_title == "":
        return [0] * 100
    n = len(cleaned_title.split())
    words = [word for word in cleaned_title.split() if word in dict_idf]
    l = [np.asarray(model.wv[w]) for w in words if words.count(w)*dict_idf[w]/n > MIN_TFIDF]
    return np.average(np.array(l), 0).tolist() if l != [] else [0] * 100
    

def choose_optimal_k(min_k, max_k, title_vec):
    cluster_scores = []
    for k in range(min_k, max_k):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(title_vec)
        lables = kmeans.labels_
        ilhouette_avg = silhouette_score(title_vec, lables)
        cluster_scores.append(round(ilhouette_avg, 3))
    optimal_k = cluster_scores.index(max(cluster_scores)) + min_k
    print("silhouette_score for each k is: ", cluster_scores)
    print("optimal k for kmeans is: ", optimal_k)
    print("silhouette_score for optimal k is: ", max(cluster_scores))
    return optimal_k
    

def main():
    start_time = time.time()
    # calculate the tfidf 
    cleaned_titles = df['cleaned'].tolist()
    tfidf_vectorizer = TfidfVectorizer()
    tfidf = tfidf_vectorizer.fit_transform(df['cleaned'])
    df_idf = pd.DataFrame(tfidf_vectorizer.idf_, index=tfidf_vectorizer.get_feature_names(),columns=["idf_weights"]) 
    vocabulary = set(tfidf_vectorizer.get_feature_names())
    dict_idf = dict(zip(vocabulary, tfidf_vectorizer.idf_.tolist()))
    # print(len(vocabulary))   # 97472
    
    # convert cleaned title to vector by word2vec and tfidf
    corpus = [title.split() for title in df['cleaned']]
    get_word2vec_model(corpus)
    model = Word2Vec.load("word_2_vec.model")
    df['Word2Vec'] = df['cleaned'].apply(lambda x: word_to_vec(model, x, dict_idf, vocabulary))
    # print("Word2Vec run time: ", round(time.time() - start_time, 2))

    title_vec = df['Word2Vec'].tolist()
    optimal_k = OPTIMAL_K
    kmeans = KMeans(n_clusters=optimal_k, random_state=0, max_iter=300, tol=1e-4).fit(title_vec)
    lables = kmeans.labels_
    df['lables'] = lables
    # print("k-means run time: ", round(time.time() - start_time, 2))

    recommendations = {}  
    recom_num = RECOM_NUM
    for lable in range(optimal_k):
        df_label = df[(df['lables'] == lable) & (df['over_18'] == False)]
        df_lable_sort_by_update = df_label.sort_values(by = 'up_votes', ascending=False)
        recommendations[lable] = df_lable_sort_by_update[0:recom_num]['title'].tolist()
    np.save('recom.npy', recommendations) 

    # reduce the demension of word2vec to 2D array
    pca = PCA(n_components=2)
    title_array = np.array(title_vec)
    dim_reduced_title_array = pca.fit_transform(title_array)
    norm_array = (320 * (dim_reduced_title_array - np.min(dim_reduced_title_array)) / np.ptp(dim_reduced_title_array)).astype(int)
    centers = kmeans.cluster_centers_
    dim_reduced_centers = pca.fit_transform(centers)
    norm_centers = (320 * (dim_reduced_centers - np.min(dim_reduced_centers)) / np.ptp(dim_reduced_centers)).astype(int)

    # Draw the figure 1
    plt.figure(1)
    plt.scatter(norm_array[:, 0], norm_array[:, 1], c=lables, s=50)
    plt.scatter(norm_centers[:, 0], norm_centers[:, 1], marker='o', c="white", alpha=1, s=200, edgecolor='k')
    for i, c in enumerate(norm_centers):
        plt.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,s=50, edgecolor='k')
    plt.title("The classification of the news with n_clusters = %d" % optimal_k)
    plt.savefig('classification plot')

    # Draw the figure 2
    plt.figure(2)
    y_lower = 10
    for lable in range(optimal_k):
        y_upper = y_lower + 25 * recom_num
        color = cm.nipy_spectral(float(lable) / optimal_k)
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, recom_num,
                          facecolor=color, edgecolor=color, alpha=0.7)
        plt.text(-0.1, y_lower + 0.5 * lable, lable)
        for i in range(recom_num):
            if i < len(recommendations[lable]):
                plt.text(0.05, y_upper - 5 - 22 * (i + 1), recommendations[lable][i])
        y_lower = y_upper + 20 
    plt.title("Top %d news with the most votes in each category" % recom_num, fontsize=12, fontweight = 'bold')
    plt.xlabel("News recommendation")
    plt.ylabel("Cluster label")
    plt.yticks([])
    plt.xticks([])
    plt.savefig('recommendation plot')
    end_time = time.time()
    print("Total time is: ", round(end_time - start_time, 2))
    # plt.show()

if __name__ == '__main__':
    main()
