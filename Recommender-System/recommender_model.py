import surprise
import sklearn
import numpy as np
import scipy as sp
import joblib
from scipy.sparse import load_npz
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from surprise import SVD
from surprise import Dataset, Reader, accuracy
from surprise.model_selection import cross_validate, train_test_split


class HybridRecommender:
    def __init__(self, data, content_weight, cf_weight):
        self.data = data
        self.user_item_matrix = None
        self.cosine_similarity = None
        self.svd_model = None
        self.content_weight = content_weight
        self.cf_weight = cf_weight

    def load_cf_model(self):
        self.svd_model = joblib.load('cfmodel.pkl')

    def load_similarity_matrix(self):
        cosine_sim_sparse = load_npz('cosine_sim_sparse.npz')
        self.cosine_similarity = cosine_sim_sparse.toarray()

    def content_based_recommendations(self, product_id, data, cosine_sim, n=5):
        # Filter to only rows matching the product_id
        product_row = data[data['product_id'] == product_id]
        if product_row.empty:
            raise ValueError(f"Product ID '{product_id}' not found in the dataset.")

        # Get the index for the first occurrence of the given product_id
        idx = product_row.index[0]

        # Calculate similarity scores for the product
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Exclude the product itself and ensure unique product recommendations
        unique_product_ids = set()
        unique_sim_scores = []
        for i, score in sim_scores:
            product_i = data.iloc[i]['product_id']
            if product_i != product_id and product_i not in unique_product_ids:
                unique_product_ids.add(product_i)
                unique_sim_scores.append((i, score))
            if len(unique_sim_scores) >= n:
                break

        # Get the top-n unique recommendations
        product_indices = [i[0] for i in unique_sim_scores]

        # Return only the specified columns for each unique product
        recommended_data = data.iloc[product_indices].drop_duplicates(subset='product_id')
        recommended_data = recommended_data[['product_id', 'product_name', 'category', 'rating', 'about_product']]

        return recommended_data, unique_sim_scores

    def combined_recommendations(self, user_id, product_id, data, cosine_sim, svd_model, content_weight=0.5,
                                 collaborative_weight=0.5, n=5):
        # Generate a combined recommendation using both content-based and collaborative filtering.

        # Get content-based recommendations
        content_based_recommendations_data, p_sim_score = self.content_based_recommendations(product_id, data, cosine_sim, n)

        # Get product_ids from content-based recommendations
        recommended_product_ids = content_based_recommendations_data['product_id'].tolist()

        # Get collaborative filtering predictions for recommended products
        collaborative_predictions = []
        for product in recommended_product_ids:
            # Predict the rating for the user-product pair using the trained SVD model
            prediction = svd_model.predict(user_id, product)
            collaborative_predictions.append((product, prediction.est))

        # Merge content-based and collaborative predictions
        combined_scores = []
        for content_product, content_score in zip(recommended_product_ids, p_sim_score):
            # Find the corresponding collaborative score
            collaborative_score = next((score for pid, score in collaborative_predictions if pid == content_product), 0)

            # Combine the scores using the weights
            combined_score = (content_weight * content_score[1]) + (collaborative_weight * collaborative_score)
            combined_scores.append((content_product, combined_score))

        # Sort the combined scores by the weighted score
        sorted_combined_scores = sorted(combined_scores, key=lambda x: x[1], reverse=True)

        # Get the top `n` recommendations
        top_recommendations = sorted_combined_scores[:n]

        # Retrieve product details for the top recommendations and remove duplicates based on product_id
        top_recommendation_ids = [product_id for product_id, _ in top_recommendations]
        final_recommendations = data[data['product_id'].isin(top_recommendation_ids)].drop_duplicates(
            subset='product_id')

        return final_recommendations[
            ['product_id', 'product_name', 'category', 'rating', 'about_product']], top_recommendations