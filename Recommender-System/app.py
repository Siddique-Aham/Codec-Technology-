import streamlit as st
import pandas as pd
from recommender_model import HybridRecommender
from preprocess import Preprocessor
import plotly.express as px
import plotly.graph_objects as go

def load_data():
    """Load and preprocess the dataset."""
    data = pd.read_csv('amazon.csv')
    data = process(data)
    return data

def process(data):
    preprocessor = Preprocessor()
    data = preprocessor.split_category(data)
    data = preprocessor.drop_columns(data)
    data = preprocessor.split_user(data)
    data = preprocessor.cleaner(data)
    data = preprocessor.add_column(data)
    data = preprocessor.encode(data)
    return data

def initialize_recommender(data):
    """Initialize and train the recommendation system."""
    recommender = HybridRecommender(data, 0.5, 0.5)
    recommender.load_cf_model()
    recommender.load_similarity_matrix()
    return recommender

def create_rating_distribution(recommendations):
    """Create a bar chart of ratings distribution."""
    recommendations['product_name'] = recommendations['product_name'].apply(
        lambda x: x[:10] + '...' if len(x) > 10 else x
    )

    fig = px.bar(
        recommendations,
        x='product_name',
        y='rating',
        title='Product Ratings',
        labels={'product_name': 'Product', 'rating': 'Rating'},
        color='rating',
        color_continuous_scale='viridis'
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        showlegend=False,
        height=400
    )
    return fig

def main():
    st.set_page_config(page_title="Product Recommendation Dashboard", layout="wide")

    # Add header
    st.title("🛍️ Product Recommendation Dashboard")
    st.markdown("""
    This dashboard provides personalized product recommendations based on user behavior and preferences.
    Enter a user ID to get started!
    """)

    # Load data and initialize recommender
    try:
        with st.spinner('Loading data and initializing recommendation system...'):
            data = load_data()
            recommender = initialize_recommender(data)

        # Create sidebar for user input
        st.sidebar.header("User Input")

        # Get unique user IDs
        unique_users = sorted(data['user_id'].unique())

        # Create user number slider
        user_number = st.sidebar.slider(
            "Select User Number",
            min_value=1,
            max_value=min(1000, len(unique_users)),
            value=1,
            help="Slide to select a user number"
        )

        # Map slider value to actual user ID
        selected_user_id = unique_users[user_number - 1]

        # Display selected user ID
        st.sidebar.info(f"Selected User ID: {selected_user_id}")

        # Get recommendations when user ID is selected
        if selected_user_id:
            # Get user's past purchases
            user_history = data[data['user_id'] == selected_user_id]

            if not user_history.empty:
                # Get a sample product ID from user's history
                sample_product_id = user_history['product_id'].iloc[0]

                cosine_similarity = recommender.cosine_similarity
                svd_model = recommender.svd_model

                # Get recommendations
                recommendations = recommender.combined_recommendations(
                    user_id=selected_user_id,
                    product_id=sample_product_id,
                    data = data,
                    cosine_sim = cosine_similarity,
                    svd_model = svd_model,
                )[0]

                if not recommendations.empty:
                    # Display recommendations in two columns
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("📊 Recommended Products")
                        for _, product in recommendations.iterrows():
                            with st.expander(f"{product['product_name']}"):
                                st.write(f"**Category:** {product['category']}")
                                st.write(f"**Rating:** ({product['rating']})")
                                if 'about_product' in product:
                                    st.write(f"**About:** {product['about_product']}")

                    with col2:
                        st.subheader("📈 Analytics")

                        # Rating distribution
                        st.plotly_chart(create_rating_distribution(recommendations))
                else:
                    st.warning("No recommendations found matching the current filters.")
            else:
                st.warning("No purchase history found for this user.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()