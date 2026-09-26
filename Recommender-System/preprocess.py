import re
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder

class Preprocessor:
    def split_category(self, data):
        data['main_category'] = data['category'].str.split('|').str[0]
        data['subcategory'] = data['category'].str.split('|').str[-1]
        return data

    def clean_text(self, text):
        text = text.lower()
        text = text.replace('&', ' and ')
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        pattern = r'http\S*?\.(com|in)'
        text = re.sub(pattern, '', text)
        stop_words = set(stopwords.words('english'))
        text = ' '.join([word for word in text.split() if word not in stop_words])
        return text

    def drop_columns(self, data):
        columns_to_drop = ['discounted_price', 'actual_price', 'discount_percentage', 'user_name', 'review_id']
        data = data.drop([col for col in columns_to_drop if col in data.columns], axis=1)
        return data

    def split_user(self, data):
        data['user_id'] = data['user_id'].str.split(',')
        data['review_title'] = data['review_title'].str.split(',')

        data['user_id'], data['review_title'] = zip(*data.apply(
            lambda row: (row['user_id'], row['review_title'] + [''] * (len(row['user_id']) - len(row['review_title'])))
            if len(row['user_id']) > len(row['review_title'])
            else (row['user_id'] + [''] * (len(row['review_title']) - len(row['user_id'])), row['review_title']),
            axis=1
        ))

        data = data.explode(['user_id', 'review_title'], ignore_index=True)
        return data

    def add_column(self, data):
        data['product_details'] = (
            data['main_category'].fillna('') + ' ' +
            data['subcategory'].fillna('') + ' ' +
            data['product_name'].fillna('') + ' ' +
            data['about_product'].fillna('') + ' ' +
            data['review_title'].fillna('') + ' ' +
            data['review_content'].fillna('')
        )
        return data

    def cleaner(self, data):
        for column in ['product_name', 'about_product', 'review_title', 'review_content', 'category', 'main_category', 'subcategory']:
            if column in data.columns:
                data[column] = data[column].apply(self.clean_text)
        return data

    def encode(self, data):
        le_user = LabelEncoder()
        le_product = LabelEncoder()
        if 'user_id' in data.columns:
            data['user_id'] = le_user.fit_transform(data['user_id'])
        if 'product_id' in data.columns:
            data['product_id'] = le_product.fit_transform(data['product_id'])
        return data
