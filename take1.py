import streamlit as st
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import edgedb
import time
import ast
import warnings
warnings.filterwarnings('ignore')
import os
import functools
import tiktoken

import openai
from openai.embeddings_utils import cosine_similarity, distances_from_embeddings, indices_of_nearest_neighbors_from_distances
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
openai.api_key = os.environ["OPENAI_API_KEY"]


import sqlite3
conn = sqlite3.connect('ecommerce.db')
client = conn.cursor()

def get_reviews_for_product(product_id):
    client.execute(
        '''
        SELECT reviews.*
        FROM reviews 
        JOIN products
        ON reviews.asin = products.asin
        WHERE products.id = ?
        ''',
        (product_id,)
    )
    review_set = client.fetchall()

    # Get column names from cursor description
    columns = [column[0] for column in client.description]
    print(product_id)
    print(columns)

    reviews_df = pd.DataFrame(review_set, columns=columns)
    reviews_df['embedding'] = reviews_df['embedding'].apply(lambda x: ast.literal_eval(x))
    return reviews_df

def view(product_id, df):
    '''View product details'''
    image_column, info_column = st.columns(2)
    product_series = df.loc[df.id == product_id, :].squeeze()
    image_column.image(f'./assets_resized/{product_series["asin"]}.jpg', use_column_width='always')
    
    print(product_id)
    info_column.write(f'**PRODUCT**: {product_series["title_text"]}')
    info_column.write(f'**CATEGORY**: {product_series["category"]}')
    info_column.write(f'**OS**: {product_series["operating_system"]}')
    info_column.write(f'**BRAND**: {product_series["brand"]}')
    info_column.write(f'**MODEL**: {product_series["model_name"]}')
    info_column.write(f'**SERIES**: {product_series["series"]}')
    info_column.write(f'**RAM**: {product_series["ram"]}')
    info_column.write(f'**STORAGE**: {product_series["storage"]}')
    info_column.write(f'**PRICE**: {product_series["price"]}')
    info_column.write(f'**RATING**: {product_series["rating"]}')
    info_column.write(f'**url**: {product_series["url"]}')
    info_column.write(f'**PRODUCT DESCRIPTION**: {product_series["seller_text"]}')
    if 'query_embedding' in st.session_state:
        reviews = get_reviews_for_product(product_id)
        print(st.session_state)
        reviews['similarities'] = reviews['review_text'].apply(lambda x: cosine_similarity(get_embedding(x, model='text-embedding-ada-002'), st.session_state.query_embedding))

        most_similar_review = reviews.loc[reviews['similarities'].idxmax()]

        info_column.write(f'**Most Similar Review to User Query**: {most_similar_review["review_text"]}')

    del st.session_state['product']
    if st.button('Back'):
        st.runtime.legacy_caching.clear_cache()
        st.session_state.page = 'search'
        st.experimental_rerun()

def set_viewed_product(product, k=7):
    '''Set viewed product'''
    st.session_state.product = product.id
    st.session_state.page = 'view'
    st.experimental_rerun()

def view_products(df, products_per_row=7):
    '''View products'''
    num_rows = min(8, int(np.ceil(len(df) / products_per_row)))
    for i in range(num_rows):
        start = i * products_per_row
        end = start + products_per_row
        products = df.iloc[start:end]
        columns = st.columns(products_per_row)
        for product, column in zip(products.iterrows(), columns):# product is a tuple of (index, row)
            container = column.container()
            button_key = f"view_{product[1]['id']}"
            if container.button('View', key=button_key):
                set_viewed_product(product=product[1], k=products_per_row)
            container.image(f'./assets_resized/{product[1]["asin"]}.jpg', use_column_width='always')

def get_embedding(text, model="text-embedding-ada-002", max_tokens=8000):
    
    text = text.replace("\n", " ")
    encoding = tiktoken.encoding_for_model(model)
    text = encoding.decode(encoding.encode(text)[:max_tokens]) if len(encoding.encode(text)) >= max_tokens else text

    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']

def recommend_products(query_embedding, df, n=1000):
    df['similarities'] = df['combined_embedding'].apply(lambda x: cosine_similarity(x, query_embedding))
    recommended = df.sort_values('similarities', ascending=False).head(n*2)
    recommended = recommended[~recommended['title_text'].str.lower().str.split().str[:2].duplicated(keep='first')]
    return recommended.head(n)

def get_most_similar_review(query_embedding, df, n=1):
    df['similarities'] = df['embedding'].apply(lambda x: cosine_similarity(x, query_embedding))
    return df.sort_values('similarities', ascending=False).head(n)

query_input = st.sidebar.text_input("Enter your query: (optional)", key='name')
st.sidebar.markdown("_Please press Enter after typing to submit the query._", unsafe_allow_html=True)

def update_query_and_sort_results(query_input):
    if query_input:
        st.session_state.query_embedding = get_embedding(query_input, model='text-embedding-ada-002')
        st.session_state.query_for_sorting = query_input
    else:
        if 'query_embedding' in st.session_state:
            del st.session_state.query_embedding
    if 'query_embedding' in st.session_state and 'filtered_products_df' in st.session_state:
        st.session_state.filtered_products_df = recommend_products(st.session_state.query_embedding, st.session_state.filtered_products_df)
        st.session_state.filtered_products_df.sort_values('similarities', ascending=False, inplace=True)
        
if 'page' not in st.session_state:
    st.session_state.page = 'search'



distinct_categories = client.execute('''SELECT DISTINCT category FROM products''').fetchall()
distinct_brands = client.execute('''SELECT DISTINCT brand FROM products''').fetchall()
distinct_operating_systems = client.execute('''SELECT DISTINCT operating_system FROM products''').fetchall()
min_max_price_tuple = client.execute('''SELECT min(price), max(price) FROM products''').fetchone()

# Flat the list of tuples
distinct_categories = [item[0] for item in distinct_categories]
distinct_brands = [item[0] for item in distinct_brands]
distinct_operating_systems = [item[0] for item in distinct_operating_systems]

category_multi_select = st.sidebar.multiselect('Product Category:', 
                                  distinct_categories, 
                                  default=distinct_categories)

brand_multi_select = st.sidebar.multiselect('Computer Brands:', 
                                  distinct_brands, 
                                  default=distinct_brands)

os_multi_select = st.sidebar.multiselect('Operating Systems:', 
                                  distinct_operating_systems, 
                                  default=distinct_operating_systems)

price_slider = st.sidebar.slider(
    'Select a price range',
    min_value=0.0, 
    max_value=6000.00,
    value=(min_max_price_tuple[0], min_max_price_tuple[1])
)


rating_slider = st.sidebar.slider(
    'Select range for ratings (1-5)',
    min_value=1, 
    max_value=5,
    value=(1,5)
)

if st.sidebar.button('Search') or st.session_state.page == 'search' or ('product' not in st.session_state):
    with st.spinner('Searching...'):
        time.sleep(1)
        
        category_selection = str(tuple(category_multi_select)) if category_multi_select else None
        brand_selection = str(tuple(brand_multi_select)) if brand_multi_select else None
        os_selection = str(tuple(os_multi_select)) if os_multi_select else None

        # Remove trailing comma for single-item tuples
        if category_selection and len(category_multi_select) == 1:
            category_selection = category_selection.replace(",", "")
        if brand_selection and len(brand_multi_select) == 1:
            brand_selection = brand_selection.replace(",", "")
        if os_selection and len(os_multi_select) == 1:
            os_selection = os_selection.replace(",", "")

        conditions = []
        if category_selection is not None:
            conditions.append(f'category IN {category_selection}')
        if brand_selection is not None:
            conditions.append(f'brand IN {brand_selection}')
        if os_selection is not None:
            conditions.append(f'operating_system IN {os_selection}')
        conditions.append(f'price >= {price_slider[0]}')
        conditions.append(f'price <= {price_slider[1]}')
        conditions.append(f'rating >= {rating_slider[0]}')
        conditions.append(f'rating <= {rating_slider[1]}')

        query = f'SELECT * FROM products WHERE {" AND ".join(conditions)}'
        print(query)
        
        result_set = client.execute(query).fetchall()



        columns = [column[0] for column in client.description]
        
        filtered_products_df = pd.DataFrame(result_set, columns=columns)
        filtered_products_df['combined_embedding'] = filtered_products_df['combined_embedding'].apply(lambda x: ast.literal_eval(x))

        # filtered_products_df['id'] = filtered_products_df['id'].astype(str)
        
        if 'query_for_sorting' in st.session_state:
            query_embedding = get_embedding(st.session_state.query_for_sorting, model='text-embedding-ada-002')
            
            st.session_state.query_embedding = query_embedding

            filtered_products_df = recommend_products(query_embedding, filtered_products_df)

            filtered_products_df.sort_values('similarities', ascending=False, inplace=True)

        st.session_state.filtered_products_df = filtered_products_df
    if 'product' in st.session_state:
        if st.session_state['product'] in filtered_products_df['id'].values:
            view(st.session_state['product'], filtered_products_df)
    view_products(filtered_products_df)
    update_query_and_sort_results(query_input)  # This line is added
elif st.session_state.page == 'view':
    if 'product' in st.session_state and st.session_state['product'] in st.session_state.filtered_products_df['id'].values:
        view(st.session_state['product'], st.session_state.filtered_products_df)

client.close()