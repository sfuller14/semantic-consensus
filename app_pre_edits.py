import streamlit as st
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')
import os
import tiktoken
import sqlite3
import pinecone
import openai
from dotenv import dotenv_values
config = dotenv_values(".env")

openai.api_key = config["OPENAI_API_KEY"]

# region load_dbs
@st.cache_resource
def load_sql():
    conn = sqlite3.connect('ecommerce.db', check_same_thread=False)
    client = conn.cursor()
    client.execute('PRAGMA journal_mode=WAL')
    return client

client = load_sql()

@st.cache_resource
def load_pinecone():
    pinecone.init(api_key=config['PINECONE_API_KEY'], environment=config['PINECONE_ENVIRONMENT'])
    return pinecone.Index("ecommerce")
pinecone_index = load_pinecone()
# endregion load_dbs

# CHAT FUNCTION
def generate_response(user_input, asin):
    '''User query --> Embedding --> Pinecone --> Combine top_k docs with prompt --> OpenAI --> Response'''
        
    # User query --> Embedding
    res = openai.Embedding.create(input=[user_input], engine="text-embedding-ada-002")
    query_embedding = res['data'][0]['embedding']

    # --> Pinecone --> Similar docs
    top_k = 8
    results = pinecone_index.query(query_embedding, 
                                    filter={
                                        "asin": {"$eq": asin},
                                    },
                                    top_k=top_k,
                                    namespace='reviews', 
                                    include_metadata=True)

 
    # --> Combine prompt with similar docs
    context = f"**USER QUESTION:**\n{user_input}\n\n**REVIEWS:**\n"

    for doc in results['matches']:#[0]['metadata']:
        context += f"{doc['metadata']['url']}: {doc['metadata']['review_text']}\n"
        
    context += "\n**RESPONSE:**\n"
    system_prompt = f"""You are a helpful assistant that reads Amazon reviews of a computer hardware product in order to answer questions for Amazon users. 
    Determine if the user's question is specific to the selected product or a general question about computers (or anything else). If general, ignore the reviews and provide a general response. If product-specific, perform the following steps:
    
    1) Read the following reviews and determine which portions of the reviews are relevant to the user's question
    2) In a bulleted list, provide 0-5 direct quotes from the reviews that are relevant to the user's question (along with the URL of the review): * "quote from review" (url of review)
    3) Finally, provide a complete but terse response to the user's question based on the reviews

    {context}
    """
    messages = [
        {"role": "system", "content": f"{system_prompt}"},
        {"role": "user", "content": user_input},
    ]


    # --> OpenAI
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0,
    )
    
    # --> Response
    return response.choices[0].message.content.strip()

# PRODUCT PAGE
def view(product_id, df):
    tab1, tab2 = st.tabs(['Product', 'Chat with Reviews'])
    
    with tab1:
        image_column, info_column = st.columns(2)
        product_series = df.loc[df.id == product_id, :].squeeze()
        image_column.image(f'./assets_resized/{product_series["asin"]}.jpg', use_column_width='always')
        
        print(product_id, product_series["asin"])
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
            asin = product_series['asin']
            
            results = pinecone_index.query(st.session_state.query_embedding, 
                                            filter={
                                                "asin": {"$eq": asin},
                                            },
                                            top_k=1,
                                            namespace='reviews', 
                                            include_metadata=True)
            
            # print(results)
            date = results['matches'][0]['metadata']['date']
            # n_tokens = results['matches'][0]['metadata']['n_tokens']
            numPeopleFoundHelpful = results['matches'][0]['metadata']['numPeopleFoundHelpful']
            rating = results['matches'][0]['metadata']['rating']
            review_text = results['matches'][0]['metadata']['review_text']
            url = results['matches'][0]['metadata']['url']
            # weighting = results['matches'][0]['metadata']['weighting']
            # max_sim = results['score']

            info_column.write(f'**Most Similar Review to User Query**: {review_text}')
            info_column.write(f'Review url: {url}')
            info_column.write(f'Date: {date}')
            info_column.write(f'Rating: {rating} / 5')
            info_column.write(f'Num People who Found this Review Helpful: {numPeopleFoundHelpful}')

        if st.button('Back'):
            del st.session_state['product']
            # st.runtime.legacy_caching.clear_cache()
            st.session_state.popped=False
            st.session_state.page = 'search'
            st.experimental_rerun()
            
    with tab2:
        
        st.title("Reviews Chat")
        user_input = st.text_input("Ask a question:")
        if user_input:
            asin = df.loc[df.id == product_id, 'asin'].item()
            with st.spinner("Generating response..."):
                chatbot_response = generate_response(user_input, asin)
            st.write(chatbot_response)
        
# SINGLE PRODUCT PAGE
# st.cache_data(experimental_allow_widgets=True)
def set_viewed_product(product):#, df):
    '''This is called when a user clicks on a product to view it'''
    st.session_state.product = product.id
    # view(product.id, df)
    st.session_state.page = 'view'
    st.experimental_rerun()

# HOME PAGE (recommended/ranked product list based on sidebar selections and user query)
# @st.cache_data(experimental_allow_widgets=True)
def view_products(df, products_per_row=7):
    '''Home page -- prior to Search button press, this just shows most popular products'''
    if 'popped' not in st.session_state or st.session_state.popped==False:
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
                    set_viewed_product(product=product[1])#, df=df)
                container.image(f'./assets_resized/{product[1]["asin"]}.jpg', use_column_width='always')
        st.session_state.popped=True

# EMBED USER INPUT (search query or chatbot question)
def get_embedding(text, model="text-embedding-ada-002", max_tokens=8000):
    '''Custom implementation (rather than openai.embeddings_utils) to avoid potential context window overflow'''
    text = text.replace("\n", " ")
    encoding = tiktoken.encoding_for_model(model)
    text = encoding.decode(encoding.encode(text)[:max_tokens]) if len(encoding.encode(text)) >= max_tokens else text

    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']

def update_query_and_sort_results():
    if 'query_for_sorting' in st.session_state and st.session_state.query_for_sorting != '':
        st.session_state.query_embedding = get_embedding(st.session_state.query_for_sorting, model='text-embedding-ada-002')
    else:
        if 'query_embedding' in st.session_state:
            del st.session_state.query_embedding
    
    if 'query_embedding' in st.session_state and st.session_state.query_embedding \
        and 'filtered_asins' in st.session_state and st.session_state.filtered_asins:# \
        # and 'filtered_products_df' in st.session_state and not st.session_state.filtered_products_df.empty:
        
        # get most similar **REVIEWS**
        results = pinecone_index.query(st.session_state.query_embedding, 
                                        filter={
                                            "asin": {"$in": st.session_state.filtered_asins},
                                        },
                                        top_k=1000,
                                        namespace='reviews', 
                                        include_metadata=True)
        
        scores = [r['score'] for r in results['_data_store']['matches']]
        filtered_reviews_df = pd.DataFrame.from_records([r['metadata'] for r in results['_data_store']['matches']])
        filtered_reviews_df['similarities'] = scores
        
        n = 56
        filtered_reviews_df.sort_values('similarities', ascending=False, inplace=True)
        # drop duplicates in filtered_reviews_df (on asin), keep first
        filtered_reviews_df.drop_duplicates(subset=['asin'], keep='first', inplace=True)
        filtered_reviews_df = filtered_reviews_df.head(n*6)
        rank_dict = dict(zip(filtered_reviews_df.asin, range(1, len(filtered_reviews_df)+1)))

        filtered_asins = filtered_reviews_df.asin.tolist()
        st.session_state.filtered_asins = filtered_asins
        
        # convert from reviews to **PRODUCTS**
        query = f"SELECT * FROM products WHERE asin IN {tuple(filtered_asins)}"
        result_set = client.execute(query).fetchall()
        
        columns = [column[0] for column in client.description]
        filtered_products_df = pd.DataFrame(result_set, columns=columns)
        
        filtered_products_df['similarities'] = filtered_products_df['asin'].map(dict(rank_dict))
        # filtered_products_df.sort_values('similarities', inplace=True)      
        
        # first sort 
        filtered_products_df = filtered_products_df.sort_values('similarities', ascending=True).head(n*2) # return more since some will be removed
        filtered_products_df = filtered_products_df[~filtered_products_df['title_text'].str.lower().str.split().str[:2].duplicated(keep='first')] # don't recommend similar products
        filtered_products_df = filtered_products_df.sort_values('similarities', ascending=True).head(n) # return desired amount
        # print(filtered_products_df.head())
        # print(filtered_products_df.columns.tolist())
        # filtered_products_df.reset_index(inplace=True, drop=True)
        # filtered_products_df.reset_index(inplace=True, drop=False)
        # filtered_products_df.rename(columns={'index': 'id'}, inplace=True) # get new ordering
        # cast to string?
        # filtered_products_df['id'] = filtered_products_df['id'].astype(str)
        
        st.session_state.filtered_products_df = filtered_products_df

def recsys():
    with st.spinner('Searching...'):
        time.sleep(1)
        
        if 'category_multi_select' in st.session_state and st.session_state.category_multi_select:
            # Get user selections for tabular filters
            category_selection = str(tuple(st.session_state.category_multi_select)) if st.session_state.category_multi_select else None
            brand_selection = str(tuple(st.session_state.brand_multi_select)) if st.session_state.brand_multi_select else None
            os_selection = str(tuple(st.session_state.os_multi_select)) if st.session_state.os_multi_select else None
            # Remove trailing comma for single-item tuples
            if category_selection and len(st.session_state.category_multi_select) == 1:
                category_selection = category_selection.replace(",", "")
            if brand_selection and len(st.session_state.brand_multi_select) == 1:
                brand_selection = brand_selection.replace(",", "")
            if os_selection and len(st.session_state.os_multi_select) == 1:
                os_selection = os_selection.replace(",", "")

            # Build tabular query string dynamically
            conditions = []
            if category_selection is not None:
                conditions.append(f'category IN {category_selection}')
            if brand_selection is not None:
                conditions.append(f'brand IN {brand_selection}')
            if os_selection is not None:
                conditions.append(f'operating_system IN {os_selection}')
            conditions.append(f'price >= {st.session_state.price_slider[0]}')
            conditions.append(f'price <= {st.session_state.price_slider[1]}')
            conditions.append(f'rating >= {st.session_state.rating_slider[0]}')
            conditions.append(f'rating <= {st.session_state.rating_slider[1]}')

            query = f'SELECT * FROM products WHERE {" AND ".join(conditions)}'
        else:
            query = f'SELECT * FROM products'
        
        result_set = client.execute(query).fetchall()
        columns = [column[0] for column in client.description]
        filtered_products_df = pd.DataFrame(result_set, columns=columns)
        st.session_state.filtered_products_df = filtered_products_df

        # st.session_state.filtered_products_df = filtered_products_df
        
        filtered_asins = filtered_products_df.asin.tolist()
        # st.ss.INIT: filtered_asins
        st.session_state.filtered_asins = filtered_asins
        
        update_query_and_sort_results()
        view_products(st.session_state.filtered_products_df)
        
# Prep tabular filter data
@st.cache_data
def get_all_tabular_categories(_client):
    distinct_categories = _client.execute('''SELECT DISTINCT category FROM products''').fetchall()
    distinct_brands = _client.execute('''SELECT DISTINCT brand FROM products''').fetchall()
    distinct_operating_systems = _client.execute('''SELECT DISTINCT operating_system FROM products''').fetchall()
    min_max_price_tuple = _client.execute('''SELECT min(price), max(price) FROM products''').fetchone()
    
    # Flatten the list of tuples
    distinct_categories = [item[0] for item in distinct_categories]
    distinct_brands = [item[0] for item in distinct_brands]
    distinct_operating_systems = [item[0] for item in distinct_operating_systems]
    
    st.session_state.distinct_categories = distinct_categories
    st.session_state.distinct_brands = distinct_brands
    st.session_state.distinct_operating_systems = distinct_operating_systems
    st.session_state.min_max_price_tuple = min_max_price_tuple
    
    return distinct_categories, distinct_brands, distinct_operating_systems, min_max_price_tuple

if ('query_for_sorting' not in st.session_state) or st.session_state.query_for_sorting == '':
    get_all_tabular_categories(client)

# def home_sidebar():
with st.sidebar.form(key='filter_form'):

    if ('query_for_sorting' not in st.session_state) or st.session_state.query_for_sorting == '':
        st.text_input("Enter your query: (optional)", key='query_for_sorting')
    else:
        st.text_input("Enter your query: (optional)", key='query_for_sorting', value=st.session_state.query_for_sorting)
    # st.markdown("_Please press Enter after typing to submit the query._", unsafe_allow_html=True)

    if ('category_multi_selection' not in st.session_state) or (not st.session_state.category_multi_selection):
        category_multi_select = st.multiselect('Product Category:', st.session_state.distinct_categories, default=st.session_state.distinct_categories, key="category_multi_selection")
        brand_multi_select = st.multiselect('Computer Brands:', st.session_state.distinct_brands, default=st.session_state.distinct_brands, key="brand_multi_selection")
        os_multi_select = st.multiselect('Operating Systems:', st.session_state.distinct_operating_systems, default=st.session_state.distinct_operating_systems, key="os_multi_selection")
    else:
        category_multi_select = st.multiselect('Product Category:', st.session_state.distinct_categories, default=st.session_state.category_multi_selection, key="category_multi_selection")
        brand_multi_select = st.multiselect('Computer Brands:', st.session_state.distinct_brands, default=st.session_state.brand_multi_selection, key="brand_multi_selection")
        os_multi_select = st.multiselect('Operating Systems:', st.session_state.distinct_operating_systems, default=st.session_state.os_multi_selection, key="os_multi_selection")
        

    price_slider = st.slider(
        'Select a price range',
        min_value=0.0, 
        max_value=6000.00,
        value=(st.session_state.min_max_price_tuple[0], st.session_state.min_max_price_tuple[1]),
        key="price_slider"
    )

    rating_slider = st.slider(
        'Select range for ratings (1-5)',
        min_value=1, 
        max_value=5,
        value=(1,5),
        key="rating_slider"
    )
    
    submit = st.form_submit_button(label='Apply Filters', on_click=recsys)

# st.ss.INIT: page
if 'page' not in st.session_state:
    st.session_state.page = 'search'

if st.session_state.page == 'search' and ('product' not in st.session_state):
    st.session_state.popped=False
    # home_sidebar()
    recsys()

elif st.session_state.page == 'view':
    if 'product' in st.session_state and st.session_state['product'] in st.session_state.filtered_products_df['id'].values:
        view(st.session_state['product'], st.session_state.filtered_products_df)