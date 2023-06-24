import streamlit as st
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')
import os
import tiktoken
import psycopg2
import pinecone
import openai
import cohere
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY
COHERE_KEY = os.getenv('COHERE_KEY')
AWS_RDS_HOSTNAME = os.getenv('AWS_RDS_HOSTNAME')
AWS_RDS_PORT = os.getenv('AWS_RDS_PORT')
AWS_RDS_DB = os.getenv('AWS_RDS_DB')
AWS_RDS_UN = os.getenv('AWS_RDS_UN')
AWS_RDS_PW = os.getenv('AWS_RDS_PW')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
co = cohere.Client(COHERE_KEY)

# region load_dbs
@st.cache_resource
def load_sql():
    conn = psycopg2.connect(
        host=AWS_RDS_HOSTNAME + '.us-east-2.rds.amazonaws.com',
        port=AWS_RDS_PORT,
        dbname=AWS_RDS_DB,
        user=AWS_RDS_UN,
        password=AWS_RDS_PW
    )
    conn.autocommit = True
    client = conn.cursor()
    return client

@st.cache_resource
def load_pinecone():
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    return pinecone.Index("ecommerce")

client = load_sql()
pinecone_index = load_pinecone()
# endregion load_dbs

# CHAT FUNCTION
def stream_response(user_input, asin):
    '''User query --> Embedding --> Pinecone --> Combine top_k docs with prompt --> OpenAI --> Response'''
        
    # User query --> Embedding
    res = openai.Embedding.create(input=[user_input], engine="text-embedding-ada-002")
    query_embedding = res['data'][0]['embedding']

    # --> Pinecone --> Similar docs
    top_k = 100
    results = pinecone_index.query(query_embedding, 
                                    filter={
                                        "asin": {"$eq": asin}
                                    },
                                    top_k=top_k,
                                    namespace='reviews', 
                                    include_metadata=True)

 
    documents = [r['metadata']['review_text'] for r in results['matches']]
    
    rerank_hits = co.rerank(query=user_input, documents=documents, top_n=12, model="rerank-multilingual-v2.0")
    pinecone_ranks = [rerank_hits.results[i].index for i in range(0, len(rerank_hits.results))]
    reranked_results = [results['matches'][idx] for idx in pinecone_ranks]

    # --> Combine prompt with similar docs
    context = f"**USER QUESTION:**\n{user_input}\n\n**REVIEWS:**\n"

    for doc in reranked_results:
        context += f"{doc['metadata']['url']}: {doc['metadata']['review_text']}\n"
        
    context += "\n**RESPONSE:**\n"
    system_prompt = f"""You are a helpful assistant that reads Amazon reviews of a computer hardware product in order to answer questions for Amazon users. 
    Determine if the user's question is specific to the selected product or a general question about computers. 
    If the question is unrelated to computers or the product, respond with "I don't know".  
    If the question is about computers generally, ignore the reviews and provide a general response. 
    If product-specific, perform the following steps:  
    
    1) Read the following reviews and determine which portions of the reviews are relevant to the user's question
    2) In a markdown bulleted list, provide 0-5 direct quotes from the reviews that are relevant to the user's question (along with the URL of the review): * "quote from review" (full url of review)
    3) Finally, provide a markdown paragraph response to the user's question based on the reviews

    {context}
    """
    messages = [
        {"role": "system", "content": f"{system_prompt}"},
        {"role": "user", "content": user_input},
    ]
    
    # --> Response
    res_box = st.empty()
    response = []
    for resp in openai.ChatCompletion.create(model="gpt-4", messages=messages, temperature=0, stream=True ):
        response.append(resp['choices'][0]['delta'].get('content', ''))
        result = "".join(response).strip()
        res_box.markdown(f'*{result}*') 

# PRODUCT PAGE
def view(product_id, df):
    
    if 'product' in st.session_state and st.session_state.product != '' and \
        'page' in st.session_state and st.session_state.page == 'view':
        tab1, tab2 = st.tabs(['Product', 'Chat with Reviews'])
        
        with tab1:
            image_column, info_column = st.columns(2)
            product_series = df.loc[df.id == product_id, :].squeeze()
            image_column.image(f'./images_resized/{product_series["asin"]}.jpg', use_column_width='always')
            
            # print(product_id, product_series["asin"])
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
                                                    "n_tokens": {"$gt": 7}
                                                },
                                                top_k=1,
                                                namespace='reviews', 
                                                include_metadata=True)
                
                documents = [r['metadata']['review_text'] for r in results['matches']]
                
                rerank_hits = co.rerank(query=st.session_state.query_for_sorting, documents=documents, top_n=1, model="rerank-multilingual-v2.0")
                pinecone_ranks = [rerank_hits.results[i].index for i in range(0, len(rerank_hits.results))]
                reranked_results = [results['matches'][idx] for idx in pinecone_ranks]
                
                date = reranked_results[0]['metadata']['date']
                numPeopleFoundHelpful = reranked_results[0]['metadata']['numPeopleFoundHelpful']
                rating = reranked_results[0]['metadata']['rating']
                review_text = reranked_results[0]['metadata']['review_text']
                url = reranked_results[0]['metadata']['url']

                info_column.write(f'**Most Similar Review to User Query**: {review_text}')
                info_column.write(f'Review url: {url}')
                info_column.write(f'Date: {date}')
                info_column.write(f'Rating: {rating} / 5')
                info_column.write(f'Num People who Found this Review Helpful: {numPeopleFoundHelpful}')

            if st.button('Back', key='back_tab1'):
                del st.session_state['product']
                st.session_state.popped=True
                st.session_state.page = 'search'
                st.experimental_rerun()
                
        with tab2:

            if st.button('Back', key='back_tab2'):
                del st.session_state['product']
                st.session_state.popped=True
                st.session_state.page = 'search'
                st.experimental_rerun()
            
            st.title("Reviews Chat")
            user_input = st.text_input("Ask a question:")
            if user_input:
                asin = df.loc[df.id == product_id, 'asin'].item()
                # with st.spinner("Generating response..."):
                #     chatbot_response = generate_response(user_input, asin)
                # st.write(chatbot_response)
                
                stream_response(user_input, asin)

        
# SINGLE PRODUCT PAGE
def set_viewed_product(product):
    '''This is called when a user clicks on a product to view it'''
    st.session_state.product = product.id
    st.session_state.page = 'view'
    st.experimental_rerun()

# HOME PAGE (recommended/ranked product list based on sidebar selections and user query)
def view_products(df, products_per_row=7):
    '''Home page -- prior to Search button press, this just shows most popular products'''
    if 'product' not in st.session_state and st.session_state.page == 'search':
        if (st.session_state.from_reload) or ('popped' not in st.session_state or st.session_state.popped==False):
            st.header('E-Commerce Semantic Consensus')
            st.caption('Tabular + Semantic Search on Product Reviews with Pinecone')
            num_rows = min(10, int(np.ceil(len(df) / products_per_row)))
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
                    container.image(f'./images_resized/{product[1]["asin"]}.jpg', use_column_width='always')
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
        
        # get most similar **REVIEWS**
        results = pinecone_index.query(st.session_state.query_embedding, 
                                        filter={
                                            "asin": {"$in": st.session_state.filtered_asins},
                                            "n_tokens": {"$gt": 7},
                                            "rating": {"$gte": 3}
                                        },
                                        top_k=750,
                                        namespace='reviews', 
                                        include_metadata=True)
        
        scores = [r['score'] for r in results['_data_store']['matches']]
        filtered_reviews_df = pd.DataFrame.from_records([r['metadata'] for r in results['_data_store']['matches']])
        filtered_reviews_df['similarities'] = scores
        
        n = 77
        filtered_reviews_df.sort_values('similarities', ascending=False, inplace=True)
        
        documents = filtered_reviews_df.review_text.tolist()

        rerank_hits = co.rerank(query=st.session_state.query_for_sorting, documents=documents, top_n=77*4, model="rerank-multilingual-v2.0")
        cohere_ranks = [i for i in range(0, len(rerank_hits.results))]
        pinecone_ranks = [rerank_hits.results[i].index for i in range(0, len(rerank_hits.results))]
        cohere_scores = [rerank_hits.results[i].relevance_score for i in range(0, len(rerank_hits.results))]

        pinecone_to_cohere_dict = dict(zip(pinecone_ranks, cohere_ranks))
        pinecone_to_cohere_scores = dict(zip(pinecone_ranks, cohere_scores))
        filtered_reviews_df.reset_index(drop=False, inplace=True)
        filtered_reviews_df.rename({'index':'pinecone_rank'}, axis=1, inplace=True)
        filtered_reviews_df['cohere_rank'] = filtered_reviews_df['pinecone_rank'].map(pinecone_to_cohere_dict)
        filtered_reviews_df['cohere_score'] = filtered_reviews_df['pinecone_rank'].map(pinecone_to_cohere_scores)
        filtered_reviews_df = filtered_reviews_df.loc[(~filtered_reviews_df.cohere_rank.isna()),:].sort_values('cohere_score', ascending=False)
        
        # drop duplicates in filtered_reviews_df (on asin), keep first
        filtered_reviews_df.drop_duplicates(subset=['asin'], keep='first', inplace=True)
        filtered_reviews_df = filtered_reviews_df.head(n*4)
        rank_dict = dict(zip(filtered_reviews_df.asin, range(1, len(filtered_reviews_df)+1)))

        filtered_asins = filtered_reviews_df.asin.tolist()
        st.session_state.filtered_asins = filtered_asins
        
        # convert from reviews to **PRODUCTS**
        query = f"SELECT * FROM products WHERE asin IN {tuple(filtered_asins)}"
        client.execute(query)
        result_set = client.fetchall()
        
        columns = [column[0] for column in client.description]
        filtered_products_df = pd.DataFrame(result_set, columns=columns)
        
        filtered_products_df['similarities'] = filtered_products_df['asin'].map(dict(rank_dict))
        # filtered_products_df.sort_values('similarities', inplace=True)      
        
        # first sort 
        filtered_products_df = filtered_products_df.sort_values('similarities', ascending=True).head(n*2) # return more since some will be removed
        filtered_products_df = filtered_products_df[~filtered_products_df['title_text'].str.lower().str.split().str[:2].duplicated(keep='first')] # don't recommend similar products
        filtered_products_df = filtered_products_df.sort_values('similarities', ascending=True).head(n) # return desired amount
        # filtered_products_df = filtered_products_df.sort_values('num_reviews', ascending=False).head(n) # return desired amount
        
        st.session_state.filtered_products_df = filtered_products_df

def recsys():
    with st.spinner('Searching...'):
        time.sleep(1)
        
        if ('FormSubmitter:filter_form-Apply Filters' in st.session_state and st.session_state['FormSubmitter:filter_form-Apply Filters']) or \
            ('from_reload' in st.session_state and st.session_state.from_reload):

            # Get user selections for tabular filters
            category_selection = str(tuple(st.session_state.category_multi_selection)) if st.session_state.category_multi_selection else None
            brand_selection = str(tuple(st.session_state.brand_multi_selection)) if st.session_state.brand_multi_selection else None
            os_selection = str(tuple(st.session_state.os_multi_selection)) if st.session_state.os_multi_selection else None
            # Remove trailing comma for single-item tuples
            if category_selection and len(st.session_state.category_multi_selection) == 1:
                category_selection = category_selection.replace(",", "")
            if brand_selection and len(st.session_state.brand_multi_selection) == 1:
                brand_selection = brand_selection.replace(",", "")
            if os_selection and len(st.session_state.os_multi_selection) == 1:
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
        
        client.execute(query)
        result_set = client.fetchall()
        columns = [column[0] for column in client.description]
        filtered_products_df = pd.DataFrame(result_set, columns=columns)
        st.session_state.filtered_products_df = filtered_products_df
        
        filtered_asins = filtered_products_df.asin.tolist()
        # st.ss.INIT: filtered_asins
        st.session_state.filtered_asins = filtered_asins
        
        update_query_and_sort_results()
        view_products(st.session_state.filtered_products_df)
        st.session_state.popped=True
        st.session_state.from_reload=False
        
# Prep tabular filter data
# @st.cache_data
def get_all_tabular_categories(_client):
    _client.execute('''SELECT DISTINCT category FROM products WHERE num_reviews > 25''')
    distinct_categories = _client.fetchall()
    _client.execute('''SELECT DISTINCT brand FROM products WHERE num_reviews > 25 ORDER BY brand''')
    distinct_brands = _client.fetchall()
    _client.execute('''SELECT DISTINCT operating_system FROM products WHERE num_reviews > 25''')
    distinct_operating_systems = _client.fetchall()
    _client.execute('''SELECT min(price), max(price) FROM products WHERE num_reviews > 25''')
    min_max_price_tuple = _client.fetchone()
    
    # Flatten the list of tuples
    distinct_categories = [item[0] for item in distinct_categories]
    distinct_brands = [item[0] for item in distinct_brands]
    distinct_operating_systems = [item[0] for item in distinct_operating_systems]
    
    st.session_state.distinct_categories = distinct_categories
    st.session_state.distinct_brands = distinct_brands
    st.session_state.distinct_operating_systems = distinct_operating_systems
    st.session_state.min_max_price_tuple = min_max_price_tuple
    

if ('query_for_sorting' not in st.session_state) or st.session_state.query_for_sorting == '':
    get_all_tabular_categories(client)

if 'page' not in st.session_state:
    st.session_state.page = 'search'

# SIDEBAR -- submit button active on home page only
with st.sidebar.form(key='filter_form'):

    if ('query_for_sorting' not in st.session_state) or st.session_state.query_for_sorting == '':
        st.text_input("Enter your query: (optional)", key='query_for_sorting')
    else:
        st.text_input("Enter your query: (optional)", key='query_for_sorting', value=st.session_state.query_for_sorting)

    if ('category_multi_selection' not in st.session_state) or (not st.session_state.category_multi_selection):
        if 'distinct_categories' not in st.session_state:
            get_all_tabular_categories(client)
        category_multi_selection = st.multiselect('Product Category:', st.session_state.distinct_categories, default=st.session_state.distinct_categories, key="category_multi_selection")
        brand_multi_selection = st.multiselect('Computer Brands:', st.session_state.distinct_brands, default=st.session_state.distinct_brands, key="brand_multi_selection")
        os_multi_selection = st.multiselect('Operating Systems:', st.session_state.distinct_operating_systems, default=st.session_state.distinct_operating_systems, key="os_multi_selection")
    else:
        category_multi_selection = st.multiselect('Product Category:', st.session_state.distinct_categories, default=st.session_state.category_multi_selection, key="category_multi_selection")
        brand_multi_selection = st.multiselect('Computer Brands:', st.session_state.distinct_brands, default=st.session_state.brand_multi_selection, key="brand_multi_selection")
        os_multi_selection = st.multiselect('Operating Systems:', st.session_state.distinct_operating_systems, default=st.session_state.os_multi_selection, key="os_multi_selection")
        

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
    
    allow_click = False
    if 'page' in st.session_state and st.session_state.page == 'view':
        allow_click = True
    st.form_submit_button(label='Apply Filters', on_click=recsys, disabled=allow_click)

# HOME/SEARCH PAGE
if st.session_state.page == 'search' and ('product' not in st.session_state):
    st.session_state.from_reload = True
    recsys()

# PRODUCT PAGE
elif st.session_state.page == 'view':
    if 'product' in st.session_state and st.session_state['product'] in st.session_state.filtered_products_df['id'].values:
        view(st.session_state['product'], st.session_state.filtered_products_df)
