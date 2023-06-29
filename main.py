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
import spacy
from dotenv import load_dotenv
load_dotenv()

# region API-keys
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
# endregion API-keys

# region streamlit-config
st.set_page_config(page_title='Recsys', layout='wide', page_icon="computer")#initial_sidebar_state='expanded', 
# endregion streamlit-config

# region load_dbs_and_highlighterfunc
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

@st.cache_resource
def load_spacy_model(model_name):
    return spacy.load(model_name)

client = load_sql()
pinecone_index = load_pinecone()
nlp = load_spacy_model('en_core_web_sm')

# @st.cache_data
def preprocess_text_spacy(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_stop]

# @st.cache_data
def highlight_matches_streamlit(input_text, lemmatized_query):
    words = input_text.split()
    highlighted_words = [f'**:violet[{word}]**' if word.rstrip('.,;:!?').lower() in lemmatized_query else word for word in words]
    return ' '.join(highlighted_words)
# endregion load_dbs_and_highlighterfunc

# region functions
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

    # top_n in the following rerank call can be arbitrarily high (will get truncated before passing to LLM if context window is exceeded), but empirically 12 works well
    rerank_hits = co.rerank(query=user_input, documents=documents, top_n=12, model="rerank-multilingual-v2.0")
    pinecone_ranks = [rerank_hits.results[i].index for i in range(0, len(rerank_hits.results))]
    reranked_results = [results['matches'][idx] for idx in pinecone_ranks]

 
    # --> Combine prompt with similar docs
    context = f"**USER QUESTION:**\n{user_input}\n\n**PRODUCT DESCRIPTION:**\n{st.session_state.asins_titletext_dict[reranked_results[0]['metadata']['asin']]}\n**REVIEWS:**\n"

    for doc in reranked_results:
        context += f"{doc['metadata']['url']}: {doc['metadata']['review_text']}\n"

    # Don't exceed 8000 tokens
    encoding = tiktoken.encoding_for_model('gpt-4')
    context = encoding.decode(encoding.encode(context)[:7700]) if len(encoding.encode(context)) >= 7700 else context
        
    context += "\n**RESPONSE TO DISPLAY TO THE USER:**"
    system_prompt = f"""You are a helpful assistant that reads Amazon reviews of a desktop/laptop product and answers a question for an Amazon user. 
    If the question is about computers generally, ignore the reviews and provide a general response. If the question is product-specific, perform the following steps:  
    
    1) Read the following context then in a markdown bulleted list, provide 0-5 direct quotes from the reviews that are relevant to the user's question (along with the FULL URL of the review). All link texts should say "See Review on Amazon". Follow this format exactly: * "Relevant portion from a review" [See Review on Amazon](https://www.amazon.com/url-associated-with-review)
    2) In a short paragraph, respond to the user's question based on the reviews. You may refer to the product description as well if it contains information that answers the user's question, but focus primarily on conveying relevant information from the reviews whenever possible. 
    
    Try to generalize the information in the reviews - do not rehash the provided review snippets verbatim (as the user will have read them); just use them to provide a concise, measured answer to the user's question. 
    Strive to provide a response, but do not make up information. If the question is unrelated to computers or the product, respond with the exact phrase "Sorry, I don't know".
    *CRITICALLY, the response you provide is being displayed directly to the user who asked the question -- RESPOND DIRECTLY TO THE USER.*
    
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
        res_box.markdown(f'{result}') 

# PRODUCT PAGE
def view(product_id, df):
    
    if 'product' in st.session_state and st.session_state.product != '' and \
        'page' in st.session_state and st.session_state.page == 'view':
        
        if st.button('Back to Search', key='back_tab1'):
            del st.session_state['product']
            st.session_state.popped=True
            st.session_state.page = 'search'
            st.experimental_rerun()
        
        tab1, tab2 = st.tabs(['Product', 'Chat with Reviews'])
        
        with tab1:
            image_column, info_column = st.columns(2)
            product_series = df.loc[df.id == product_id, :].squeeze()
            image_column.image(f'./thumbnails/{product_series["asin"]}.jpg', use_column_width='always')
            
            title_text = highlight_matches_streamlit(product_series["title_text"], st.session_state.lemmatized_query) if 'lemmatized_query' in st.session_state else product_series["title_text"]
            info_column.write(f'''
                              * **PRODUCT**:    {title_text}
                              * **PRICE**:    ${product_series["price"]}
                              * **RATING**:    {product_series["rating"]}
                              * **CATEGORY**:    {product_series["category"]}
                              * **OS**:    {product_series["operating_system"]}
                              * **BRAND**:    {product_series["brand"]}
                              * **SERIES**:    {product_series["series"]}
                              * **RAM**:    {product_series["ram"]}
                              * **STORAGE**:    {product_series["storage"]}
                              * **CPU MANUFACTURER**:    {product_series["cpu_manufacturer"]}
                              * **CPU MODEL**:    {product_series["cpu_model"]}
                              * **GPU MANUFACTURER**:    {product_series["gpu_manufacturer"]}
                              * **GPU MODEL**:    {product_series["gpu_model"]}
                              * **RELEASE DATE**:    {product_series["date_first_available"]}
                              * **url**:    {product_series["url"]}
                              ''')
            
            if 'query_embedding' in st.session_state:
                if 'lemmatized_query' not in st.session_state:
                    st.session_state.lemmatized_query = preprocess_text_spacy(st.session_state.query_for_sorting)
                
                asin = product_series['asin']
                
                results = pinecone_index.query(st.session_state.query_embedding, 
                                                filter={
                                                    "asin": {"$eq": asin},
                                                    "n_tokens": {"$gt": 7},
                                                    "rating": {"$gt": 3}
                                                },
                                                top_k=35,
                                                namespace='reviews', 
                                                include_metadata=True)
                
                documents = [r['metadata']['review_text'] for r in results['matches']]

                rerank_hits = co.rerank(query=st.session_state.query_for_sorting, documents=documents, top_n=5, model="rerank-multilingual-v2.0")
                pinecone_ranks = [rerank_hits.results[i].index for i in range(len(rerank_hits.results))]
                reranked_results = [results['matches'][idx] for idx in pinecone_ranks]

                date = reranked_results[0]['metadata']['date']
                numPeopleFoundHelpful = reranked_results[0]['metadata']['numPeopleFoundHelpful']
                rating = reranked_results[0]['metadata']['rating']
                review_text = reranked_results[0]['metadata']['review_text']
                url = reranked_results[0]['metadata']['url']

                image_column.write('---')
                image_column.write(f'''**#1 Relevant Review to User Query** ({round(rerank_hits.results[0].relevance_score*100,1)} Relevance Score):\n''') 
                image_column.write(f'''{highlight_matches_streamlit(review_text, st.session_state.lemmatized_query)}''')
                image_column.write(f'''[See Review on Amazon]({url})''')
                image_column.write(f'''* Date:   {date}''')
                image_column.write(f'''* Rating:   {int(rating)} / 5''')
                image_column.write(f'''* Num People who Found this Review Helpful:   {int(numPeopleFoundHelpful)}''')
                image_column.write('---')
                
                review_expdr = image_column.expander('Other relevant reviews', expanded=False)                
                for i in range(1, len(rerank_hits.results)):
                    date = reranked_results[i]['metadata']['date']
                    numPeopleFoundHelpful = reranked_results[i]['metadata']['numPeopleFoundHelpful']
                    rating = reranked_results[i]['metadata']['rating']
                    review_text = reranked_results[i]['metadata']['review_text']
                    url = reranked_results[i]['metadata']['url']

                    review_expdr.write(f'''* **#{i+1} Relevant Review to User Query**:   {highlight_matches_streamlit(review_text, st.session_state.lemmatized_query)}''')
                    review_expdr.write(f'''* Relevance score:   {round(rerank_hits.results[i].relevance_score*100,2)} ''')
                    review_expdr.write(f'''* Review url:   {url}''')
                    review_expdr.write(f'''* Date:   {date}''')
                    review_expdr.write(f'''* Rating:   {int(rating)} / 5''')
                    review_expdr.write(f'''* Num People who Found this Review Helpful:   {int(numPeopleFoundHelpful)}''')
                    review_expdr.write('---')
                    
            
            seller_expdr = info_column.expander('Product Description from Seller')
            seller_expdr.write(f'**PRODUCT DESCRIPTION**: {highlight_matches_streamlit(product_series["seller_text"], st.session_state.lemmatized_query) if "lemmatized_query" in st.session_state else product_series["seller_text"]}')
                            
        with tab2:

            st.title("Reviews Chat")
            user_input = st.text_input("Ask a question:")
            if user_input:
                asin = df.loc[df.id == product_id, 'asin'].item()
                
                stream_response(user_input, asin)

# GOTO PRODUCT PAGE
def set_viewed_product(product):
    '''This is called when a user clicks on a product to view it'''
    st.session_state.product = product.id
    st.session_state.page = 'view'
    st.experimental_rerun()

# region CSS
st.markdown("""
        <style>
               .block-container {
                    padding-top: 3rem;
                    padding-bottom: 0rem;
                }
        </style>
        """, unsafe_allow_html=True)

def header(header_text):
    st.markdown(f'<p style="font-size:42px;font-weight:bold;font-family:sans-serif;color:#ffb86c;">{header_text}</p>', unsafe_allow_html=True)
def subheader(header_text):
    st.markdown(f'<p style="font-size:18px;font-family:sans-serif;color:#50fa7bff;">{header_text}</p>', unsafe_allow_html=True)
# endregion CSS

# HOME PAGE (display hybrid search results)
def view_products(df, products_per_row=4):
    '''Home page -- prior to Search button press, this just shows most popular products'''
    if 'product' not in st.session_state and st.session_state.page == 'search':
        if (st.session_state.from_reload) or ('popped' not in st.session_state or st.session_state.popped==False):
            header('E-Commerce Recommendation System')
            subheader('Tabular + Semantic Search on Product Reviews with Pinecone & cohere.rerank()')
            st.markdown(f'[![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/sfuller14/semantic-consensus/)')
            st.markdown("""---""")
            
            num_rows = min(10, int(np.ceil(len(df) / products_per_row)))
            
            for i in range(num_rows):
                start = i * products_per_row
                end = start + products_per_row
                products = df.iloc[start:end]
                
                columns = st.columns(products_per_row)
                
                for product, column in zip(products.iterrows(), columns):# product is a tuple of (index, row)
                    
                    container = column.container()
                    
                    if 'query_for_sorting' in st.session_state and st.session_state.query_for_sorting != '' and 'asin_topreview_dict' in st.session_state:
                        tooltip = highlight_matches_streamlit(f'**Most relevant review:** ({round(product[1]["cohere_score"]*100,2)} relevance score) \n\n{st.session_state.asin_topreview_dict[product[1]["asin"]]}',
                                                              st.session_state.lemmatized_query)
                    else:
                        tooltip = 'Enter a query to see relevant reviews here'
                    title_suffix = '...' if len(product[1]["true_title"]) > 25 else ''
                    try:
                        container.markdown(f'**<p style="font-size: 17px;">{"{:.25}".format(product[1]["true_title"]) + title_suffix}</p>**', unsafe_allow_html=True)
                    except:
                        container.write(f'**{product[1]["true_title"]}**')                    
                    container.markdown(f'{product[1]["rating"]} :star: ({product[1]["num_reviews"]:,} ratings)',
                                       help=tooltip)

                    container.image(f'./thumbnails/{product[1]["asin"]}.jpg', use_column_width='always')
                    
                    button_key = f"view_{product[1]['id']}"
                    if container.button('View', key=button_key, use_container_width=True):
                        set_viewed_product(product=product[1])
                    
                    gpu_model = product[1]["gpu_model"].replace(product[1]["gpu_manufacturer"], '').strip()
                    gpu_suffix = '...' if len(product[1]["gpu_model"]) > 20 else ''
                    cpu_model = product[1]["cpu_model"].replace(product[1]["cpu_manufacturer"], '').strip()
                    cpu_suffix = '...' if len(product[1]["cpu_model"]) > 20 else ''
                    container.markdown((f'* **Price:** ${product[1]["price"]}\n'
                                        f'* **CPU:** {product[1]["cpu_manufacturer"]} ({"{:.20}".format(cpu_model) + cpu_suffix})\n'
                                        f'* **GPU:** {product[1]["gpu_manufacturer"]} ({"{:.20}".format(gpu_model) + gpu_suffix})\n'
                                        f'* **Release Date:** {product[1]["date_first_available"]}\n'
                                        ), unsafe_allow_html=True
                                       )
                    
                    container.divider()
            
            st.session_state.popped=True

# EMBED USER INPUT (search query or chatbot question)
def get_embedding(text, model="text-embedding-ada-002", max_tokens=8000):
    '''Custom implementation (rather than openai.embeddings_utils) to avoid potential context window overflow'''
    text = text.replace("\n", " ")
    encoding = tiktoken.encoding_for_model(model)
    text = encoding.decode(encoding.encode(text)[:max_tokens]) if len(encoding.encode(text)) >= max_tokens else text

    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']

# Determine products to display based on sidebar selections and query
def update_query_and_sort_results():
    if 'query_for_sorting' in st.session_state and st.session_state.query_for_sorting != '':
        st.session_state.query_embedding = get_embedding(st.session_state.query_for_sorting, model='text-embedding-ada-002')
        st.session_state.lemmatized_query = preprocess_text_spacy(st.session_state.query_for_sorting)
    else:
        if 'query_embedding' in st.session_state:
            del st.session_state.query_embedding
        if 'lemmatized_query' in st.session_state:
            del st.session_state.lemmatized_query
    
    if 'query_embedding' in st.session_state and st.session_state.query_embedding \
        and 'filtered_asins' in st.session_state and st.session_state.filtered_asins:# \
        
        if 'asin_topreview_dict' in st.session_state:
            del st.session_state.asin_topreview_dict
        
        # get most similar **REVIEWS**
        results = pinecone_index.query(st.session_state.query_embedding, 
                                        filter={
                                            "asin": {"$in": st.session_state.filtered_asins},
                                            "n_tokens": {"$gt": 10},
                                            "rating": {"$gte": 3}
                                        },
                                        top_k=750,
                                        namespace='reviews', 
                                        include_metadata=True)
        
        scores = [r['score'] for r in results['_data_store']['matches']]
        filtered_reviews_df = pd.DataFrame.from_records([r['metadata'] for r in results['_data_store']['matches']])
        filtered_reviews_df['similarities'] = scores
        
        # Concat review text and product description text for combo-reranking, if desired
        # filtered_reviews_df['seller_text'] = filtered_reviews_df['asin'].map(st.session_state.asins_sellertext_dict)
        # filtered_reviews_df['combined_text'] = '**PRODUCT REVIEW:**\n' + filtered_reviews_df['review_text'] + '\n**PRODUCT DESCRIPTION:**\n' + filtered_reviews_df['seller_text']
        
        n = 78
        filtered_reviews_df.sort_values('similarities', ascending=False, inplace=True)

        documents = filtered_reviews_df.review_text.tolist() # use filtered_reviews_df.combined_text if combo-reranking

        rerank_hits = co.rerank(query=st.session_state.query_for_sorting, documents=documents, top_n=n*4, model="rerank-multilingual-v2.0")
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
        filtered_reviews_df = filtered_reviews_df.head(n)
        rank_dict = dict(zip(filtered_reviews_df.asin, range(1, len(filtered_reviews_df)+1)))
        cohere_score_dict = dict(zip(filtered_reviews_df.asin, filtered_reviews_df.cohere_score))

        filtered_asins = filtered_reviews_df.asin.tolist()
        st.session_state.filtered_asins = filtered_asins
        
        st.session_state.asin_topreview_dict = dict(zip(filtered_reviews_df.asin, filtered_reviews_df.review_text))
        
        # convert from reviews to **PRODUCTS**
        query = f"SELECT * FROM products WHERE asin IN {tuple(filtered_asins)}"
        client.execute(query)
        result_set = client.fetchall()
        
        columns = [column[0] for column in client.description]
        filtered_products_df = pd.DataFrame(result_set, columns=columns)
        
        filtered_products_df['similarities'] = filtered_products_df['asin'].map(dict(rank_dict))
        filtered_products_df['cohere_score'] = filtered_products_df['asin'].map(dict(cohere_score_dict))
        
        # first sort 
        filtered_products_df = filtered_products_df.sort_values('similarities', ascending=True).head(n) # return desired amount
        
        st.session_state.filtered_products_df = filtered_products_df

def recsys():
    if (st.session_state.get('FormSubmitter:filter_form-Search', False)) or (st.session_state.get('filtered_products_df', None) is None):
        with st.spinner('Searching...'):
            time.sleep(1)
            
            if ('FormSubmitter:filter_form-Search' in st.session_state and st.session_state['FormSubmitter:filter_form-Search']) or \
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

                query = f'SELECT * FROM products WHERE num_reviews > 35 AND {" AND ".join(conditions)}'
            else:
                query = f'SELECT * FROM products WHERE num_reviews > 35'
            
            client.execute(query)
            result_set = client.fetchall()
            columns = [column[0] for column in client.description]
            filtered_products_df = pd.DataFrame(result_set, columns=columns)
            filtered_products_df.sort_values('num_reviews', ascending=False, inplace=True)
            st.session_state.filtered_products_df = filtered_products_df
            
            filtered_asins = filtered_products_df.asin.tolist()
            # Note: filtered_asins is only used in update_query_and_sort_results
            st.session_state.filtered_asins = filtered_asins
            st.session_state.asins_sellertext_dict = dict(zip(filtered_products_df.asin, filtered_products_df.seller_text))
            st.session_state.asins_titletext_dict = dict(zip(filtered_products_df.asin, filtered_products_df.title_text))
        
        update_query_and_sort_results()
        
    view_products(st.session_state.filtered_products_df)
    st.session_state.popped=True
    st.session_state.from_reload=False
        
# Prep tabular filter data
def get_all_tabular_categories(_client):
    _client.execute('''SELECT DISTINCT category FROM products WHERE num_reviews > 30''')
    distinct_categories = _client.fetchall()
    _client.execute('''SELECT DISTINCT brand FROM products WHERE num_reviews > 30 ORDER BY brand''')
    distinct_brands = _client.fetchall()
    _client.execute('''SELECT DISTINCT operating_system FROM products WHERE num_reviews > 30''')
    distinct_operating_systems = _client.fetchall()
    _client.execute('''SELECT min(price), max(price) FROM products WHERE num_reviews > 30''')
    min_max_price_tuple = _client.fetchone()
    _client.execute('''SELECT min(date_first_available_dt), max(date_first_available_dt) FROM products WHERE num_reviews > 30''')
    min_max_date_tuple = _client.fetchone()
    
    # Flatten the list of tuples
    distinct_categories = [item[0] for item in distinct_categories]
    distinct_brands = [item[0] for item in distinct_brands]
    distinct_operating_systems = [item[0] for item in distinct_operating_systems]
    
    st.session_state.distinct_categories = distinct_categories
    st.session_state.distinct_brands = distinct_brands
    st.session_state.distinct_operating_systems = distinct_operating_systems
    st.session_state.min_max_price_tuple = min_max_price_tuple
    st.session_state.min_max_date_tuple = min_max_date_tuple
# endregion functions

# region front-end
# Default to search page on app open
if 'page' not in st.session_state:
    st.session_state.page = 'search'

# If no user search query, get all products
if ('query_for_sorting' not in st.session_state) or st.session_state.query_for_sorting == '':
    get_all_tabular_categories(client)

# SIDEBAR -- disable submit button on product 'view' page
with st.sidebar:
    with st.expander("How to Use:", expanded=True):
        st.markdown('''* Enter a query and hover over the :grey_question: icon to see relevant reviews\n* View an item to chat with reviews and product specifications\n\n''')
    st.write('\n\n\n')
    
with st.sidebar.form(key='filter_form'):

    st.markdown('# Filter Products')
    st.markdown('\n\n\n\n')
    
    click_disabled = False
    if 'page' in st.session_state and st.session_state.page == 'view':
        click_disabled = True
        
    if ('query_for_sorting' not in st.session_state) or st.session_state.query_for_sorting == '':
        st.text_input("Enter your query: (optional)", key='query_for_sorting', disabled=click_disabled)
    else:
        st.text_input("Enter your query: (optional)", key='query_for_sorting', value=st.session_state.query_for_sorting, disabled=click_disabled)

    if ('category_multi_selection' not in st.session_state) or (not st.session_state.category_multi_selection):
        if 'distinct_categories' not in st.session_state:
            get_all_tabular_categories(client)
        category_multi_selection = st.multiselect('Product Category:', st.session_state.distinct_categories, default=st.session_state.distinct_categories, key="category_multi_selection", disabled=click_disabled)
        os_multi_selection = st.multiselect('Operating Systems:', st.session_state.distinct_operating_systems, default=st.session_state.distinct_operating_systems, key="os_multi_selection", disabled=click_disabled)
    else:
        category_multi_selection = st.multiselect('Product Category:', st.session_state.distinct_categories, default=st.session_state.category_multi_selection, key="category_multi_selection", disabled=click_disabled)
        os_multi_selection = st.multiselect('Operating Systems:', st.session_state.distinct_operating_systems, default=st.session_state.os_multi_selection, key="os_multi_selection", disabled=click_disabled)
        

    price_slider = st.slider(
        'Price range',
        min_value=0.0, 
        max_value=st.session_state.min_max_price_tuple[1],
        value=(st.session_state.min_max_price_tuple[0], st.session_state.min_max_price_tuple[1]),
        key="price_slider",
        disabled=click_disabled
    )

    rating_slider = st.slider(
        'Range for ratings (1-5)',
        min_value=1, 
        max_value=5,
        value=(1,5),
        key="rating_slider",
        disabled=click_disabled
    )
    
    with st.expander("More filters:"):
        if ('brand_multi_selection' not in st.session_state) or (not st.session_state.brand_multi_selection):
            brand_multi_selection = st.multiselect('Brands:', st.session_state.distinct_brands, default=st.session_state.distinct_brands, key="brand_multi_selection", disabled=click_disabled)
        else:
            brand_multi_selection = st.multiselect('Brands:', st.session_state.distinct_brands, default=st.session_state.brand_multi_selection, key="brand_multi_selection", disabled=click_disabled)
    
        st.markdown('\n\n\n\n')
        date_slider = st.slider(
            'Release date:',
            min_value=st.session_state.min_max_date_tuple[0], 
            max_value=st.session_state.min_max_date_tuple[1],
            value=(st.session_state.min_max_date_tuple[0], st.session_state.min_max_date_tuple[1]),
            key="date_slider",
            disabled=click_disabled
        )
    
    st.markdown('\n\n\n')
    st.form_submit_button(label='Search', on_click=recsys, disabled=click_disabled)

# HOME/SEARCH PAGE
if st.session_state.page == 'search' and ('product' not in st.session_state):
    st.session_state.from_reload = True
    recsys()

# PRODUCT PAGE
elif st.session_state.page == 'view':
    if 'product' in st.session_state and st.session_state['product'] in st.session_state.filtered_products_df['id'].values:
        view(st.session_state['product'], st.session_state.filtered_products_df)
# endregion front-end