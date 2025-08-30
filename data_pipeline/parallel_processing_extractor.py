# !curl -O https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/api_request_parallel_processor.py

import json
import pandas as pd
import openai
import asyncio
from typing import Any
import os

df = pd.read_excel("products.xlsx")

from api_request_parallel_processor import process_api_requests_from_file

openai.api_key = os.getenv("OPENAI_API_KEY")

# product_df = df[['id', 'brand', 'series', 'title_text']]
# with open('asin_title.jsonl', 'w') as jsonlfile:
#     for _, row in product_df.iterrows():
#         id = row['id']
#         brand = row['brand']
#         series = row['series']
#         title = row['title_text']
        
#         data = {
#             "id": id,
#             "BRAND": brand,
#             "SERIES": series,
#             "TITLE": title
#         }

#         jsonlfile.write(json.dumps(data) + '\n')

async def extract_succinct_title(product_text: str) -> Any:
    # Define the system message and user message
    system_message = {
        "role": "system",
        "content": (
            "You are a data cleaning specialist skilled in textual data manipulation and extraction. You have been tasked with extracting a succinct product title given a product's BRAND, SERIES, and current (improperly formatted) TITLE. "
            "The BRAND field is normally standardized, but the SERIES and TITLE fields are not. The current TITLE field will generally contain excess information that should not be in the product title field. "
            "Your job is to extract the appropriate new TITLE value from the provided fields and return a JSON lines entry with the result as a single string. The title you provide should be ultra succinct, but should uniquely identify the product if possible. "
            "***If possible, the string you return should be in the form [BRAND] [SERIES] [MODEL].***  Here are a few examples of good responses: "
            "* ['Data': {'id': '5', 'BRAND': 'Lenovo', 'SERIES': 'Flex 5 14', 'TITLE': 'Lenovo Flex 5 14 2-in-1 Laptop, 14.0\" FHD Touch Display, AMD Ryzen 5 4500U, 16GB RAM, 256GB Storage, AMD Radeon Graphics, Digital Pen Included, Win 10'}, 'Response': {'id': '5', 'true_title': 'Lenovo Flex 5'}] "
            "* ['Data': {'id': '55', 'BRAND': 'Skytech Gaming', 'SERIES': 'Azure', 'TITLE': 'Skytech Azure Gaming PC Desktop  Intel Core i5 11400F 2.6 GHz, NVIDIA RTX 3060, 500GB NVME SSD, 16GB DDR4 RAM 3200, 650W Gold PSU, 11AC Wi-Fi, Windows 11 Home 64-bit'}, 'Response': {'id': '55', 'true_title': 'Skytech Gaming Azure'}] "
            "* ['Data': {'id': '678', 'BRAND': 'Apple', 'SERIES': 'Apple Mac Mini', 'TITLE': '2018 Apple Mac Mini with 3.2GHz Intel Core i7 (32GB RAM, 128GB SSD) Space Gray (Renewed)'}, 'Response': {'id': '678', 'true_title': 'Apple Mac Mini 2018'}] "
            "* ['Data': {'id': '1', 'BRAND': 'HP', 'SERIES': 'TPN-I140', 'TITLE': 'HP 17 Laptop, 17.3\" Full HD Display, AMD Ryzen 7 5825U, 32GB RAM, 1TB PCIe SSD, Webcam, HDMI, Fingerprint Reader, Wi-Fi, Windows 11 Home'}, 'Response': {'id': '1', 'true_title': 'HP TPN-I140 17'}] "
            "***Critically - only respond with the requested information in the requested format (JSON lines) and no supplementary comments.***"
        ),
    }
    user_message = {"role": "user", 
                    "content": (
                        "Extract the appropriate new TITLE value from the provided fields and return it as a single string in a field called 'true_title'. "
                        "Return the result with the id as a valid JSON lines entry in the format "
                        '{"id": _, "true_title"": _}. '
                        ' Do NOT respond with anything supplementary commentary around the generated JSON lines.\nData: '
                    ) + product_text}
    
    # Create a single request JSON object for the Chat API
    request_json = {
        "model": "gpt-3.5-turbo",
        "messages": [system_message, user_message],
    }


    return request_json

async def main():
    file_path = "asin_title.jsonl"
    output_file_path = "results.jsonl"
    requests_filepath = "requests.jsonl"

    # Prepare the list of requests for the Chat API
    with open(file_path, 'r', encoding='utf-8') as file, open(requests_filepath, 'w', encoding='utf-8') as requests_file:
        num_processed = 0
        # only do first 10 lines as test
        for line in file:
            user_message_content = line            
            request_json = await extract_succinct_title(user_message_content)
            json.dump(request_json, requests_file, ensure_ascii=False)
            requests_file.write('\n')
            num_processed += 1

    # Process requests and save the results
    await process_api_requests_from_file(
        requests_filepath=requests_filepath,
        save_filepath=output_file_path,
        request_url="https://api.openai.com/v1/chat/completions",
        api_key=openai.api_key,
        max_requests_per_minute=908,  # Replace with your desired value or a fraction of your limit
        max_tokens_per_minute=6250000,  # Replace with your desired value or a fraction of your limit
        token_encoding_name='cl100k_base',  # Use the correct encoding name for the model engine
        max_attempts=20,
        logging_level=20  # Use an appropriate logging level; 20 (INFO) is the default
    )


# Run the main async function
asyncio.run(main())

## to get the results:
# title_dict = {}

# with open('results.jsonl', 'r', encoding='utf-8') as file:
#     num_processed = 0
#     for line in file:
#         # if num_processed > 10:
#         #     break
#         content = line            
#         ret_dict = ast.literal_eval(ast.literal_eval(line)[1]['choices'][0]['message']['content'].replace('\"\"', '\"')) # <--oops
#         # print(ret_dict['true_title'])
#         title_dict[ret_dict['id']] = ret_dict['true_title']
#         num_processed += 1