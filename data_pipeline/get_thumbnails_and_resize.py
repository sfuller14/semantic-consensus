import requests
from tqdm import tqdm

for id, url in zip(import_product_df.asin, import_product_df.thumbnailImage):
    picture_name = f'{id}.jpg'
    response = requests.get(url, stream=True)

    file_size = int(response.headers.get("Content-Length", 0))
    progress = tqdm(response.iter_content(1024), f"Downloading {picture_name}", total=file_size, unit="B", unit_scale=True, unit_divisor=1024)
    
    os.makedirs('assets', exist_ok=True)
    with open(os.path.join('assets', picture_name), 'wb') as f:
        for data in progress.iterable:
            f.write(data)
            progress.update(len(data))

from PIL import Image

def get_most_common_size(directory):
    sizes = {}
    for filename in os.listdir(directory):
        with Image.open(os.path.join(directory, filename)) as img:
            size = img.size
            if size in sizes:
                sizes[size] += 1
            else:
                sizes[size] = 1
    most_common_size = max(sizes, key=sizes.get)
    return most_common_size

def resize_image(input_image_path, output_image_path, size):
    original_image = Image.open(input_image_path)
    width, height = original_image.size
    max_width, max_height = size
    # calculate ratio
    ratio = min(max_width/width, max_height/height)
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    resized_image = original_image.resize((new_width, new_height))
    
    # create new image with white background
    new_image = Image.new("RGB", size, (255, 255, 255))
    # paste resized image into new image
    upper_left = (max_width-new_width)//2, (max_height-new_height)//2
    new_image.paste(resized_image, upper_left)

    new_image.save(output_image_path)

# get the most common size
directory1 = "assets"
directory2 = "assets_resized"
most_common_size = get_most_common_size(directory1)

# resize all images to most common size
for filename in os.listdir(directory1):
    resize_image(os.path.join(directory1, filename), os.path.join(directory2, filename), most_common_size)
