from openai import OpenAI
import glob
from tqdm import tqdm
import json
import os
import base64

key = "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"
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=key,
)

folders = glob.glob("/home/user1/dataset/combined/train/*")

metadata_file = "metadata.json"
metadata = {}
if os.path.exists(metadata_file):
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    print(f"Loaded existing metadata for {len(metadata)} folders")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

for folder in tqdm(folders):
    folder_name = folder.split('/')[-1]
    
    if folder_name in metadata:
        print(f"Skipping {folder_name} - already processed")
        continue
    
    imgs = glob.glob(folder + '/*')
    
    if not imgs: 
        print(f"No images found in {folder_name}")
        continue
    
    image_path = imgs[0]
    base64_image = encode_image(image_path)

    try:
        completion = client.chat.completions.create(        
            model="qwen/qwen2.5-vl-72b-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Please analyze this image and provide information about the animal in the following exact format:

    animal type: {cat or dog}
    breed classify: {specific breed name}
    age type: {baby or adult or senior}
    description: {brief description of the animal's appearance and characteristics}"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        )
        metadata[folder_name] = completion.choices[0].message.content
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
            
    except Exception as e:
        print(f"Error processing {folder_name}: {str(e)}")
        continue

with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=4)

print(f"Processing complete. Metadata saved for {len(metadata)} folders.")

