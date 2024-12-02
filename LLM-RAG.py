


import ollama
import os

# List of images to pass
images = ['./Data/Gordons-21-09/LLM/fish_240/frame_806.jpg', 
          './Data/Gordons-21-09/LLM/fish_240/frame_807.jpg', 
          './Data/Gordons-21-09/LLM/fish_240/frame_808.jpg']

res = ollama.chat(
    model='llava:13b',
    messages=[
        {'role': 'user',
         'content': 'You are an assistant marine biologist who is taking detailed notes to then be passed to your senior. Describe the fish in this image IN DETAIL, focus on their exact features, any colours, body shape etc. - your notes will be passed to your senior who will need to identify the fish. The images are taken in and around Sydney Harbour (Shallow water). You have been passed several images but they are all of the same fish, just at different times ',
        'images': ['./Data/Gordons-21-09/LLM/fish_240/frame_806.jpg']
        }
    ],
    options={
        'num_gpu_layers': 1  # Adjust based on your GPU's VRAM
    }
)

print(res['message']['content'])


#########################################
#TESTING IF INCREASING NUM GPU LAYERS HELPS AT ALL AND IF THERE IS TRADEOFF
import ollama
import pandas as pd
import time
import subprocess

# List of images to pass
images = [
    './Data/Gordons-21-09/LLM/fish_240/frame_806.jpg',
    './Data/Gordons-21-09/LLM/fish_240/frame_807.jpg',
    './Data/Gordons-21-09/LLM/fish_240/frame_808.jpg'
]

# Define the prompt
prompt = (
    'You are an assistant marine biologist who is taking detailed notes to then be passed to your senior. '
    'Describe the fish in this image IN DETAIL, focus on their exact features, any colours, body shape etc. '
    '- your notes will be passed to your senior who will need to identify the fish. The images are taken in '
    'and around Sydney Harbour (Shallow water). You have been passed several images but they are all of the '
    'same fish, just at different times.'
)

# Initialize a list to store results
results = []

max_execution_time = 300  # Maximum time in seconds (e.g., 5 minutes)
max_retries = 1  # Maximum number of retries per num_gpu_layers value

for num_layers in range(1, 11):
    print(f"Running with num_gpu_layers={num_layers}")
    retries = 0
    success = False
    while retries <= max_retries and not success:
        start_time = time.time()
        try:
            res = ollama.chat(
                model='llava:13b',
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                        'images': images
                    }
                ],
                options={
                    'num_gpu_layers': num_layers
                },
                keep_alive=0  # Unload the model after use
            )
            end_time = time.time()
            elapsed_time = end_time - start_time

            # Append the result
            results.append({
                'num_gpu_layers': num_layers,
                'time_taken_sec': elapsed_time,
                'status': 'Success'
            })
            print(f"num_gpu_layers={num_layers} completed in {elapsed_time:.2f} seconds.\n")
            success = True  # Exit the retry loop since the operation was successful

        except Exception as e:
            end_time = time.time()
            elapsed_time = end_time - start_time

            # Check if execution time exceeded
            if elapsed_time > max_execution_time:
                status_message = f'Failed: Execution time exceeded {max_execution_time} seconds'
                print(f"num_gpu_layers={num_layers} failed after {elapsed_time:.2f} seconds with error: {status_message}")
                # Attempt to stop the model
                print("Attempting to stop the model...")
                try:
                    subprocess.run(["ollama", "stop", "llava:13b"], check=True)
                    print("Model llava:13b stopped successfully.")
                except subprocess.CalledProcessError as e_sub:
                    print(f"Error stopping model: {e_sub}")
                # Increment retry counter
                retries += 1
                if retries > max_retries:
                    # Append the result
                    results.append({
                        'num_gpu_layers': num_layers,
                        'time_taken_sec': elapsed_time,
                        'status': status_message
                    })
                    print(f"Max retries exceeded for num_gpu_layers={num_layers}. Moving to next value.\n")
            else:
                status_message = f'Failed: {str(e)}'
                # Append the result
                results.append({
                    'num_gpu_layers': num_layers,
                    'time_taken_sec': elapsed_time,
                    'status': status_message
                })
                print(f"num_gpu_layers={num_layers} failed after {elapsed_time:.2f} seconds with error: {status_message}\n")
                success = True  # Exit the retry loop since it's a different exception

# Create a DataFrame from the results
df = pd.DataFrame(results)

# Optionally, set the 'num_gpu_layers' as integer type
df['num_gpu_layers'] = df['num_gpu_layers'].astype(int)

# Print the DataFrame
print("\nTiming Results:")
print(df)



################
#take 2

import ollama
import pandas as pd
import time
import subprocess
import multiprocessing

# Define the function at the top level
def run_ollama_chat(num_layers, prompt, images, result_dict):
    """
    Executes the ollama.chat() function with specified parameters.
    Records the time taken and status in the shared result_dict.
    """
    start_time = time.time()
    try:
        res = ollama.chat(
            model='llava:13b',
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                    'images': images
                }
            ],
            options={
                'num_gpu_layers': num_layers
            },
            keep_alive=0  # Unload the model after use
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        result_dict['time_taken_sec'] = elapsed_time
        result_dict['status'] = 'Success'
        print(f"num_gpu_layers={num_layers} completed in {elapsed_time:.2f} seconds.\n")
    except Exception as e:
        end_time = time.time()
        elapsed_time = end_time - start_time
        result_dict['time_taken_sec'] = elapsed_time
        result_dict['status'] = f'Failed: {str(e)}'
        print(f"num_gpu_layers={num_layers} failed after {elapsed_time:.2f} seconds with error: {str(e)}\n")

def stop_model(model_name):
    """
    Attempts to stop the specified Ollama model.
    """
    try:
        subprocess.run(["ollama", "stop", model_name], check=True)
        print(f"Model {model_name} stopped successfully.")
    except subprocess.CalledProcessError as e_sub:
        print(f"Error stopping model: {e_sub}")

"""
Main function to iterate through num_gpu_layers values, execute ollama.chat(),
handle timeouts, and record results.
"""
# List of images to pass
images = [
    './Data/Gordons-21-09/LLM/fish_240/frame_806.jpg',
    './Data/Gordons-21-09/LLM/fish_240/frame_807.jpg',
    './Data/Gordons-21-09/LLM/fish_240/frame_808.jpg'
]

# Define the prompt
prompt = (
    'You are an assistant marine biologist who is taking detailed notes to then be passed to your senior. '
    'Describe the fish in this image IN DETAIL, focus on their exact features, any colours, body shape etc. '
    '- your notes will be passed to your senior who will need to identify the fish. The images are taken in '
    'and around Sydney Harbour (Shallow water). You have been passed several images but they are all of the '
    'same fish, just at different times.'
)

# Initialize a list to store results
results = []

max_execution_time = 300  # Maximum time in seconds (e.g., 5 minutes)
max_retries = 1  # Maximum number of retries per num_gpu_layers value

for num_layers in range(1, 11):
    print(f"Running with num_gpu_layers={num_layers}")
    retries = 0
    success = False
    while retries <= max_retries and not success:
        # Create a manager for shared data
        manager = multiprocessing.Manager()
        result_dict = manager.dict()

        # Initialize and start the process
        p = multiprocessing.Process(target=run_ollama_chat, args=(num_layers, prompt, images, result_dict))
        p.start()

        # Wait for the process to complete or timeout
        p.join(timeout=max_execution_time)

        if p.is_alive():
            # The process is still running; terminate it
            print(f"num_gpu_layers={num_layers} failed: Execution time exceeded {max_execution_time} seconds.")
            p.terminate()
            p.join()

            # Attempt to stop the model to free resources
            print("Attempting to stop the model...")
            stop_model("llava:13b")

            # Increment retry counter
            retries += 1
            if retries > max_retries:
                # Record the failure due to timeout
                results.append({
                    'num_gpu_layers': num_layers,
                    'time_taken_sec': max_execution_time,
                    'status': f'Failed: Execution time exceeded {max_execution_time} seconds'
                })
                print(f"Max retries exceeded for num_gpu_layers={num_layers}. Moving to next value.\n")
            else:
                print(f"Retrying num_gpu_layers={num_layers}...\n")
        else:
            # Process finished within timeout
            results.append({
                'num_gpu_layers': num_layers,
                'time_taken_sec': result_dict.get('time_taken_sec', max_execution_time),
                'status': result_dict.get('status', 'Unknown')
            })
            if result_dict.get('status') == 'Success':
                success = True
            else:
                retries += 1
                if retries > max_retries:
                    print(f"Max retries exceeded for num_gpu_layers={num_layers}. Moving to next value.\n")
                else:
                    print(f"Retrying num_gpu_layers={num_layers}...\n")

# Create a DataFrame from the results
df = pd.DataFrame(results)

# Optionally, set the 'num_gpu_layers' as integer type
df['num_gpu_layers'] = df['num_gpu_layers'].astype(int)

# Print the DataFrame
print("\nTiming Results:")
print(df)

######################
#Workflow

######################




#####################
#PORTION HERE FOR LLM + RAG
#####################


#from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"

import cv2
import os
from ultralytics import YOLO

import os
import cv2
import numpy as np

def extract_frames_per_fish_id(model, video_path, output_folder, already_processed='F'):
    fish_frames = {}
    
    # If frames are already processed, read from output_folder
    if already_processed == 'T':
        # Check the output folder for processed images
        for fish_id_folder in os.listdir(output_folder):
            fish_folder_path = os.path.join(output_folder, fish_id_folder)
            if os.path.isdir(fish_folder_path):
                fish_id = int(fish_id_folder.split('_')[1])  # Assuming folder name format: 'fish_<id>'
                fish_frames[fish_id] = []
                for frame_file in sorted(os.listdir(fish_folder_path)):
                    frame_path = os.path.join(fish_folder_path, frame_file)
                    fish_frames[fish_id].append(frame_path)
        return fish_frames
    
    # If not processed, extract frames using YOLO model tracking
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Error opening video file {video_path}"
    
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Resize frame to reduce memory consumption (adjust scale as needed)
        frame = cv2.resize(frame, (320, 180))  # Example resolution reduction

        results = model.track(frame, persist=True, classes=[0], conf=0.6)

        if results and len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
            boxes = results[0].boxes
            if boxes.id is not None:
                ids = boxes.id.cpu().numpy()
                for i, id_ in enumerate(ids):
                    id_ = int(id_)
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                    fish_crop = frame[y1:y2, x1:x2]
                    fish_folder = os.path.join(output_folder, f"fish_{id_}")
                    os.makedirs(fish_folder, exist_ok=True)
                    frame_filename = os.path.join(fish_folder, f"frame_{frame_count}.jpg")
                    cv2.imwrite(frame_filename, fish_crop)
                    if id_ not in fish_frames:
                        fish_frames[id_] = []
                    fish_frames[id_].append(frame_filename)
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    return fish_frames




def load_fish_species_info_from_csv(csv_file_path):
    fish_species_info = []
    with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Combine Identification and Introduction for description
            description = f"Identification: {row['Identification']}\nIntroduction: {row['Introduction']}"
            fish_species_info.append({
                'name': row['Fish Name'],
                'description': description
            })
    return fish_species_info


from sentence_transformers import SentenceTransformer
import numpy as np

def index_species_info(fish_species_info):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    species_embeddings = []
    for species in fish_species_info:
        embedding = embedder.encode(species['description'])
        species['embedding'] = embedding
        species_embeddings.append(embedding)
    return fish_species_info, np.vstack(species_embeddings), embedder



from PIL import Image
'''
def generate_captions_for_images(model, processor, image_paths):
    captions = []
    for image_path in image_paths:
        image = Image.open(image_path).convert('RGB')
        inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
        with torch.no_grad():
            output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)
        captions.append(caption)
    return captions

'''

def retrieve_relevant_species(captions, fish_species_info, species_embeddings, embedder, top_k=5):
    retrieved_species = []
    for caption in captions:
        caption_embedding = embedder.encode(caption)
        similarities = np.dot(species_embeddings, caption_embedding)
        top_indices = np.argsort(similarities)[-top_k:]  # Get top_k species
        top_species = [fish_species_info[i] for i in reversed(top_indices)]
        retrieved_species.extend(top_species)
    # Remove duplicates
    unique_species = {species['name']: species for species in retrieved_species}
    return list(unique_species.values())





from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers import AutoModelForCausalLM, AutoTokenizer
'''
def load_local_llm():
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
    llm_model = AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B')
    llm_model.to(device)
    llm_model.eval()
    return llm_model, tokenizer

'''
'''
def load_local_llm():
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
    llm_model = AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B')
    llm_model.to('cpu')
    llm_model.eval()
    return llm_model, tokenizer
'''
'''
def load_local_llm():
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    llm_model = AutoModelForCausalLM.from_pretrained('gpt2')
    llm_model.to('cpu')  # Explicitly move the model to CPU
    llm_model.eval()
    return llm_model, tokenizer

'''

#gpt2
'''
def load_local_llm():
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    llm_model = AutoModelForCausalLM.from_pretrained('gpt2')
    # Set pad_token_id to eos_token_id
    tokenizer.pad_token_id = tokenizer.eos_token_id
    llm_model.to('cpu')
    llm_model.eval()
    return llm_model, tokenizer


'''

#gpt-neo


def load_local_llm():
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
    llm_model = AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B')
    # Set pad_token_id to eos_token_id
    tokenizer.pad_token_id = tokenizer.eos_token_id
    llm_model.to('cpu')
    llm_model.eval()
    return llm_model, tokenizer


###
#TRying for even more context
'''

def load_local_llm():
    tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_13b")
    model = AutoModelForCausalLM.from_pretrained("openlm-research/open_llama_13b", max_position_embeddings=4096)  # Extended context length)
    # Set pad_token_id to eos_token_id
    tokenizer.pad_token_id = tokenizer.eos_token_id
    llm_model.to('cpu')
    llm_model.eval()
    return llm_model, tokenizer

'''

def load_local_llm():
    tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_13b", use_fast=False)
    llm_model = AutoModelForCausalLM.from_pretrained("openlm-research/open_llama_13b")

    # Set pad_token_id to eos_token_id
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Move the model to CPU
    llm_model.to('cpu')
    llm_model.eval()

    return llm_model, tokenizer


import ollama
def generate_captions_for_images_ollama(image_paths):
    captions = []
    for image_path in image_paths:
        res = ollama.chat(
            model='llava-13b',
            messages=[
                {
                    'role': 'user',
                    'content': (
                        'You are an assistant marine biologist who is taking detailed notes to then be passed to your senior. '
                        'Describe the fish in this image IN DETAIL, focusing on their exact features, any colours, body shape, etc. '
                        'Your notes will be passed to your senior who will need to identify the fish.'
                    ),
                    'images': [image_path]
                }
            ]
        )
        caption = res['message']['content']
        captions.append(caption)
    return captions


def generate_species_prediction(llm_model, tokenizer, captions, retrieved_species, max_tokens=2048):
    # Prepare the prompt
    context = "\n\n".join([f"{species['name']}:\n{species['description']}" for species in retrieved_species])
    observations = "\n".join([f"- {caption}" for caption in captions])
    prompt = (
        f"You are a marine biologist familiar with fish species in Sydney Harbour.\n\n"
        f"Observations:\n{observations}\n\n"
        f"Known Species Information:\n{context}\n\n"
        f"Based on the observations and known species information, identify the fish species seen in the observations. "
        f"Provide the species name and a brief justification."
    )

    # Tokenize the prompt
    inputs = tokenizer.encode(prompt, return_tensors='pt')

    # Truncate if exceeds max tokens
    max_length = llm_model.config.max_position_embeddings - 150  # Reserve tokens for generation
    if inputs.size(1) > max_length:
        inputs = inputs[:, -max_length:]

    # Move inputs to device
    inputs = inputs.to(llm_model.device)

    # Generate prediction
    with torch.no_grad():
        outputs = llm_model.generate(
            inputs,
            max_new_tokens=150,
            num_beams=5,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id
        )
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return prediction



from tqdm import tqdm
import os

def identify_fish_species_per_id(fish_frames, fish_species_info, species_embeddings, embedder, llm_model, tokenizer, output_folder):
    fish_id_to_species = {}

    # Add progress bar using tqdm
    total_fish = len(fish_frames)
    progress_bar = tqdm(total=total_fish, desc="Processing Fish IDs", unit="fish")

    for fish_id, image_paths in fish_frames.items():
        # Select up to 5 random frames
        if len(image_paths) > 5:
            selected_images = random.sample(image_paths, 5)
        else:
            selected_images = image_paths
        
        # Generate captions
        captions = generate_captions_for_images_ollama(selected_images)
        #captions = generate_captions_for_images(model, processor, selected_images)
        
        # Retrieve relevant species
        retrieved_species = retrieve_relevant_species(captions, fish_species_info, species_embeddings, embedder)
        
        # Generate species prediction
        prediction = generate_species_prediction(llm_model, tokenizer, captions, retrieved_species)
        
        # Save the prediction to the fish_frames dictionary
        fish_id_to_species[fish_id] = prediction
        
        # Write the output incrementally for each fish ID
        fish_folder = os.path.join(output_folder, f"fish_{fish_id}")
        os.makedirs(fish_folder, exist_ok=True)
        output_file = os.path.join(fish_folder, "prediction.txt")
        with open(output_file, 'w') as f:
            f.write(f"Fish ID: {fish_id}\n")
            f.write(f"Prediction: {prediction}\n")

        # Update the progress bar
        progress_bar.update(1)
    
    progress_bar.close()  # Close the progress bar when done
    return fish_id_to_species







def update_tracking_video_with_species_labels(video_path, output_video_path, fish_id_to_species, model):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Error opening video file {video_path}"
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    video_writer = cv2.VideoWriter(
        output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        results = model.track(frame, persist=True, classes=[0], conf=0.6)
        
        if results and len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
            boxes = results[0].boxes
            if boxes.id is not None:
                ids = boxes.id.cpu().numpy()
                for i, id_ in enumerate(ids):
                    id_ = int(id_)
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                    label = fish_id_to_species.get(id_, "Fish")
                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                    )
        video_writer.write(frame)
    
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

def load_fish_species_info_from_csv(csv_file_path):
    fish_species_info = []
    with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Combine Identification and Introduction for description
            description = f"Identification: {row['Identification']}\nIntroduction: {row['Introduction']}"
            fish_species_info.append({
                'name': row['Fish Name'],
                'description': description
            })
    return fish_species_info

#Semantic Similarity Retrieval w/ RAG
# Assuming this is already done
fish_species_info, species_embeddings, embedder = index_species_info(fish_species_info)




import os
import random
import torch
from ultralytics import YOLO








# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"
'''processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
blip_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float16
)
blip_model.to(device)
blip_model.eval()'''


embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load sentence embedder
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Load LLM
llm_model, tokenizer = load_local_llm()

# Load YOLO model
model_path = 'C:/Users/decla/Object Detection/runs/detect/yolov8n_custom74/weights/last.pt'
model = YOLO(model_path)
model.to(device)

# Paths
video_path = 'C:/Users/decla/OneDrive/Desktop/Dean-Vid.mp4'
output_folder = 'C:/Users/decla/Object Detection/Data/Gordons-21-09/LLM'
output_video_path = 'C:/Users/decla/Object Detection/Data/Gordons-21-09/LLM/path_to_updated_video.mp4'

torch.cuda.empty_cache()

Already_processed = 'T'

# Extract frames per fish ID
fish_frames = extract_frames_per_fish_id(model, video_path, output_folder, Already_processed)

# Load fish species information from CSV
csv_file_path = 'C:/Users/decla/fish_list_extended.csv'
fish_species_info = load_fish_species_info_from_csv(csv_file_path)

# Index species information
fish_species_info, species_embeddings, embedder = index_species_info(fish_species_info)

# Identify fish species per ID
fish_id_to_species = identify_fish_species_per_id(
    fish_frames, fish_species_info, species_embeddings, embedder, llm_model, tokenizer, output_folder
)

# Update tracking video with new fish labels
update_tracking_video_with_species_labels(video_path, output_video_path, fish_id_to_species, model)

# Output fish ID to species mapping
for fish_id, species_info in fish_id_to_species.items():
    print(f"Fish ID {fish_id}: {species_info}")

'''
llm_model, tokenizer = load_local_llm()

# Load YOLO model
model_path = 'C:/Users/decla/Object Detection/runs/detect/yolov8n_custom74/weights/last.pt'  #'path_to_your_yolo_model.pt'
model = YOLO(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



#path = 'C:/Users/decla/Object Detection/Data/Gordons-21-09/LLM/path_to_updated_video.mp4'
video_path = 'C:/Users/decla/OneDrive/Desktop/Dean-Vid.mp4'
output_folder = 'C:/Users/decla/Object Detection/Data/Gordons-21-09/LLM'
output_video_path = 'C:/Users/decla/Object Detection/Data/Gordons-21-09/LLM/path_to_updated_video.mp4'

torch.cuda.empty_cache()

Already_processed = 'T' #If you've already done this, it saves time to just read from the pre-written files than running again
# Extract frames per fish ID
fish_frames = extract_frames_per_fish_id(model, video_path, output_folder,Already_processed)
import csv
# Load fish species information from CSV
csv_file_path = 'C:/Users/decla/fish_list_extended.csv'  # Update with your CSV file path
fish_species_info = load_fish_species_info_from_csv(csv_file_path)

# Index species information
fish_species_info, species_embeddings, embedder = index_species_info(fish_species_info)

# Identify fish species per ID
fish_id_to_species = identify_fish_species_per_id(
    fish_frames, fish_species_info, species_embeddings, embedder, llm_model, tokenizer, output_folder
)


# Update tracking video with new fish labels
update_tracking_video_with_species_labels(video_path, output_video_path, fish_id_to_species, model)

# Output fish ID to species mapping
for fish_id, species_info in fish_id_to_species.items():
    print(f"Fish ID {fish_id}: {species_info}")

'''















































####################################
#Get single prediction 
####################################



import os
import random
import torch
from ultralytics import YOLO
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import ollama
import numpy as np
from tqdm import tqdm
import csv

# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"
'''
# Load LLM
def load_local_llm():
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
    llm_model = AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    llm_model.to('cpu')
    llm_model.eval()
    return llm_model, tokenizer

llm_model, tokenizer = load_local_llm()
'''
'''def load_local_llm():
    tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_13b", use_fast=False)
    llm_model = AutoModelForCausalLM.from_pretrained("openlm-research/open_llama_13b")
    # Set pad_token_id to eos_token_id
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # Move the model to CPU
    llm_model.to('cpu')
    llm_model.eval()
    return llm_model, tokenizer
'''

'''
def load_local_llm():
    from transformers import LlamaForCausalLM, LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained("openlm-research/open_llama_13b")
    llm_model = LlamaForCausalLM.from_pretrained("openlm-research/open_llama_13b")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    llm_model.to('cuda')
    llm_model.eval()
    return llm_model, tokenizer
'''
def load_local_llm():
    from transformers import LlamaForCausalLM, LlamaTokenizer
    from accelerate import init_empty_weights, infer_auto_device_map
    import torch

    tokenizer = LlamaTokenizer.from_pretrained("openlm-research/open_llama_13b")
    with init_empty_weights():
        llm_model = LlamaForCausalLM.from_pretrained("openlm-research/open_llama_13b")

    # Ensure CUDA is available and set memory constraints
    if torch.cuda.is_available():
        device_map = infer_auto_device_map(llm_model, max_memory={'cuda': '7GB', 'cpu': '15GB'})
    else:
        device_map = infer_auto_device_map(llm_model, max_memory={'cpu': '15GB'})

    # Load the model with device map and memory offloading
    llm_model = llm_model.from_pretrained(
        "openlm-research/open_llama_13b",
        device_map=device_map,
        offload_folder='offload',  # Folder to store offloaded weights
        offload_state_dict=True
    )

    tokenizer.pad_token_id = tokenizer.eos_token_id
    llm_model.eval()
    return llm_model, tokenizer


# Load YOLO model
model_path = 'C:/Users/decla/Object Detection/runs/detect/yolov8n_custom74/weights/last.pt'
model = YOLO(model_path)
model.to(device)

# Paths
video_path = 'C:/Users/decla/OneDrive/Desktop/Dean-Vid.mp4'
output_folder = 'C:/Users/decla/Object Detection/Data/Gordons-21-09/LLM'
output_video_path = 'C:/Users/decla/Object Detection/Data/Gordons-21-09/LLM/path_to_updated_video.mp4'

torch.cuda.empty_cache()

Already_processed = 'T'

# Function to extract frames per fish ID (Assuming this function is already defined)

def extract_frames_per_fish_id(model, video_path, output_folder, already_processed='F'):
    fish_frames = {}
    
    # If frames are already processed, read from output_folder
    if already_processed == 'T':
        # Check the output folder for processed images
        for fish_id_folder in os.listdir(output_folder):
            fish_folder_path = os.path.join(output_folder, fish_id_folder)
            if os.path.isdir(fish_folder_path):
                fish_id = int(fish_id_folder.split('_')[1])  # Assuming folder name format: 'fish_<id>'
                fish_frames[fish_id] = []
                for frame_file in sorted(os.listdir(fish_folder_path)):
                    frame_path = os.path.join(fish_folder_path, frame_file)
                    fish_frames[fish_id].append(frame_path)
        return fish_frames
    
    # If not processed, extract frames using YOLO model tracking
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Error opening video file {video_path}"
    
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Resize frame to reduce memory consumption (adjust scale as needed)
        frame = cv2.resize(frame, (320, 180))  # Example resolution reduction

        results = model.track(frame, persist=True, classes=[0], conf=0.6)

        if results and len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
            boxes = results[0].boxes
            if boxes.id is not None:
                ids = boxes.id.cpu().numpy()
                for i, id_ in enumerate(ids):
                    id_ = int(id_)
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                    fish_crop = frame[y1:y2, x1:x2]
                    fish_folder = os.path.join(output_folder, f"fish_{id_}")
                    os.makedirs(fish_folder, exist_ok=True)
                    frame_filename = os.path.join(fish_folder, f"frame_{frame_count}.jpg")
                    cv2.imwrite(frame_filename, fish_crop)
                    if id_ not in fish_frames:
                        fish_frames[id_] = []
                    fish_frames[id_].append(frame_filename)
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    return fish_frames

# Load fish species information from CSV
def load_fish_species_info_from_csv(csv_file_path):
    fish_species_info = []
    with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            description = f"Identification: {row['Identification']}\nIntroduction: {row['Introduction']}"
            fish_species_info.append({
                'name': row['Fish Name'],
                'description': description
            })
    return fish_species_info

csv_file_path = 'C:/Users/decla/fish_list_extended.csv'
fish_species_info = load_fish_species_info_from_csv(csv_file_path)

# Index species information
def index_species_info(fish_species_info):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    species_embeddings = []
    for species in fish_species_info:
        embedding = embedder.encode(species['description'])
        species['embedding'] = embedding
        species_embeddings.append(embedding)
    return fish_species_info, np.vstack(species_embeddings), embedder

fish_species_info, species_embeddings, embedder = index_species_info(fish_species_info)

# Extract frames per fish ID
fish_frames = extract_frames_per_fish_id(model, video_path, output_folder, Already_processed)

# Updated function to generate captions using Ollama
def generate_captions_for_images_ollama(selected_images):
    res = ollama.chat(
        model='llava:13b',
        messages=[
            {
                'role': 'user',
                'content': (
                    'You are an assistant marine biologist taking detailed notes to be passed to your senior. '
                    'The images are taken in and around Sydney Harbour (Shallow water). You have been passed several images, '
                    'but they are all of the same fish, just at different times.'
                    '\n\nBefore you begin, note that distinguishing fish species often relies on specific features such as dorsal fin shape, '
                    'primary and secondary colors, fin colors, gill shape, body patterns, and unique markings.'
                    '\n\nPlease provide a detailed description of the fish, focusing on the following checklist of features relevant for fish identification:'
                    '\n- Dorsal fin shape'
                    '\n- Primary color'
                    '\n- Secondary color'
                    '\n- Fin color'
                    '\n- Gill shape'
                    '\n- Body shape'
                    '\n- Unique markings or patterns'
                    '\n\nYour notes will be passed to your senior who will need to identify the fish.'
                ),
                'images': selected_images
            }
        ]
    )
    caption = res['message']['content']
    return [caption]

# Function to retrieve relevant species
def retrieve_relevant_species(captions, fish_species_info, species_embeddings, embedder, top_k=5):
    combined_caption = " ".join(captions)
    caption_embedding = embedder.encode(combined_caption)
    similarities = np.dot(species_embeddings, caption_embedding)
    similarities /= np.linalg.norm(species_embeddings, axis=1)
    similarities /= np.linalg.norm(caption_embedding)
    top_indices = np.argsort(similarities)[-top_k:]
    retrieved_species = [fish_species_info[i] for i in reversed(top_indices)]
    return retrieved_species
def generate_species_prediction(llm_model, tokenizer, captions, retrieved_species, max_tokens=4096):
    # Tokenizer and max_token considerations
    eos_token_id = tokenizer.eos_token_id
    tokenizer.pad_token_id = eos_token_id  # Ensure pad_token_id is set
    
    # Prepare the context with a prompt containing retrieved species info
    context = "\n\n".join([f"{species['name']}:\n{species['description']}" for species in retrieved_species])
    observations = "\n".join([f"- {caption}" for caption in captions])
    
    # Create the full prompt for the LLM
    prompt = (
        f"You are a marine biologist familiar with fish species in Sydney Harbour.\n\n"
        f"Observations:\n{observations}\n\n"
        f"Known Species Information:\n{context}\n\n"
        f"Based on the observations and known species information, identify the fish species seen in the observations. "
        f"Provide the species name and a brief justification."
    )

    # Tokenize the prompt and calculate the total length
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    # Calculate how many tokens are needed for the input and how much space we have for generation
    input_length = inputs.size(1)
    available_tokens = max_tokens - input_length

    # If the input exceeds the maximum token limit, truncate context to fit within limit
    if available_tokens < 0:
        truncated_species = len(retrieved_species) - 1
        while input_length > max_tokens and truncated_species > 0:
            retrieved_species = retrieved_species[:truncated_species]  # Remove some species
            context = "\n\n".join([f"{species['name']}:\n{species['description']}" for species in retrieved_species])
            prompt = (
                f"You are a marine biologist familiar with fish species in Sydney Harbour.\n\n"
                f"Observations:\n{observations}\n\n"
                f"Known Species Information:\n{context}\n\n"
                f"Based on the observations and known species information, identify the fish species seen in the observations. "
                f"Provide the species name and a brief justification."
            )
            inputs = tokenizer.encode(prompt, return_tensors='pt')
            input_length = inputs.size(1)
            truncated_species -= 1
    
    # Pass the prompt to the LLM with the attention mask
    attention_mask = inputs.ne(tokenizer.pad_token_id).long()
    inputs = inputs.to(llm_model.device)
    attention_mask = attention_mask.to(llm_model.device)

    # Generate the prediction
    with torch.no_grad():
        outputs = llm_model.generate(
            inputs,
            attention_mask=attention_mask,  # Explicitly passing attention mask
            max_new_tokens=available_tokens,
            num_beams=5,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode the output prediction
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return prediction



# Function to identify fish species per ID
def identify_fish_species_per_id(fish_frames, fish_species_info, species_embeddings, embedder, llm_model, tokenizer, output_folder):
    fish_id_to_species = {}
    total_fish = len(fish_frames)
    progress_bar = tqdm(total=total_fish, desc="Processing Fish IDs", unit="fish")

    for fish_id, image_paths in fish_frames.items():
        # Select up to 5 random frames
        if len(image_paths) > 5:
            selected_images = random.sample(image_paths, 5)
        else:
            selected_images = image_paths

        # Generate captions using Ollama
        captions = generate_captions_for_images_ollama(selected_images)

        # Retrieve relevant species
        retrieved_species = retrieve_relevant_species(captions, fish_species_info, species_embeddings, embedder)

        # Generate species prediction
        prediction = generate_species_prediction(llm_model, tokenizer, captions, retrieved_species)

        # Save the prediction
        fish_id_to_species[fish_id] = prediction
        fish_folder = os.path.join(output_folder, f"fish_{fish_id}")
        os.makedirs(fish_folder, exist_ok=True)
        output_file = os.path.join(fish_folder, "prediction.txt")
        with open(output_file, 'w') as f:
            f.write(f"Fish ID: {fish_id}\n")
            f.write(f"Prediction: {prediction}\n")

        progress_bar.update(1)

    progress_bar.close()
    return fish_id_to_species


# Single-use run for a specific fish ID
fish_id = 240  # Replace with your desired fish ID
image_paths = fish_frames[fish_id]

def identify_fish_species_for_single_id(
    fish_id, image_paths, fish_species_info, species_embeddings, embedder, llm_model, tokenizer, output_folder
):
    # Select up to 5 random frames
    if len(image_paths) > 1:
        selected_images = random.sample(image_paths, 2)
    else:
        selected_images = image_paths

    # Generate captions using Ollama
    captions = generate_captions_for_images_ollama(selected_images)

    # Retrieve relevant species
    retrieved_species = retrieve_relevant_species(captions, fish_species_info, species_embeddings, embedder)

    # Generate species prediction
    prediction = generate_species_prediction(llm_model, tokenizer, captions, retrieved_species)

    # Save the prediction
    fish_folder = os.path.join(output_folder, f"fish_{fish_id}")
    os.makedirs(fish_folder, exist_ok=True)
    output_file = os.path.join(fish_folder, "prediction.txt")
    with open(output_file, 'w') as f:
        f.write(f"Fish ID: {fish_id}\n")
        f.write(f"Prediction: {prediction}\n")

    print(f"Fish ID {fish_id}: {prediction}")
    return prediction

# Load LLM
llm_model, tokenizer = load_local_llm()

# Run the single-use function
prediction = identify_fish_species_for_single_id(
    fish_id, image_paths, fish_species_info, species_embeddings, embedder, llm_model, tokenizer, output_folder
)







#MULITPLE FISH ID PREDCITION 
######################
# Identify fish species per ID
#RUN FOR FULL VIDEO
fish_id_to_species = identify_fish_species_per_id(
    fish_frames, fish_species_info, species_embeddings, embedder, llm_model, tokenizer, output_folder
)

# Function to update tracking video (Assuming this function is already defined)
# def update_tracking_video_with_species_labels(...)

# Update tracking video with new fish labels
update_tracking_video_with_species_labels(video_path, output_video_path, fish_id_to_species, model)

# Output fish ID to species mapping
for fish_id, species_info in fish_id_to_species.items():
    print(f"Fish ID {fish_id}: {species_info}")

##########################




























######################
#PREDICT VIA NEW OLLAMA--LLAVA ONLY PROCESS
######################




####################################
#Get single prediction 
####################################



import os
import random
import torch
from ultralytics import YOLO
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import ollama
import numpy as np
from tqdm import tqdm
import csv

# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load YOLO model
model_path = 'C:/Users/decla/Object Detection/runs/detect/yolov8n_custom74/weights/last.pt'
model = YOLO(model_path)
model.to(device)

# Paths
video_path = 'C:/Users/decla/OneDrive/Desktop/Dean-Vid.mp4'
output_folder = 'C:/Users/decla/Object Detection/Data/Gordons-21-09/LLM'
output_video_path = 'C:/Users/decla/Object Detection/Data/Gordons-21-09/LLM/path_to_updated_video.mp4'

torch.cuda.empty_cache()

Already_processed = 'T'

# Function to extract frames per fish ID (Assuming this function is already defined)

def extract_frames_per_fish_id(model, video_path, output_folder, already_processed='F'):
    fish_frames = {}
    
    # If frames are already processed, read from output_folder
    if already_processed == 'T':
        # Check the output folder for processed images
        for fish_id_folder in os.listdir(output_folder):
            fish_folder_path = os.path.join(output_folder, fish_id_folder)
            if os.path.isdir(fish_folder_path):
                fish_id = int(fish_id_folder.split('_')[1])  # Assuming folder name format: 'fish_<id>'
                fish_frames[fish_id] = []
                for frame_file in sorted(os.listdir(fish_folder_path)):
                    frame_path = os.path.join(fish_folder_path, frame_file)
                    fish_frames[fish_id].append(frame_path)
        return fish_frames
    
    # If not processed, extract frames using YOLO model tracking
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Error opening video file {video_path}"
    
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Resize frame to reduce memory consumption (adjust scale as needed)
        frame = cv2.resize(frame, (320, 180))  # Example resolution reduction

        results = model.track(frame, persist=True, classes=[0], conf=0.6)

        if results and len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
            boxes = results[0].boxes
            if boxes.id is not None:
                ids = boxes.id.cpu().numpy()
                for i, id_ in enumerate(ids):
                    id_ = int(id_)
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                    fish_crop = frame[y1:y2, x1:x2]
                    fish_folder = os.path.join(output_folder, f"fish_{id_}")
                    os.makedirs(fish_folder, exist_ok=True)
                    frame_filename = os.path.join(fish_folder, f"frame_{frame_count}.jpg")
                    cv2.imwrite(frame_filename, fish_crop)
                    if id_ not in fish_frames:
                        fish_frames[id_] = []
                    fish_frames[id_].append(frame_filename)
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    return fish_frames

# Load fish species information from CSV
def load_fish_species_info_from_csv(csv_file_path):
    fish_species_info = []
    with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            description = f"Identification: {row['Identification']}\nIntroduction: {row['Introduction']}"
            fish_species_info.append({
                'name': row['Fish Name'],
                'description': description
            })
    return fish_species_info

csv_file_path = 'C:/Users/decla/fish_list_extended.csv'
fish_species_info = load_fish_species_info_from_csv(csv_file_path)

# Index species information
def index_species_info(fish_species_info):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    species_embeddings = []
    for species in fish_species_info:
        embedding = embedder.encode(species['description'])
        species['embedding'] = embedding
        species_embeddings.append(embedding)
    return fish_species_info, np.vstack(species_embeddings), embedder

fish_species_info, species_embeddings, embedder = index_species_info(fish_species_info)

# Extract frames per fish ID
fish_frames = extract_frames_per_fish_id(model, video_path, output_folder, Already_processed)
def generate_species_prediction_with_llava(selected_images, fish_species_info):
    # Construct the species information context
    species_context = "\n\n".join([f"{species['name']}:\n{species['description']}" for species in fish_species_info])

    # Construct the prompt
    prompt = (
        'You are an assistant marine biologist tasked with identifying fish species from images. '
        'The images are taken in and around Sydney Harbour (shallow water). '
        'Please analyze the images and, using the provided species information, identify the fish species seen in the images. '
        'Provide the species name and a brief justification based on the observed features.'
        '\n\nSpecies Information:\n'
        f'{species_context}'
    )

    # Call LLAVA with the prompt and images
    res = ollama.chat(
        model='llava:13b',
        messages=[
            {
                'role': 'user',
                'content': prompt,
                'images': selected_images
            }
        ]
    )
    prediction = res['message']['content'] #res['choices'][0]['content']
    return prediction

# Function to retrieve relevant species
def retrieve_relevant_species(captions, fish_species_info, species_embeddings, embedder, top_k=5):
    combined_caption = " ".join(captions)
    caption_embedding = embedder.encode(combined_caption)
    similarities = np.dot(species_embeddings, caption_embedding)
    similarities /= np.linalg.norm(species_embeddings, axis=1)
    similarities /= np.linalg.norm(caption_embedding)
    top_indices = np.argsort(similarities)[-top_k:]
    retrieved_species = [fish_species_info[i] for i in reversed(top_indices)]
    return retrieved_species


def identify_fish_species_for_single_id(
    fish_id, image_paths, fish_species_info, output_folder
):
    # Select up to 5 random frames
    if len(image_paths) > 5:
        selected_images = random.sample(image_paths, 5)
    else:
        selected_images = image_paths

    # Generate species prediction using LLAVA
    prediction = generate_species_prediction_with_llava(selected_images, fish_species_info)

    # Save the prediction
    fish_folder = os.path.join(output_folder, f"fish_{fish_id}")
    os.makedirs(fish_folder, exist_ok=True)
    output_file = os.path.join(fish_folder, "prediction.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Fish ID: {fish_id}\n")
        f.write(f"Prediction: {prediction}\n")

    print(f"Fish ID {fish_id}: {prediction}")
    return prediction

# Single-use run for a specific fish ID
fish_id = 240  # Replace with your desired fish ID
image_paths = fish_frames[fish_id]


# Run the single-use function
# Run the single-use function
prediction = identify_fish_species_for_single_id(
    fish_id, image_paths, fish_species_info, output_folder
)





