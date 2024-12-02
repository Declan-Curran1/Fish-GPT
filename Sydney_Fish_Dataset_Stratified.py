import os
import json
from typing import Any, Dict, List, Tuple
import random
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

class StratifiedSydneyFishDataset(Dataset):
    """
    PyTorch Dataset for Sydney Harbour Fish Species with stratified sampling.
    This ensures each species is represented in train/val/test splits.
    """

    def __init__(
        self,
        data_dir: str,  # Directory containing species folders
        split: str = "train",
        test_ratio: float = 0.5,  # Ratio of images to use for testing
        sort_json_key: bool = True,
        seed: int = 42
    ):
        super().__init__()
        
        print(f"Initializing dataset with directory: {data_dir}")
        
        self.split = split
        self.sort_json_key = sort_json_key
        self.data_dir = data_dir
        
        # Set random seed for reproducibility
        random.seed(seed)
        
        if not os.path.exists(data_dir):
            print(f"ERROR: Data directory {data_dir} does not exist")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Directory contents: {os.listdir('.')}")
            sys.exit(1)
            
        self.entries = []
        
        # Get all species folders
        species_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
        print(f"\nFound species folders: {species_folders}")
        
        if not species_folders:
            print(f"ERROR: No species folders found in {data_dir}")
            print(f"Directory contents: {os.listdir(data_dir)}")
            sys.exit(1)
        
        print(f"\nLoading images from species folders:")
        for species_folder in species_folders:
            species_path = os.path.join(data_dir, species_folder)
            image_files = [f for f in os.listdir(species_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            print(f"\n{species_folder} - Found {len(image_files)} images")
            
            # Randomly select test_ratio percentage of images
            num_images = max(1, int(len(image_files) * test_ratio))
            selected_images = random.sample(image_files, num_images)
            
            print(f"Selected {len(selected_images)} images for {split} set")
            
            for img_file in selected_images:
                img_path = os.path.join(species_path, img_file)
                print(f"  Loading {img_path}")
                
                try:
                    # Verify image can be opened
                    with Image.open(img_path) as img:
                        pass
                    
                    # Use actual species name from folder
                    ground_truth = {
                        "gt_parse": {
                            "species": {
                                "name": species_folder.replace("_", " ")  # Convert folder name to readable species name
                            }
                        }
                    }
                    
                    entry = {
                        "image_path": img_path,
                        "ground_truth": json.dumps(ground_truth)
                    }
                    self.entries.append(entry)
                except Exception as e:
                    print(f"ERROR loading {img_path}: {str(e)}")
        
        if not self.entries:
            print("ERROR: No valid images were loaded")
            sys.exit(1)
        
        # Shuffle the entries
        random.shuffle(self.entries)
        self.dataset_length = len(self.entries)
        
        print(f"\n{split} set contains {self.dataset_length} samples")
        
        # Process ground truths
        self.gt_token_sequences = []
        for entry in self.entries:
            ground_truth = json.loads(entry["ground_truth"])
            gt_jsons = [ground_truth["gt_parse"]]
            
            self.gt_token_sequences.append(
                [
                    self.json2token(
                        gt_json,
                        sort_json_key=self.sort_json_key,
                    )
                    for gt_json in gt_jsons
                ]
            )

    def get_species_distribution(self) -> Dict[str, int]:
        """Returns the distribution of species in the current split."""
        distribution = defaultdict(int)
        for entry in self.entries:
            ground_truth = json.loads(entry["ground_truth"])
            species = ground_truth["gt_parse"]["species"]["name"]
            distribution[species] += 1
        return dict(distribution)

    def json2token(self, obj: Any, sort_json_key: bool = True) -> str:
        """Convert JSON object into token sequence string."""
        if type(obj) == dict:
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                if sort_json_key:
                    keys = sorted(obj.keys(), reverse=True)
                else:
                    keys = obj.keys()
                for k in keys:
                    output += (
                        fr"<s_{k}>"
                        + self.json2token(obj[k], sort_json_key)
                        + fr"</s_{k}>"
                    )
                return output
        elif type(obj) == list:
            return r"<sep/>".join(
                [self.json2token(item, sort_json_key) for item in obj]
            )
        else:
            obj = str(obj)
            return obj

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Tuple[Image.Image, str]:
        """Returns one item of the dataset."""
        entry = self.entries[idx]
        
        # Load and convert image
        image = Image.open(entry["image_path"]).convert("RGB")
        
        # Get ground truth sequence
        target_sequence = random.choice(self.gt_token_sequences[idx])

        return image, target_sequence


def display_image_with_label(image: Image.Image, label: str, title: str):
    """Helper function to display an image with its label"""
    plt.figure(figsize=(10, 5))
    plt.imshow(image)
    plt.title(f"{title}\nLabel: {label}")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    print(f"Current working directory: {os.getcwd()}")
    print(f"Data directory exists: {os.path.exists('data-fish-test')}")
    print(f"Directory contents: {os.listdir('.')}")
    
    try:
        # Create dataset instance for test data
        test_dataset = StratifiedSydneyFishDataset(
            data_dir="data-fish-test",  # Directory containing species folders
            split="test",
            test_ratio=0.5,  # Use 50% of images from each species folder
            seed=42
        )
        
        # Print dataset statistics
        print("\nDataset Statistics:")
        print(f"Test samples: {len(test_dataset)}")
        
        # Print species distribution
        print("\nSpecies Distribution:")
        distribution = test_dataset.get_species_distribution()
        for species, count in distribution.items():
            print(f"{species}: {count} samples")
        
        # Display some examples
        print("\nExample entries:")
        for i in range(min(5, len(test_dataset))):
            image, target_sequence = test_dataset[i]
            print(f"\nSample {i+1}:")
            print(f"[INST] <image>\nExtract JSON [/INST] {target_sequence}")
            
            # Extract species name from target sequence
            import re
            species_name = re.search(r'<s_name>(.*?)</s_name>', target_sequence).group(1)
            
            # Display the image with its label
            display_image_with_label(image, species_name, f"Sample {i+1}")
            
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
