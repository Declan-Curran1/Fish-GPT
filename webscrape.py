import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import csv
import time
import os
import shutil
from google_images_search import GoogleImagesSearch

# ... [Previous code remains unchanged] ...

##########

# EXPANDED VERSION

def scrape_fish_species_info_expanded():
    base_url = 'https://australian.museum'
    url = 'https://australian.museum/learn/animals/fishes/fishes-of-sydney-harbour/'
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    # Create Scraped_Fish_Information folder
    output_folder = 'Scraped_Fish_Information'
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        table = soup.find('table')
        fish_list = []

        for row in table.find_all('tr'):
            cells = row.find_all('td')
            if len(cells) >= 3:
                fish_name_cell = cells[2]
                link_tag = fish_name_cell.find('a')
                
                if link_tag:
                    fish_name = link_tag.get_text(strip=True)
                    relative_link = link_tag['href']
                    link = urljoin(base_url, relative_link)
                    
                    # Create folder for each fish species
                    fish_folder = os.path.join(output_folder, fish_name.replace(' ', '_'))
                    os.makedirs(fish_folder, exist_ok=True)
                    
                    # Scrape fish page
                    fish_response = requests.get(link, headers=headers)
                    fish_soup = BeautifulSoup(fish_response.content, 'html.parser')
                    
                    # Extract introduction and identification
                    introduction = ''
                    identification = ''
                    h4_tags = fish_soup.find_all('h4')
                    for h4 in h4_tags:
                        heading = h4.get_text(strip=True)
                        if heading == 'Introduction':
                            introduction = extract_section_content(h4)
                        elif heading == 'Identification':
                            identification = extract_section_content(h4)
                    
                    # Scrape image from fish page
                    img_tag = fish_soup.find('img', {'class': 'figure__image'})
                    if img_tag and 'src' in img_tag.attrs:
                        img_url = urljoin(base_url, img_tag['src'])
                        save_image(img_url, os.path.join(fish_folder, 'main_image.jpg'))
                    
                    # Google image search and scrape
                    google_image_search(fish_name, fish_folder)
                    
                    fish_list.append({
                        'Fish Name': fish_name,
                        'Link': link,
                        'Identification': identification,
                        'Introduction': introduction
                    })
                    
                    # Sleep to avoid overwhelming the server
                    time.sleep(1)
        
        # Save to a CSV file
        with open(os.path.join(output_folder, 'fish_list_extended.csv'), 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Fish Name', 'Link', 'Identification', 'Introduction']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for fish in fish_list:
                writer.writerow(fish)
        
        return fish_list
    except Exception as e:
        print(f"Scraping failed: {e}")
        return []

def extract_section_content(h4_tag):
    content = ''
    sibling = h4_tag.find_next_sibling()
    while sibling and sibling.name != 'h4':
        if sibling.name == 'p':
            content += sibling.get_text(separator=' ', strip=True) + ' '
        sibling = sibling.find_next_sibling()
    return content.strip()

def save_image(url, file_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)

def google_image_search(query, output_folder):
    # Replace with your own API key and CX
    gis = GoogleImagesSearch('AIzaSyAO3Yrza37O9yfHOmvZI62skGNvQkSMcIk', 'a79096328085f4a61') #GoogleImagesSearch('YOUR_API_KEY', 'YOUR_CX')
    
    params = {
        'q': query,
        'num': 20,
        'fileType': 'jpg',
        'rights': 'cc_publicdomain|cc_attribute|cc_sharealike|cc_noncommercial|cc_nonderived'
    }
    
    gis.search(search_params=params, path_to_dir=output_folder)

if __name__ == "__main__":
    fish_species_info = scrape_fish_species_info_expanded()
    print(f"Scraped information for {len(fish_species_info)} fish species.")
