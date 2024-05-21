import requests
from bs4 import BeautifulSoup
import csv
import os

pdf_directory = "/home/hb/bgp_papers"

def parse_page(url, csv_writer):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the <dl> element containing the paper entries
    dl_element = soup.find('dl')
    
    if dl_element is None:
        print("No entries on this page.")
        return

    # Iterate through each <dt> and <dd> pair within the <dl> element
    for dt, dd in zip(dl_element.find_all('dt'), dl_element.find_all('dd')):
        # Extract the arXiv identifier and link to the paper page
        identifier_tag = dt.find('span', class_='list-identifier')
        identifier_link = identifier_tag.find('a')['href']
        paper_url = f'https://arxiv.org{identifier_link}'
        # print(f"paper url: {paper_url}")

        paper_response = requests.get(paper_url)
        paper_soup = BeautifulSoup(paper_response.content, 'html.parser')

        # Extract the title and abstract
        title_tag = paper_soup.find('h1', class_='title mathjax')
        if title_tag is not None:
            title = title_tag.text.strip().replace('Title:', '').strip()
        else:
            print("No title")
            title = "N/A"

        abstract_tag = paper_soup.find('blockquote', class_='abstract mathjax')
        if abstract_tag is not None:
            abstract = abstract_tag.text.strip().replace('Abstract:', '').strip()
        else:
            print("No abstract")
            abstract = "N/A"
            
        if "BGP" in abstract or "Border Gateway Protocol" in abstract:
            print("BGP related title:", title)
            print("BGP related abstract:", abstract)
            csv_writer.writerow({'Title': title, 'Abstract': abstract})
            # Download PDF file and save it in the specified directory
            pdf_url = paper_url.replace('/abs/', '/pdf/') + '.pdf'
            pdf_response = requests.get(pdf_url)
            if pdf_response.status_code == 200:
                pdf_content = pdf_response.content
                pdf_filename = os.path.join(pdf_directory, f"{title.replace(' ', '_')}.pdf")
                with open(pdf_filename, "wb") as pdf_file:
                    pdf_file.write(pdf_content)
                print(f"Downloaded PDF: {pdf_filename}")
            else:
                print(f"Failed to download PDF. Status code: {pdf_response.status_code}")

        else:
            print("No BGP related papers")

base_url = 'https://arxiv.org/list/cs.NS/'

start_year = 2000
end_year = 2023

with open('bgp_papers_GT.csv', mode='w', newline='') as csv_file:
    fieldnames = ['Title', 'Abstract']
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            year_month = f'{year % 100:02d}{month:02d}'
            print(f"Parsing year: {year}, month: {month}")
            
            url = f'{base_url}{year_month}?skip=0&show=25'
            parse_page(url, csv_writer)
