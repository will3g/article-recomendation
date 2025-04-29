import os
import requests

from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

_pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("INDEX_REGION"))
index = _pinecone.Index(os.getenv("INDEX_NAME"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_text_embedding(text):
    try:
        response = client.embeddings.create(model="text-embedding-ada-002", input=text)
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def generate_sitemap_urls(base_url, start_date, days):
    sitemap_urls = []
    current_date = start_date
    for _ in range(days):
        url = f"{base_url}/{current_date.year}/{current_date.strftime('%m')}/{current_date.strftime('%d')}_1.xml"
        sitemap_urls.append(url)
        current_date -= timedelta(days=1)
    return sitemap_urls

def extract_news_data_from_sitemap(sitemap_url):
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content)
        news_data = []
        for url in soup.find_all("url"):
            loc = url.find("loc").text
            lastmod = url.find("lastmod").text if url.find("lastmod") else None
            if loc and lastmod:
                news_data.append((loc, lastmod))
        return news_data
    except Exception as e:
        print(f"Error accessing {sitemap_url}: {e}")
        return []

def convert_to_timestamp(lastmod):
    try:
        dt = datetime.fromisoformat(lastmod.replace("Z", "+00:00"))
        return int(dt.timestamp())
    except Exception as e:
        print(f"Error converting date: {e}")
        return None

def scrape_article_content(article_url):
    try:
        response = requests.get(article_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        article = soup.find("article")
        if article:
            return article.get_text(strip=True)
        else:
            return None
    except Exception as e:
        print(f"Error accessing {article_url}: {e}")
        return None

# exemplo de url: https://oglobo.globo.com/sitemap/oglobo/2024/12/07_1.xml
base_url = "https://oglobo.globo.com/sitemap/oglobo"
start_date = datetime(2025, 4, 8)
days_to_scrape = 7

sitemap_urls = generate_sitemap_urls(base_url, start_date, days_to_scrape)

for sitemap_url in sitemap_urls:
    print(f"Processing sitemap: {sitemap_url}")
    news_data = extract_news_data_from_sitemap(sitemap_url)
    for news_url, lastmod in news_data:
        print(f"Scraping article: {news_url}")
        content = scrape_article_content(news_url)

        if content:
            embedding = get_text_embedding(content)
            if embedding:
                timestamp = convert_to_timestamp(lastmod)
                article_id = news_url.split('/')[-1]
                index.upsert(
                    vectors=[
                        {
                            "id": article_id,
                            "values": embedding,
                            "metadata": {
                                "url": news_url,
                                "content": content,
                                "date": timestamp
                            }
                        }
                    ],
                    namespace="news_namespace" # namespace do indice
                )
                print(f"Inserted article: {news_url}")
