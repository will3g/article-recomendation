import os
import re
import math
import numpy as np

from datetime import datetime, timedelta
from collections import defaultdict
from dotenv import load_dotenv
from openai import OpenAI
from pinecone.grpc import PineconeGRPC as Pinecone
from sklearn.cluster import KMeans

load_dotenv()

_pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("INDEX_REGION"))
index = _pinecone.Index(os.getenv("INDEX_NAME"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

LAMBDA_FACTOR = 0.01
INITIAL_DATE_LIMIT = 14
MAX_DATE_LIMIT = 30

def extract_categories_from_source(source_url):
    pattern = r".*.com\/(?P<categories>[^\/]+(?:\/[^\/]+)*)\/\d{4}\/\d{2}"
    match = re.search(pattern, source_url)
    if match:
        categories = match.group('categories').split('/')
        return categories
    else:
        return []

def adjust_score_by_date(original_score, date_str, current_date_str):
    current_date = datetime.strptime(current_date_str, "%Y-%m-%d")
    try:
        # Tentar interpretar como ISO-8601
        document_date = datetime.strptime(date_str, "%Y-%m-%d")
    except:
        try:
            # Tentar interpretar como timestamp
            document_date = datetime.fromtimestamp(int(date_str))
        except:
            return original_score
    days_difference = (current_date - document_date).days
    time_decay = math.exp(-LAMBDA_FACTOR * days_difference)
    adjusted_score = original_score * time_decay
    return adjusted_score

def filter_documents_by_date(documents, current_date_str, date_limit_days):
    filtered_documents = []
    current_date = datetime.strptime(current_date_str, "%Y-%m-%d")
    date_limit = current_date - timedelta(days=date_limit_days)
    for doc in documents:
        document_date_str = doc['metadata'].get('date', '')
        if not document_date_str:
            continue
        try:
            try:
                document_date = datetime.strptime(document_date_str, "%Y-%m-%d")
            except:
                document_date = datetime.fromtimestamp(int(document_date_str))
            if document_date >= date_limit:
                filtered_documents.append(doc)
        except:
            continue
    return filtered_documents

def filter_already_read_articles(results, user_history_urls):
    filtered = []
    for match in results:
        source_url = match['metadata'].get('url', '')
        if source_url not in user_history_urls:
            filtered.append(match)
    return filtered

def dynamic_search(query_embedding, current_date_str, initial_date_limit, max_date_limit):
    date_limit = initial_date_limit
    all_search_results = []
    while date_limit <= max_date_limit:
        search_results = index.query(
            vector=query_embedding.tolist(),
            top_k=5,
            namespace="news_namespace",
            include_metadata=True,
            include_values=True
        )
        all_search_results.extend(search_results['matches'])
        filtered_results = filter_documents_by_date(search_results['matches'], current_date_str, date_limit)
        if filtered_results:
            return filtered_results, all_search_results
        date_limit *= 2
    return [], all_search_results

def adjust_score_by_source(original_score, document_categories, user_categories, category_counter):
    matching_categories = set(document_categories).intersection(set(user_categories))
    penalty = sum([category_counter[category] for category in matching_categories])
    if matching_categories:
        adjusted_score = original_score * (1 + 0.1 * len(matching_categories)) / (1 + 0.01 * penalty)
    else:
        adjusted_score = original_score
    return adjusted_score

def get_embedding_from_url(url):
    embedding_dimension = 1536
    dummy_vector = [0] * embedding_dimension
    response = index.query(
        vector=dummy_vector,
        top_k=5,
        namespace="news_namespace",
        include_values=True,
        filter={"url": url}
    )
    if response['matches']:
        match = response['matches'][0]
        embedding = match['values']
        return embedding
    else:
        return None

def cluster_embeddings(embeddings, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)
    cluster_centers = kmeans.cluster_centers_
    return cluster_centers

def is_duplicate_url(url, recommended_urls):
    return url in recommended_urls

def get_recommendations(user_urls):
    user_history_urls = []
    user_history_categories = []

    for input_url in user_urls:
        user_history_urls.append(input_url)
        categories = extract_categories_from_source(input_url)
        user_history_categories.extend(categories)

    user_history_categories = list(set(user_history_categories))

    user_embeddings = []
    for url in user_urls:
        embedding = get_embedding_from_url(url)
        if embedding is not None:
            user_embeddings.append(embedding)

    if not user_embeddings:
        return [], {}, {}

    num_clusters = min(len(user_embeddings), len(user_history_categories))
    cluster_centers = cluster_embeddings(user_embeddings, num_clusters)

    all_clusters_embeddings = {}
    cluster_results = []
    current_date_str = datetime.now().strftime("%Y-%m-%d")

    for i, center in enumerate(cluster_centers):
        filtered_results, search_results = dynamic_search(center, current_date_str, INITIAL_DATE_LIMIT, MAX_DATE_LIMIT)
        cluster_results.append(filtered_results)

        user_articles_embeddings = []
        recommended_articles_embeddings = []
        other_articles_embeddings = []

        for match in search_results:
            if match['metadata'].get('url', '') in user_history_urls:
                user_articles_embeddings.append(match['values'])
            elif match in filtered_results:
                recommended_articles_embeddings.append(match['values'])
            else:
                other_articles_embeddings.append(match['values'])

        all_clusters_embeddings[f"cluster_{i+1}"] = {
            "user_articles_embeddings": np.array(user_articles_embeddings) if user_articles_embeddings else np.array([]),
            "recommended_articles_embeddings": np.array(recommended_articles_embeddings) if recommended_articles_embeddings else np.array([]),
            "other_articles_embeddings": np.array(other_articles_embeddings) if other_articles_embeddings else np.array([])
        }

    cluster_size = min(len(cluster) for cluster in cluster_results if len(cluster) > 0) if cluster_results else 0
    balanced_results = []
    if cluster_size > 0:
        for i in range(cluster_size):
            for cluster in cluster_results:
                if i < len(cluster):
                    balanced_results.append(cluster[i])
    balanced_results = filter_already_read_articles(balanced_results, user_history_urls)

    adjusted_results = []
    category_counter = defaultdict(int)
    unwanted_terms = ['/blogs/', '/kogut/', '/post/', '/videos/', '/galeria/', '/comentarios/']

    recommended_urls = []

    for match in balanced_results:
        original_score = match['score']
        document_date = match['metadata'].get('date', '')
        source_url = match['metadata'].get('url', '')

        if any(term in source_url for term in unwanted_terms):
            continue

        if is_duplicate_url(source_url, recommended_urls):
            continue

        document_categories = extract_categories_from_source(source_url)

        adjusted_score = adjust_score_by_date(original_score, document_date, current_date_str)
        adjusted_score = adjust_score_by_source(adjusted_score, document_categories, user_history_categories, category_counter)

        for category in document_categories:
            category_counter[category] += 1

        adjusted_results.append({
            'adjusted_score': adjusted_score,
            'url': source_url,
            'date': document_date,
            'categories': document_categories
            
        })

        if source_url not in user_history_urls:
            recommended_urls.append(source_url)

    adjusted_results = sorted(adjusted_results, key=lambda x: x['date'], reverse=True)
    recommended_categories = dict(sorted(category_counter.items(), key=lambda item: item[1], reverse=True))

    return adjusted_results, recommended_categories, all_clusters_embeddings

if __name__ == "__main__":
    user_urls = [
        "https://oglobo.globo.com/rio/noticia/2024/12/07/primaverao-com-mais-de-41oc-rio-tem-terceiro-dia-mais-quente-do-ano-veja-a-previsao.ghtml",
        "https://oglobo.globo.com/politica/noticia/2024/12/07/lula-parabeniza-janja-apos-homenagem-em-sp-reconhecida-por-seu-trabalho-e-incansavel-empenho.ghtml",
        "https://oglobo.globo.com/ela/gente/noticia/2024/12/07/ana-hickmann-afirma-que-descobriu-nova-divida-milionaria-contraida-em-seu-nome-pelo-ex-marido.ghtml",
        "https://oglobo.globo.com/esportes/noticia/2024/12/07/atacante-do-west-ham-sofre-grave-acidente-de-carro-na-inglaterra-estado-de-saude-e-estavel.ghtml",
        "https://oglobo.globo.com/brasil/noticia/2024/12/07/homem-e-preso-suspeito-de-matar-esposa-para-receber-seguro-de-vida-de-r-1-milhao-em-minas-gerais.ghtml"
    ]
    recommendations, categories, embeddings = get_recommendations(user_urls)
    print('-' * 50)
    print(recommendations)
    print('-' * 50)
    print(categories)
    print('-' * 50)
