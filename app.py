import model
import numpy as np
import plotly.express as px
import streamlit as st

from sklearn.decomposition import PCA


st.title("Recomendação de Notícias Personalizadas")
st.info("""
    Ess projeto tem como objetivo demonstrar um sistema de recomendação de notícias personalizadas
    utilizando técnicas de IA (embeddings), clustering (kmeans) e banco vetorial (Pinecone).
""")

st.header("Insira as URLs das matérias:")
st.warning("As matérias que você selecionar devem estar no banco vetorial (Pinecone)")

user_input = st.text_area("Cole as URLs aqui (uma por linha)", height=150)
example_urls = """
    https://oglobo.globo.com/politica/noticia/2024/12/07/lula-parabeniza-janja-apos-homenagem-em-sp-reconhecida-por-seu-trabalho-e-incansavel-empenho.ghtml
    https://oglobo.globo.com/ela/gente/noticia/2024/12/07/ana-hickmann-afirma-que-descobriu-nova-divida-milionaria-contraida-em-seu-nome-pelo-ex-marido.ghtml
    https://oglobo.globo.com/esportes/noticia/2024/12/07/atacante-do-west-ham-sofre-grave-acidente-de-carro-na-inglaterra-estado-de-saude-e-estavel.ghtml
    https://oglobo.globo.com/brasil/noticia/2024/12/07/homem-e-preso-suspeito-de-matar-esposa-para-receber-seguro-de-vida-de-r-1-milhao-em-minas-gerais.ghtml
"""
st.info(f"Exemplo de utilização (estas estão no banco vetorial): \n {example_urls}")

user_urls = user_input.split('\n')
user_urls = [url.strip() for url in user_urls if url.strip()]

if st.button("Obter Recomendações"):
    if user_urls:
        st.write("Processando as recomendações...")

        recommended_articles, recommended_categories, all_clusters_embeddings = model.get_recommendations(user_urls)

        if recommended_articles:
            st.header("Artigos Recomendados:")
            for article in recommended_articles:
                st.write(f"Matéria: {article['url']}")
                st.write(f"Categoria(s): {', '.join(article['categories'])}")
                st.write(f"Publicado em: {article['date']}")
                st.write(f"Pontuação Ajustada: {article['adjusted_score']:.4f}")
                st.markdown("---")
        else:
            st.write("Nenhuma recomendação encontrada para os URLs fornecidos.")

        if recommended_categories:
            st.header("Categorias Recomendadas com base no seu histórico:")
            for category, count in recommended_categories.items():
                st.write(f"{category}: {count}")

        def concatenate_all_embeddings(cluster_embeddings):
            user_embeddings = []
            recommended_embeddings = []
            other_embeddings = []
            labels = []
            cluster_labels = []

            for cluster_name, cluster_data in cluster_embeddings.items():
                # Extrair o nome do cluster (ex: "cluster_1" -> "Cluster 1")
                cluster_id = cluster_name.replace('_', ' ').title()

                if cluster_data['user_articles_embeddings'].size > 0:
                    user_embeddings.append(cluster_data['user_articles_embeddings'])
                    labels += ['Matéria (Lido pelo Usuário)'] * len(cluster_data['user_articles_embeddings'])
                    cluster_labels += [cluster_id] * len(cluster_data['user_articles_embeddings'])

                if cluster_data['recommended_articles_embeddings'].size > 0:
                    recommended_embeddings.append(cluster_data['recommended_articles_embeddings'])
                    labels += ['Recomendação'] * len(cluster_data['recommended_articles_embeddings'])
                    cluster_labels += [cluster_id] * len(cluster_data['recommended_articles_embeddings'])

                if cluster_data['other_articles_embeddings'].size > 0:
                    other_embeddings.append(cluster_data['other_articles_embeddings'])
                    labels += ['Outras matérias'] * len(cluster_data['other_articles_embeddings'])
                    cluster_labels += [cluster_id] * len(cluster_data['other_articles_embeddings'])

            all_embeddings = np.vstack(user_embeddings + recommended_embeddings + other_embeddings)
            return all_embeddings, labels, cluster_labels

        all_embeddings, all_labels, all_cluster_labels = concatenate_all_embeddings(all_clusters_embeddings)

        # Mapeamento de símbolos para cada tipo de matéria
        symbol_map = {
            'Matéria (Lido pelo Usuário)': 'star',
            'Recomendação': 'circle',
            'Outras matérias': 'triangle-up'
        }

        # Definindo cores para cada cluster (exemplo para até 4 clusters)
        cluster_colors = {
            'Cluster 1': '#018cff',  # azul
            'Cluster 2': '#14c21d',  # verde
            'Cluster 3': '#ff0000',  # vermelho
            'Cluster 4': '#fdd663'   # amarelo
        }

        st.subheader("Gráfico de dispersão dos documentos no banco vetorial:")
        st.info("""
            **PCA (Principal Component Analysis)**: técnica linear de redução de dimensionalidade que projeta os dados em 
            um espaço de menor dimensão, maximizando a variância.
        """)
        reducer_pca = PCA(n_components=2, random_state=0)
        vectors_pca_2d = reducer_pca.fit_transform(np.array(all_embeddings))

        fig_pca = px.scatter(
            x=vectors_pca_2d[:, 0],
            y=vectors_pca_2d[:, 1],
            color=all_cluster_labels,      # Cores para os clusters
            symbol=all_labels,             # Símbolos para o tipo de matéria
            symbol_map=symbol_map,
            color_discrete_map=cluster_colors,  # Aplicando as cores definidas para clusters
            title="Documentos no Banco Vetorial (PCA)",
            labels={"x": "Dimensão 1", "y": "Dimensão 2"}
        )

        st.plotly_chart(fig_pca)

    else:
        st.write("Por favor, insira pelo menos uma URL para continuar.")
