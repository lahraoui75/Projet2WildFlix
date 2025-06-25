import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Recommandation de Films", layout="wide")

# Authentification basique
users = {
    "user": {"mdp": "1234", "role": "utilisateur"},
    "admin": {"mdp": "adminpass", "role": "admin"}
}

st.sidebar.title("🔐 Connexion")
identifiant = st.sidebar.text_input("Identifiant")
mdp = st.sidebar.text_input("Mot de passe", type="password")
connecté = False
role = None

if identifiant in users and mdp == users[identifiant]["mdp"]:
    connecté = True
    role = users[identifiant]["role"]
    st.sidebar.success(f"Connecté en tant que **{role}**")
elif identifiant and mdp:
    st.sidebar.error("Identifiants incorrects")

# Si connecté, afficher l'app principale
if connecté:

    # En-tête
    st.image("https://github.com/lahraoui75/Projet2WildFlix/blob/main/image%20group%202.png?raw=true", use_column_width=True)
    st.title("🎬 Explorateur & Recommandateur de Films")
    st.markdown("Choisissez un film pour voir ses détails et découvrir des suggestions similaires.")

    @st.cache_data
    def load_data():
        df = pd.read_csv("https://github.com/lahraoui75/Projet2WildFlix/blob/main/df_final_20250625.csv")
        return df

    df = load_data()
    titre_col = 'Title' if 'Title' in df.columns else 'titre'

    features_similitude = [
        "nb_critiques_expert", "like_realisateur", "mots_cle-theme_encoding",
        "genre 1_encoding", "like_casting", "like_film",
        "acteur1_encoding", "acteur2_encoding", "acteur3_encoding",
        "Director1_encoding", "Writer1_encoding", "Title_encoding",
        "Language1_encoding", "Country1_encoding",
        "Rated_encoding", "imdbVotes", "Plot_encoding"
    ]

    similarity_matrix = cosine_similarity(df[features_similitude])

    film_selectionne = st.selectbox("🎞️ Sélectionnez un film :", df[titre_col].dropna().unique())
    film_info = df[df[titre_col] == film_selectionne]
    ligne = film_info.index[0] if not film_info.empty else None

    col_gauche, col_droite = st.columns([2, 1])

    with col_gauche:
        if ligne is not None:
            row = df.loc[ligne]
            st.markdown(f"## ⭐ {row[titre_col]} ({row['Year']})")

            col1, col2 = st.columns([1, 2])
            with col1:
                if pd.notna(row.get("Poster", None)):
                    st.image(row["Poster"], width=250)
            with col2:
                st.markdown(f"**Réalisateur :** {row['Director']}")
                st.markdown(f"**Acteurs :** {row['Actors']}")
                st.markdown(f"**Genre :** {row['Genre']}")
                st.markdown(f"**Durée :** {row['Runtime']}")
                st.markdown(f"**Note IMDB :** ⭐ {row['imdbRating']}")
                st.markdown(f"**Box Office :** {row['BoxOffice']}")
                st.markdown(f"**Résumé :** {row['Plot']}")

    with col_droite:
        def afficher_films_similaires(titre_film, n=5):
            try:
                index_film = df[df[titre_col] == titre_film].index[0]
                scores = list(enumerate(similarity_matrix[index_film]))
                scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
                indices = [i[0] for i in scores]
                similaires = df.loc[indices][[titre_col, "Year", "Poster", "imdbRating", "Genre", "Director"]]

                st.subheader("🎯 Films similaires")
                for _, film in similaires.iterrows():
                    subcol1, subcol2 = st.columns([1, 3])
                    with subcol1:
                        if pd.notna(film["Poster"]):
                            st.image(film["Poster"], width=100)
                    with subcol2:
                        st.markdown(f"**{film[titre_col]} ({film['Year']})** – ⭐ {film['imdbRating']}")
                        st.markdown(f"_Genre : {film['Genre']}  |  Réalisateur : {film['Director']}_")
                    st.markdown("---")
            except Exception as e:
                st.warning(f"Erreur : {e}")

        afficher_films_similaires(film_selectionne)

    # Zone Admin : accès exclusif
    if role == "admin":
        st.sidebar.markdown("---")
        st.sidebar.header("👑 Zone Admin")
        if st.sidebar.button("Aperçu des colonnes"):
            st.write("🧾 Colonnes du DataFrame :", df.columns.tolist())
        if st.sidebar.button("Afficher un échantillon"):
            st.dataframe(df.sample(5))
        if st.sidebar.checkbox("📥 Importer un nouveau CSV"):
            fichier = st.sidebar.file_uploader("Choisir un fichier CSV", type="csv")
            if fichier is not None:
                try:
                    nouveau_df = pd.read_csv(fichier)
                    st.success("✅ Fichier chargé avec succès !")
                    st.write(nouveau_df.head())
                except Exception as err:
                    st.error(f"❌ Erreur lors du chargement : {err}")

else:
    st.warning("🔐 Veuillez vous connecter pour accéder à l'application.")


