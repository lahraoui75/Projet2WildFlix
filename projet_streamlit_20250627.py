import streamlit as st
import pandas as pd
import json
import os
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity

if "show_registration" not in st.session_state:
    st.session_state.show_registration = False


# Configuration de la page
st.set_page_config(page_title="Recommandation de Films", layout="wide")



# Données utilisateurs simulées
users = {
    "user": {"mdp": "1234", "role": "utilisateur"},
    "admin": {"mdp": "adminpass", "role": "admin"}
}



# Initialisation des états de session
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "role" not in st.session_state:
    st.session_state.role = None

# Sidebar : Connexion ou Déconnexion
st.sidebar.title("🔐 Connexion")



if st.session_state.logged_in:
    st.sidebar.success(f"Connecté en tant que **{st.session_state.role}**")
    if st.sidebar.button("Se déconnecter"):
        st.session_state.logged_in = False
        st.session_state.role = None
        st.rerun()
else:
    identifiant = st.sidebar.text_input("Identifiant")
    mdp = st.sidebar.text_input("Mot de passe", type="password")

    if identifiant in users and mdp == users[identifiant]["mdp"]:
        st.session_state.logged_in = True
        st.session_state.role = users[identifiant]["role"]
        st.rerun()
    elif identifiant and mdp:
        st.sidebar.error("Identifiants incorrects")

if os.path.exists("db.csv")== False:

    df = pd.DataFrame([["user", "12345678","utilisateur"],
                   ["admin", "adminpass",  "admin"]], columns=["name", "mdp", "role"])
    df.to_csv("db.csv",index=False)


st.title("Inscription")


username=st.text_input("indiquez votre username",key="username")
mdp=st.text_input("indiquez votre MDP",type ="password",key="mdp")
mdp2=st.text_input("confirmez votre MDP",type="password",key="mdp2")

if st.button("Valider votre inscription",key="inscription"):
    if len(username) == 0 or len(mdp)==0 or len(mdp2)== 0:
        st.write("les informations doivent être completées")
    elif username.lower() in pd.read_csv("db.csv").name.values:
        st.write("l' username indiquée est deja pris")
    else :
        if ( mdp != mdp2) or (len(mdp2)<8) or  (len(mdp)<8):
            st.write("les mots de passes ne correspondent pas ou bien leur taille sont inferieur à 8 ")
        else:
            df_new=pd.DataFrame([[username.lower(),mdp,"utilisateur"]],columns=["name", "mdp", "role"])
            pd.concat([pd.read_csv("db.csv"),df_new],axis=0).to_csv("db.csv",index=False)

st.title("Connexion")

usernamedb=st.text_input("indiquez votre username",key="usernamedb")
mdpdb=st.text_input("indiquez votre MDP",type ="password",key="mdpdb")
if st.button("Connexion",key="connexion"):
    if usernamedb in pd.read_csv("db.csv").name.values:
        db=pd.read_csv("db.csv")
        if db[db.name==usernamedb].mdp.iloc[0]==mdpdb:
            st.write("Bienvenue vous êtes identifiés")
        else:
            st.write("le mot de passe n'est pas bon : ressayez")

#st.dataframe(pd.read_csv("db.csv"))


# Zone principale
st.title("🎬 Application de Recommandation de Films")

if st.session_state.logged_in:
    st.success(f"Bienvenue {st.session_state.role} !")
    st.write("Ici s’affichera ton moteur de recommandation.")
else:
    st.info("Connecte-toi pour accéder aux recommandations.")

# Si connecté, afficher l'app principale


if st.session_state.logged_in:
    

    st.image("https://img.phonandroid.com/2023/05/netflix-films-audiences-baisse.jpg", use_container_width=True)
    st.title("🎬 Explorateur & Recommandateur de Films")
    st.markdown("Choisissez un film pour voir ses détails et découvrir des suggestions similaires.")

    #@st.cache_data
    def load_data():
        df = pd.read_csv(r"C:\Users\wildc\Desktop\Projet 2\df_final.csv")
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

    # ✅ Zone Admin (bien protégée)
    if st.session_state.role == "admin":
        st.sidebar.markdown("---")
        st.sidebar.header("👑 Zone Admin")

        show_page1 = st.sidebar.checkbox("Selection et recommadation")
        show_genre_chart = st.sidebar.checkbox("📊 Graphique : Films par genre")
        show_actor_chart = st.sidebar.checkbox("👥 Graphique : Films par acteur")
        show_10_film_les_mieux_notés=st.sidebar.checkbox("les 10 films les mieux notés")
        show_duree_film_par_genre=st.sidebar.checkbox("Duree moyenne de film par genre")
        show_duree_film_par_annee=st.sidebar.checkbox("Duree moyenne de film par Année")
        show_nb_film_par_annee=st.sidebar.checkbox("Nombre de films par année")



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


        #if show_genre_chart:
          #st.subheader("📊 Répartition des films par genre")
          

          #data_genre_film = df[["Title", "genre 1"]]
          #data_nb_genre = data_genre_film["genre 1"].value_counts().reset_index()
          #data_nb_genre.columns = ["genre", "nombre_films"]
          #data_nb_genre = data_nb_genre.sort_values(by="nombre_films", ascending=False)

          #fig, ax = plt.subplots(figsize=(10, 6))
          #ax.bar(data_nb_genre["genre"], data_nb_genre["nombre_films"], color="royalblue", edgecolor="black")
          #ax.set_xlabel("Genre")
          #ax.set_ylabel("Nombre de films")
          #ax.set_title("🎞️ Nombre de films par genre")
          #plt.xticks(rotation=45, ha="right")
          #st.pyplot(fig)


        if show_genre_chart:
            st.subheader("📊 Répartition des films par genre")

            data_genre_film = df[["Title", "genre 1"]]
            data_nb_genre = data_genre_film["genre 1"].value_counts().reset_index()
            data_nb_genre.columns = ["genre", "nombre_films"]
            data_nb_genre = data_nb_genre.sort_values(by="nombre_films", ascending=False)

            # 🎛️ Widget interactif pour filtrer les genres
            genres_disponibles = data_nb_genre["genre"].tolist()
            genres_choisis = st.multiselect("🎬 Choisis les genres à afficher :", genres_disponibles, default=genres_disponibles)

            # Filtrer les données selon les genres choisis
            data_filtrée = data_nb_genre[data_nb_genre["genre"].isin(genres_choisis)]

            # 🎨 Création du graphique dynamique
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(data_filtrée["genre"], data_filtrée["nombre_films"], color="royalblue", edgecolor="black")
            ax.set_xlabel("Genre")
            ax.set_ylabel("Nombre de films")
            ax.set_title("🎞️ Nombre de films par genre sélectionné")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)


        if show_nb_film_par_annee:
            st.subheader("Nombre de films par année")
            df_year_filtered = df[df["Year"].notna() &
            (df["Year"] != 0) ]
            if 'Year' in df.columns:
             st.bar_chart(df_year_filtered["Year"].value_counts().sort_index())
            else:
             st.info("La colonne 'Year' n'est pas disponible pour ce graphique.")


        if show_actor_chart:
          st.subheader("📊 Nombre de film par acteur")
          concat_actors=pd.concat([df['acteur1'],df['acteur2'],df['acteur3']],axis=0)
          df_actors=concat_actors.value_counts().head(10)
          df_actors = df_actors.reset_index()
          df_actors.columns=["acteur","count"]

          # Création de la figure matplotlib
          fig, ax = plt.subplots(figsize=(12, 6))
          ax.barh(df_actors["acteur"], df_actors["count"], color="steelblue")
          ax.set_xlabel("Nombre de films")
          ax.set_title("🎬 Nombre de films par acteur")
          ax.invert_yaxis()  # Affiche les plus fréquents en haut
          plt.tight_layout()

            # Affichage dans Streamlit
          st.pyplot(fig)


        if show_10_film_les_mieux_notés:
            st.subheader("📊 les 10 films les mieux notés")
            df_note=df[["Title","imdbRating"]]
            df_note_1=sorted(df_note["imdbRating"],reverse=True)
            df_note1 = df_note.sort_values(by=['imdbRating'],ascending=False).head(10)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(df_note1["Title"], df_note1["imdbRating"], color="royalblue", edgecolor="black")
            ax.set_xlabel("Note de film")
            ax.set_ylabel("Titre des films")
            ax.set_title("🎞️ les 10 films les mieux notés")
            ax.set_ylim(8, 10)
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)

        if show_duree_film_par_genre:
            st.subheader("📊 Duree moyenne de film par genre")
            df_duree=df[["Title","Runtime","genre 1","Year"]]
            df_duree_moyenne_par_genre = df_duree.groupby('genre 1')['Runtime'].mean().reset_index()
            df_duree_moyenne_par_genre=df_duree_moyenne_par_genre.sort_values(by="Runtime",ascending=True)

            fig, ax = plt.subplots(figsize=(8, 10))
            ax.barh(df_duree_moyenne_par_genre["genre 1"], df_duree_moyenne_par_genre["Runtime"], color="cornflowerblue")
            ax.set_title("⏱️ Durée moyenne par genre", fontsize=14)
            ax.set_xlabel("Durée moyenne (minutes)")
            ax.set_ylabel("Genre")
            st.pyplot(fig)

        
        if show_duree_film_par_annee:
            st.subheader("📊 Duree moyenne de film par annee")
            df_duree1=df[["Title","Runtime","genre 1","Year"]]
            df_duree_moyenne_par_annee = df_duree1.groupby('Year')['Runtime'].mean().reset_index()
            df_duree_moyenne_par_annee = df_duree_moyenne_par_annee.sort_values(by="Year", ascending=False).head(25)

            # Courbe
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df_duree_moyenne_par_annee["Year"], df_duree_moyenne_par_annee["Runtime"], marker='o', linestyle='-', color='teal')

            # Améliorations graphiques
            ax.set_title("⏱️ Durée moyenne des films par genre", fontsize=14)
            ax.set_xlabel("Année")
            ax.set_ylabel("Durée moyenne (minutes)")
            plt.xticks(rotation=45, ha='right')
            plt.grid(True)
            plt.show()

            # Affichage dans Streamlit
            st.pyplot(fig)

        



