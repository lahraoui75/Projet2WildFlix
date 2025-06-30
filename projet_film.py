import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors
import base64
import time 

if "show_registration" not in st.session_state:
    st.session_state.show_registration = False

# Configuration de la page
st.set_page_config(
    page_title="Wildflix",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


dark_css = """
<style>
/* Fond général */
[data-testid="stAppViewContainer"] {
    background-color: #232846;
    color: white;
}

/* Boutons secondaires mieux visibles */
.baseButton-secondary {
    color: white;
    background-color: #3c4454;
    border: 1px solid #5c667a;
}

/* Titres principaux et sous-titres */
.st-emotion-cache-1whx7iy,
#titre-principal,
#sous-titre {
    color: white !important;
}

/* Sections personnalisées */
#mon-id-custom {
    color: white !important;
}

/* Onglets actifs/inactifs */
.streamlit-tab-text {
    color: white !important;
}
.streamlit-tab-list {
    background-color: #2a2f3a !important;
}

/* Listes déroulantes, sélecteurs, etc. */
.stSelectbox, .st-emotion-cache-1inwz65 {
    color: white;
    background-color: #2a2f3a;
}

/* Supprimer les fonds opaques gênants */
.user-select-none {
    background-color: transparent;
}
</style>
"""

st.markdown(dark_css, unsafe_allow_html=True)

def play_audio(file_path):
    if not os.path.isfile(file_path):
        st.error(f"❌ Le fichier audio '{file_path}' est introuvable.")
        return
    with open(file_path, "rb") as f:
        data = f.read()
    b64_encoded = base64.b64encode(data).decode()
    audio_html = f"""
    <audio autoplay>
        <source src="data:audio/mp3;base64,{b64_encoded}" type="audio/mp3">
        Votre navigateur ne prend pas en charge l'audio HTML5.
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)



# Initialisation de session
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "role" not in st.session_state:
    st.session_state.role = None
if "current_page" not in st.session_state:
    st.session_state.current_page = "init"


if st.session_state.current_page == "init":
    st.title("🆕 Créer un compte")

    new_username = st.text_input("Choisissez un nom d'utilisateur", key="signup_user")
    new_password = st.text_input("Choisissez un mot de passe", type="password", key="signup_mdp")
    confirm_password = st.text_input("Confirmez le mot de passe", type="password", key="signup_confirm")

    if st.button("Valider l'inscription"):
        if not new_username or not new_password or not confirm_password:
            st.warning("Merci de remplir tous les champs.")
        elif new_password != confirm_password:
            st.error("Les mots de passe ne correspondent pas.")
        elif len(new_password) < 8:
            st.error("Le mot de passe doit contenir au moins 8 caractères.")
        else:
            users_df = pd.read_csv("db.csv")
            if new_username.lower() in users_df["name"].values:
                st.error("Ce nom d'utilisateur est déjà utilisé.")
            else:
                nouveau = pd.DataFrame([[new_username.lower(), new_password, "utilisateur"]], columns=["name", "mdp", "role"])
                nouveau_df = pd.concat([users_df, nouveau], ignore_index=True)
                nouveau_df.to_csv("db.csv", index=False)
                st.success("Votre compte a été créé avec succès ! Vous pouvez maintenant vous connecter.")



# Chargement des identifiants depuis le fichier CSV
if not os.path.exists("db.csv"):
    pd.DataFrame([
        ["admin", "adminpass", "admin"],
        ["user", "12345678", "utilisateur"]
    ], columns=["name", "mdp", "role"]).to_csv("db.csv", index=False)

# Si l'utilisateur n'est PAS connecté : afficher le formulaire de connexion
if not st.session_state.logged_in:
    st.title("🔐 Connexion Wildflix")

    login_username = st.text_input("Nom d'utilisateur")
    login_password = st.text_input("Mot de passe", type="password")

    if st.button("Se connecter"):
        utilisateurs = pd.read_csv("db.csv")
        if login_username in utilisateurs.name.values:
            row = utilisateurs[utilisateurs.name == login_username].iloc[0]
            if login_password == row["mdp"]:
                st.session_state.logged_in = True
                st.session_state.role = row["role"]

                # 🔊 Lecture du son d’accueil (corrige le chemin si besoin)
                play_audio("Son NETFLIX.mp3")
                time.sleep(1)  # ⏳ Donne le temps de lire le son avant le reru


                st.success(f"Bienvenue **{row['role']}** {login_username} !")
                st.rerun()

            else:
                st.error("Mot de passe incorrect.")
        else:
            st.error("Nom d'utilisateur inconnu.")
    st.stop()  # ⛔️ On bloque tout le reste du script tant qu'on n'est pas connecté

if st.session_state.get("logged_in", False):
    st.sidebar.success(f"Connecté en tant que {st.session_state.role}")
    
    if st.sidebar.button("Se déconnecter 🔓"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


st.title("🎥 WildFlix — Plateforme de Recommandation de Films")

st.image("https://img.phonandroid.com/2023/05/netflix-films-audiences-baisse.jpg", use_container_width=True)

titres_onglets = ['Bienvenue', 'Recommandation', 'Dashboard']
onglet1, onglet2, onglet3 = st.tabs(titres_onglets)

with onglet1:
        st.write('Votre plateforme de recommandation WildFlix vous offre une fonctionnalité de recommandation personnalisée.')
        st.write('Les utilisateurs peuvent indiquer un film et le système leur suggérera des films correspondants.')
        st.markdown("<span style='color: grey;'>© Olivier, Nicolas, Mustapha, Tawfik 2025</span>", unsafe_allow_html=True)

with onglet2:
        st.title("🎬 Explorateur & Recommandateur de Films")

        @st.cache_data
        def load_data():
            df = pd.read_csv(r"https://raw.githubusercontent.com/lahraoui75/Projet2WildFlix/refs/heads/main/df_final_V1.csv")
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

# Choix du modèle de recommandation
        modele = st.radio("🧠 Choisissez la méthode de recommandation :", ["Cosine Similarity", "KNN (plus proches voisins)"])

        # Sélection du film
        film_selectionne = st.selectbox("🎞️ Sélectionnez un film :", df[titre_col].dropna().unique())
        film_info = df[df[titre_col] == film_selectionne]
        ligne = film_info.index[0] if not film_info.empty else None

        # Calcul des recommandations
        def obtenir_recommandations(titre_film, n=5):
            try:
                index_film = df[df[titre_col] == titre_film].index[0]
                X = df[features_similitude].fillna(0)

                if modele == "Cosine Similarity":
                    similarity_matrix = cosine_similarity(X)
                    scores = list(enumerate(similarity_matrix[index_film]))
                    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
                    indices = [i[0] for i in scores]

                else:  # KNN
                    knn = NearestNeighbors(n_neighbors=n+1, metric="euclidean")
                    knn.fit(X)
                    distances, indices_array = knn.kneighbors([X.iloc[index_film]])
                    indices = indices_array[0][1:]  # on exclut le film lui-même

                return df.loc[indices][[titre_col, "Year", "Poster", "imdbRating", "Genre", "Director"]]

            except Exception as e:
                st.warning(f"Erreur : {e}")
                return pd.DataFrame()

        # Affichage de l’info du film sélectionné
        if ligne is not None:
            row = df.loc[ligne]
            col1, col2 = st.columns([1, 2])
            with col1:
                if pd.notna(row.get("Poster", None)):
                    st.image(row["Poster"], width=250)
            with col2:
                st.markdown(f"## ⭐ {row[titre_col]} ({row['Year']})")
                st.markdown(f"**Réalisateur :** {row['Director']}")
                st.markdown(f"**Acteurs :** {row['Actors']}")
                st.markdown(f"**Genre :** {row['Genre']}")
                st.markdown(f"**Durée :** {row['Runtime']}")
                st.markdown(f"**Note IMDB :** ⭐ {row['imdbRating']}")
                st.markdown(f"**Box Office :** {row['BoxOffice']}")
                st.markdown(f"**Résumé :** {row['Plot']}")

        # Affichage des recommandations
        similaires = obtenir_recommandations(film_selectionne)
        if not similaires.empty:
            st.subheader("🎯 Films recommandés")
            for _, film in similaires.iterrows():
                subcol1, subcol2 = st.columns([1, 3])
                with subcol1:
                    if pd.notna(film["Poster"]):
                        st.image(film["Poster"], width=100)
                with subcol2:
                    st.markdown(f"**{film[titre_col]} ({film['Year']})** – ⭐ {film['imdbRating']}")
                    st.markdown(f"_Genre : {film['Genre']}  |  Réalisateur : {film['Director']}_")
                st.markdown("---")
        

if st.session_state.role == "admin":
 with onglet3:
        st.sidebar.markdown("---")
        st.sidebar.header("👑 Zone Admin")

        #st.dataframe(pd.read_csv("db.csv"))

        show_affichage_nb_compte=st.sidebar.checkbox("Aperçu du nombre des comptes")
        show_affichage_colonnes=st.sidebar.checkbox("Aperçu des colonnes")
        show_genre_chart = st.sidebar.checkbox("📊 Graphique : Films par genre")
        show_actor_chart = st.sidebar.checkbox("👥 Graphique :  nombre de Films par acteur")
        show_10_film_les_mieux_notés=st.sidebar.checkbox("les 10 films les mieux notés")
        show_duree_film_par_genre=st.sidebar.checkbox("Duree moyenne de film par genre")
        show_duree_film_par_annee=st.sidebar.checkbox("Duree moyenne de film par Année")
        show_nb_film_par_annee=st.sidebar.checkbox("Nombre de films par année")
        
        
    
        if show_affichage_nb_compte:
            st.subheader(" Aperçu du nombre des comptes")
            st.dataframe(pd.read_csv("db.csv"))

        if show_affichage_colonnes:
            st.subheader(" Aperçu des colonnes")
            st.write("🧾 Colonnes du DataFrame :", df.columns.tolist())


      
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

        if show_nb_film_par_annee:
            st.subheader("Nombre de films par année")
            df_year_filtered = df[df["Year"].notna() &
            (df["Year"] != 0) ]
            if 'Year' in df.columns:
             st.bar_chart(df_year_filtered["Year"].value_counts().sort_index())
            else:
             st.info("La colonne 'Year' n'est pas disponible pour ce graphique.")

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
            #plt.show()

            # Affichage dans Streamlit
            st.pyplot(fig)
        

       


        


   
