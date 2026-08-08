import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Movie Recommendation & Prediction",
    page_icon="🎬",
    layout="wide"
)


# ============================================================
# LOAD FILES
# ============================================================

@st.cache_resource
def load_files():

    model = joblib.load("movie_hit_flop_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    genre_encoder = joblib.load("genre_encoder.pkl")

    movie_data = joblib.load("movie_data.pkl")
    genre_features = joblib.load("genre_features.pkl")

    return (
        model,
        label_encoder,
        genre_encoder,
        movie_data,
        genre_features
    )


try:

    (
        model,
        label_encoder,
        genre_encoder,
        movie_data,
        genre_features
    ) = load_files()

except Exception as e:

    st.error("❌ Could not load the required files.")

    st.code(str(e))

    st.info(
        """
        Make sure these files are in the same folder as app.py:

        movie_hit_flop_model.pkl
        label_encoder.pkl
        genre_encoder.pkl
        movie_data.pkl
        genre_features.pkl
        """
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("🎬 Movie Recommendation & Prediction System")

st.markdown(
    """
    Enter a movie name to get:

    🎭 Movie information  
    🤖 Hit / Flop prediction  
    📊 Prediction probabilities  
    🍿 Similar movie recommendations
    """
)

st.divider()


# ============================================================
# MOVIE SEARCH
# ============================================================

st.subheader("🔎 Search for a Movie")

movie_name = st.text_input(
    "Enter movie name:",
    placeholder="Example: Toy Story"
)


if not movie_name.strip():

    st.info("👆 Enter a movie name to get started.")

    st.stop()


# ============================================================
# FIND MOVIE
# ============================================================

matches = movie_data[
    movie_data["title"]
    .astype(str)
    .str.contains(
        movie_name.strip(),
        case=False,
        na=False
    )
]


if matches.empty:

    st.error(
        f"❌ No movie found matching **{movie_name}**."
    )

    st.info(
        "Try entering a shorter part of the movie name."
    )

    st.stop()


# ============================================================
# MULTIPLE MATCHES
# ============================================================

if len(matches) > 1:

    st.info(
        f"🔎 Found {len(matches)} matching movies."
    )

    selected_movie = st.selectbox(
        "Choose a movie:",
        matches["title"].tolist()
    )

    selected_movie_data = matches[
        matches["title"] == selected_movie
    ].iloc[0]

else:

    selected_movie_data = matches.iloc[0]

    selected_movie = selected_movie_data["title"]


# ============================================================
# MOVIE INFORMATION
# ============================================================

st.divider()

st.subheader("🎥 Movie Information")

try:

    average_rating = float(
        selected_movie_data["average_rating"]
    )

except Exception:

    average_rating = 0.0


genres = str(
    selected_movie_data["genres"]
)


year = selected_movie_data.get(
    "year",
    "N/A"
)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "⭐ Average Rating",
        f"{average_rating:.2f}"
    )


with col2:

    st.metric(
        "📅 Release Year",
        str(year)
    )


with col3:

    genre_list = genres.split("|")

    st.metric(
        "🎭 Number of Genres",
        len(genre_list)
    )


st.write(
    f"**Genres:** {genres}"
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.divider()

analyze = st.button(
    "🔮 Analyze Movie",
    type="primary",
    use_container_width=True
)


if analyze:

    # ========================================================
    # PREDICTION MODEL
    # ========================================================

    st.subheader("🤖 Hit / Flop Prediction")

    try:

        # ----------------------------------------------------
        # GET MOVIE GENRES
        # ----------------------------------------------------

        selected_genres = [
            genre.strip()
            for genre in genres.split("|")
        ]


        # ----------------------------------------------------
        # CREATE INPUT DATAFRAME
        #
        # IMPORTANT:
        # Use FLOAT instead of INT because average_rating
        # contains decimal values.
        # ----------------------------------------------------

        prediction_df = pd.DataFrame(
            0.0,
            index=[0],
            columns=model.feature_names_in_,
            dtype=float
        )


        # ----------------------------------------------------
        # SET GENRE FEATURES
        # ----------------------------------------------------

        for genre in selected_genres:

            if genre in prediction_df.columns:

                prediction_df.loc[
                    0,
                    genre
                ] = 1.0


        # ----------------------------------------------------
        # SET AVERAGE RATING
        # ----------------------------------------------------

        if "average_rating" in prediction_df.columns:

            prediction_df.loc[
                0,
                "average_rating"
            ] = average_rating


        # ----------------------------------------------------
        # ENSURE EXACT FEATURE ORDER
        # ----------------------------------------------------

        prediction_df = prediction_df[
            model.feature_names_in_
        ]


        # ----------------------------------------------------
        # MAKE PREDICTION
        # ----------------------------------------------------

        prediction_number = model.predict(
            prediction_df
        )[0]


        # ----------------------------------------------------
        # CONVERT NUMERIC CLASS TO TEXT
        # ----------------------------------------------------

        prediction = label_encoder.inverse_transform(
            [int(prediction_number)]
        )[0]


        # ----------------------------------------------------
        # GET PROBABILITIES
        # ----------------------------------------------------

        probabilities = model.predict_proba(
            prediction_df
        )[0]


        # ----------------------------------------------------
        # MAP PROBABILITIES TO FLOP / HIT
        # ----------------------------------------------------

        model_class_names = (
            label_encoder.inverse_transform(
                model.classes_.astype(int)
            )
        )


        probability_dict = dict(
            zip(
                model_class_names,
                probabilities
            )
        )


        flop_probability = float(
            probability_dict.get(
                "Flop",
                0.0
            ) * 100
        )


        hit_probability = float(
            probability_dict.get(
                "Hit",
                0.0
            ) * 100
        )


        # ----------------------------------------------------
        # DISPLAY RESULT
        # ----------------------------------------------------

        if prediction == "Hit":

            st.success(
                "🎉 This movie is predicted to be a **HIT**!"
            )

        else:

            st.warning(
                "🎭 This movie is predicted to be a **FLOP**."
            )


        # ----------------------------------------------------
        # PROBABILITY METRICS
        # ----------------------------------------------------

        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "🎉 Hit Probability",
                f"{hit_probability:.2f}%"
            )


        with col2:

            st.metric(
                "🎭 Flop Probability",
                f"{flop_probability:.2f}%"
            )


        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        confidence = max(
            hit_probability,
            flop_probability
        )


        st.write("### 📊 Prediction Confidence")

        st.progress(
            min(
                max(
                    int(round(confidence)),
                    0
                ),
                100
            )
        )


        st.caption(
            f"Model confidence: **{confidence:.2f}%**"
        )


    except Exception as e:

        st.error(
            "❌ Prediction failed."
        )

        st.code(str(e))


    # ========================================================
    # RECOMMENDATION MODEL
    # ========================================================

    st.divider()

    st.subheader("🍿 Recommended Movies")

    try:

        # ----------------------------------------------------
        # GET SELECTED MOVIE INDEX
        # ----------------------------------------------------

        selected_index = selected_movie_data.name


        # ----------------------------------------------------
        # CHECK IF INDEX EXISTS
        # ----------------------------------------------------

        if selected_index not in genre_features.index:

            st.warning(
                "⚠️ Recommendation data for this movie "
                "could not be found."
            )

        else:

            # ------------------------------------------------
            # GET SELECTED MOVIE GENRE VECTOR
            # ------------------------------------------------

            selected_vector = genre_features.loc[
                [selected_index]
            ]


            # ------------------------------------------------
            # COSINE SIMILARITY
            # ------------------------------------------------

            similarity_scores = cosine_similarity(
                selected_vector,
                genre_features
            )[0]


            # ------------------------------------------------
            # CREATE SIMILARITY TABLE
            # ------------------------------------------------

            similarity_df = pd.DataFrame(
                {
                    "movie_index": genre_features.index,
                    "similarity": similarity_scores
                }
            )


            # ------------------------------------------------
            # REMOVE SELECTED MOVIE
            # ------------------------------------------------

            similarity_df = similarity_df[
                similarity_df["movie_index"]
                != selected_index
            ]


            # ------------------------------------------------
            # TOP 10 RECOMMENDATIONS
            # ------------------------------------------------

            top_recommendations = (
                similarity_df
                .sort_values(
                    "similarity",
                    ascending=False
                )
                .head(10)
            )


            # ------------------------------------------------
            # GET MOVIE INFORMATION
            # ------------------------------------------------

            recommendations = movie_data.loc[
                movie_data.index.isin(
                    top_recommendations[
                        "movie_index"
                    ]
                )
            ].copy()


            # ------------------------------------------------
            # ADD SIMILARITY SCORE
            # ------------------------------------------------

            similarity_lookup = dict(
                zip(
                    top_recommendations[
                        "movie_index"
                    ],
                    top_recommendations[
                        "similarity"
                    ]
                )
            )


            recommendations["similarity"] = (
                recommendations.index.map(
                    similarity_lookup
                )
            )


            recommendations = (
                recommendations
                .sort_values(
                    "similarity",
                    ascending=False
                )
            )


            # ------------------------------------------------
            # DISPLAY RECOMMENDATIONS
            # ------------------------------------------------

            if recommendations.empty:

                st.info(
                    "No recommendations available."
                )

            else:

                for number, (_, movie) in enumerate(
                    recommendations.iterrows(),
                    start=1
                ):

                    col1, col2, col3 = st.columns(
                        [0.08, 0.62, 0.30]
                    )


                    with col1:

                        st.markdown(
                            f"### {number}"
                        )


                    with col2:

                        st.markdown(
                            f"**{movie['title']}**"
                        )

                        st.caption(
                            str(movie["genres"])
                        )


                    with col3:

                        try:

                            recommendation_rating = float(
                                movie["average_rating"]
                            )

                        except Exception:

                            recommendation_rating = 0.0


                        st.write(
                            f"⭐ "
                            f"{recommendation_rating:.2f}"
                        )


                        st.write(
                            "Similarity: "
                            f"{float(movie['similarity']) * 100:.1f}%"
                        )


                    st.divider()


    except Exception as e:

        st.error(
            "❌ Recommendation system failed."
        )

        st.code(str(e))


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🎬 About")

    st.write(
        """
        ### 🤖 Prediction Model

        Predicts whether a movie is likely to be
        a **Hit** or **Flop** using:

        • Movie genres  
        • Average rating  


        ### 🍿 Recommendation Model

        Recommends similar movies using:

        • Genre features  
        • Cosine similarity
        """
    )

    st.divider()

    st.write(
        "**Dataset:** MovieLens"
    )

    st.write(
        "**Models:** XGBoost + Cosine Similarity"
    )

    st.write(
        "**Framework:** Streamlit"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🎬 Movie Recommendation & Prediction System | "
    "Built with Python, XGBoost and Streamlit"
)