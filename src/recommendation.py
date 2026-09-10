# recommendation.py
# Tourism Experience Analytics
# Objective: Personalized Attraction Recommendation
# Method: Item-Based Collaborative Filtering

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "processed",
    "master_tourism_data.csv"
)

DATA_PATH = os.path.abspath(DATA_PATH)

print("=" * 60)
print("TOURISM EXPERIENCE ANALYTICS - RECOMMENDATION")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# 2. CREATE USER-ITEM MATRIX
# ============================================================

print("\nCreating user-attraction rating matrix...")

user_item_matrix = df.pivot_table(
    index="UserId",
    columns="AttractionId",
    values="Rating",
    aggfunc="mean"
).fillna(0)

print("User-item matrix shape:", user_item_matrix.shape)


# ============================================================
# 3. CALCULATE ATTRACTION SIMILARITY
# ============================================================

print("\nCalculating attraction similarity...")

attraction_matrix = user_item_matrix.T

similarity_matrix = cosine_similarity(attraction_matrix)

attraction_similarity = pd.DataFrame(
    similarity_matrix,
    index=user_item_matrix.columns,
    columns=user_item_matrix.columns
)

print("Similarity matrix shape:", attraction_similarity.shape)


# ============================================================
# 4. ATTRACTION INFORMATION
# ============================================================

attraction_info = (
    df[
        [
            "AttractionId",
            "Attraction",
            "AttractionType",
            "AttractionCity",
            "Country",
            "Region"
        ]
    ]
    .drop_duplicates("AttractionId")
    .set_index("AttractionId")
)


# ============================================================
# 5. RECOMMENDATION FUNCTION
# ============================================================

def recommend_attractions(user_id, top_n=5):

    if user_id not in user_item_matrix.index:
        return pd.DataFrame()

    user_ratings = user_item_matrix.loc[user_id]

    # Attractions already visited
    visited_attractions = user_ratings[
        user_ratings > 0
    ].index.tolist()

    # Attractions rated positively
    liked_attractions = user_ratings[
        user_ratings >= 4
    ].index.tolist()

    # If no rating >= 4, use visited attractions
    if len(liked_attractions) == 0:
        liked_attractions = visited_attractions

    scores = {}

    for candidate in user_item_matrix.columns:

        # Never recommend an attraction already visited
        if candidate in visited_attractions:
            continue

        score = 0.0
        similarity_sum = 0.0

        for liked in liked_attractions:

            similarity = attraction_similarity.loc[
                candidate,
                liked
            ]

            rating = user_ratings[liked]

            score += similarity * rating
            similarity_sum += abs(similarity)

        if similarity_sum > 0:
            scores[candidate] = score / similarity_sum

    if not scores:
        return pd.DataFrame()

    recommendations = (
        pd.Series(scores)
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )

    recommendations.columns = [
        "AttractionId",
        "RecommendationScore"
    ]

    recommendations = recommendations.merge(
        attraction_info.reset_index(),
        on="AttractionId",
        how="left"
    )

    return recommendations[
        [
            "AttractionId",
            "Attraction",
            "AttractionType",
            "AttractionCity",
            "Country",
            "Region",
            "RecommendationScore"
        ]
    ]


# ============================================================
# 6. TEST RECOMMENDER
# ============================================================

print("\n" + "=" * 60)
print("TESTING RECOMMENDER")
print("=" * 60)

user_visit_counts = df.groupby("UserId").size()

eligible_test_users = user_visit_counts[
    user_visit_counts >= 2
].index

test_user = eligible_test_users[0]

print("Test UserId:", test_user)

recommendations = recommend_attractions(
    test_user,
    top_n=5
)

print("\nTop 5 Recommended Attractions:")

if recommendations.empty:
    print("No recommendations available.")

else:
    print(
        recommendations[
            [
                "Attraction",
                "AttractionType",
                "AttractionCity",
                "Country",
                "RecommendationScore"
            ]
        ].to_string(index=False)
    )


# ============================================================
# 7. HOLDOUT EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("RECOMMENDATION EVALUATION")
print("=" * 60)

print("\nPerforming leave-one-out evaluation...")


def evaluate_user(user_id, k=5):

    user_data = df[df["UserId"] == user_id].copy()

    # Positive interactions
    positive_data = user_data[
        user_data["Rating"] >= 4
    ]

    # Need at least 2 positive interactions
    if len(positive_data) < 2:
        return None

    # Hold out the last positive interaction
    test_attraction = positive_data.iloc[-1]["AttractionId"]

    # Remaining interactions are training data
    train_data = user_data[
        user_data.index != positive_data.index[-1]
    ]

    # Create user's training ratings
    train_ratings = (
        train_data.groupby("AttractionId")["Rating"]
        .mean()
    )

    visited = set(train_ratings.index)

    liked = train_ratings[
        train_ratings >= 4
    ].index.tolist()

    if len(liked) == 0:
        return None

    scores = {}

    for candidate in user_item_matrix.columns:

        # Do not recommend attractions already in training history
        if candidate in visited:
            continue

        score = 0.0
        similarity_sum = 0.0

        for liked_id in liked:

            similarity = attraction_similarity.loc[
                candidate,
                liked_id
            ]

            rating = train_ratings[liked_id]

            score += similarity * rating
            similarity_sum += abs(similarity)

        if similarity_sum > 0:
            scores[candidate] = score / similarity_sum

    if not scores:
        return None

    recommended_ids = (
        pd.Series(scores)
        .sort_values(ascending=False)
        .head(k)
        .index
        .tolist()
    )

    # Precision@5
    hit = 1 if test_attraction in recommended_ids else 0

    precision = hit / k

    return precision


# ============================================================
# 8. RUN EVALUATION
# ============================================================

evaluation_users = (
    df.groupby("UserId")
    .filter(lambda x: len(x) >= 2)["UserId"]
    .drop_duplicates()
    .head(1000)
    .tolist()
)

precision_scores = []

for user_id in evaluation_users:

    score = evaluate_user(
        user_id,
        k=5
    )

    if score is not None:
        precision_scores.append(score)


if precision_scores:

    average_precision = np.mean(
        precision_scores
    )

else:

    average_precision = 0.0


print("\nUsers evaluated:", len(precision_scores))

print(
    f"Precision@5: {average_precision:.4f}"
)


# ============================================================
# 9. SAVE RECOMMENDATION DATA
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# Save user-item matrix
USER_ITEM_PATH = os.path.join(
    MODEL_DIR,
    "user_item_matrix.pkl"
)

joblib.dump(
    user_item_matrix,
    USER_ITEM_PATH
)


# Save attraction similarity
SIMILARITY_PATH = os.path.join(
    MODEL_DIR,
    "attraction_similarity.pkl"
)

joblib.dump(
    attraction_similarity,
    SIMILARITY_PATH
)


# Save attraction information
ATTRACTION_INFO_PATH = os.path.join(
    MODEL_DIR,
    "attraction_info.pkl"
)

joblib.dump(
    attraction_info,
    ATTRACTION_INFO_PATH
)


# ============================================================
# 10. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("RECOMMENDATION SYSTEM COMPLETED")
print("=" * 60)

print("\nSaved files:")
print(USER_ITEM_PATH)
print(SIMILARITY_PATH)
print(ATTRACTION_INFO_PATH)

print(
    "\nFinal Precision@5:",
    round(average_precision, 4)
)

print("\nRecommendation system completed successfully!")