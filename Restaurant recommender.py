"""
Restaurant Recommendation System — Content-Based Filtering
Dataset: Zomato Restaurant Dataset (9,551 restaurants)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. LOAD & PREPROCESS
# ─────────────────────────────────────────────

def load_and_preprocess(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"✅ Loaded {len(df):,} restaurants with {df.shape[1]} features")

    # ── Handle missing values ──────────────────────────────────────────────────
    missing_before = df.isnull().sum().sum()
    df["Cuisines"] = df["Cuisines"].fillna("Unknown")  # only 9 missing
    missing_after = df.isnull().sum().sum()
    print(f"   Missing values: {missing_before} → {missing_after} (imputed Cuisines with 'Unknown')")

    # ── Encode binary Yes/No columns ──────────────────────────────────────────
    binary_cols = ["Has Table booking", "Has Online delivery", "Is delivering now"]
    for col in binary_cols:
        df[col] = (df[col].str.strip().str.lower() == "yes").astype(int)

    # ── Normalise numeric features ─────────────────────────────────────────────
    scaler = MinMaxScaler()
    df[["Avg Cost Scaled", "Rating Scaled", "Votes Scaled"]] = scaler.fit_transform(
        df[["Average Cost for two", "Aggregate rating", "Votes"]]
    )

    # ── Exclude unrated restaurants from recommendations ──────────────────────
    df = df[df["Rating text"] != "Not rated"].reset_index(drop=True)
    print(f"   Removed 'Not rated' restaurants → {len(df):,} usable restaurants remain")

    # ── Parse cuisines into lists ──────────────────────────────────────────────
    df["Cuisine List"] = df["Cuisines"].apply(
        lambda x: [c.strip() for c in x.split(",")]
    )

    print(f"   Unique cuisine tags: {len(set(c for lst in df['Cuisine List'] for c in lst))}")
    return df


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING — build content matrix
# ─────────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame):
    """
    Feature vector per restaurant (columns):
      • Cuisine one-hot (MultiLabelBinarizer)
      • Price range one-hot (1-4)
      • Has Table booking (0/1)
      • Has Online delivery (0/1)
      • Normalised average cost
      • Normalised aggregate rating
    """
    # Cuisine features
    mlb = MultiLabelBinarizer()
    cuisine_matrix = mlb.fit_transform(df["Cuisine List"])
    cuisine_df = pd.DataFrame(cuisine_matrix, columns=mlb.classes_, index=df.index)

    # Price range one-hot
    price_dummies = pd.get_dummies(df["Price range"], prefix="price").astype(int)
    price_dummies.index = df.index

    # Combine all features
    feature_df = pd.concat([
        cuisine_df,
        price_dummies,
        df[["Has Table booking", "Has Online delivery",
            "Avg Cost Scaled", "Rating Scaled"]].reset_index(drop=True)
    ], axis=1)

    print(f"\n✅ Feature matrix: {feature_df.shape[0]:,} restaurants × {feature_df.shape[1]} features")
    return feature_df.values, mlb


# ─────────────────────────────────────────────
# 3. USER PROFILE BUILDER
# ─────────────────────────────────────────────

def build_user_profile(user_prefs: dict, df: pd.DataFrame,
                        feature_matrix: np.ndarray, mlb) -> np.ndarray:
    """
    Build a pseudo-restaurant vector from user preferences and
    return its feature vector for similarity search.

    user_prefs keys (all optional):
        cuisines        : list[str]  e.g. ["Italian", "Pizza"]
        price_range     : int        1=cheap … 4=expensive
        online_delivery : bool
        table_booking   : bool
        min_rating      : float      e.g. 3.5
    """
    # Cuisine one-hot
    cuisines = user_prefs.get("cuisines", [])
    known_classes = mlb.classes_
    cuisine_vec = np.array(
        [1 if c in cuisines else 0 for c in known_classes], dtype=float
    )

    # Price range one-hot (4 positions → price_1 … price_4)
    price_vec = np.zeros(4, dtype=float)
    if "price_range" in user_prefs:
        pr = int(user_prefs["price_range"])
        if 1 <= pr <= 4:
            price_vec[pr - 1] = 1.0

    # Binary features
    table_book   = float(user_prefs.get("table_booking", 0))
    online_deliv = float(user_prefs.get("online_delivery", 0))

    # Cost / rating placeholders (neutral 0.5 unless preference given)
    cost_scaled   = 0.5
    rating_scaled = user_prefs.get("min_rating", 3.5) / 4.9   # normalise to [0,1]

    user_vector = np.hstack([
        cuisine_vec,
        price_vec,
        [table_book, online_deliv, cost_scaled, rating_scaled]
    ])
    return user_vector


# ─────────────────────────────────────────────
# 4. RECOMMENDATION ENGINE
# ─────────────────────────────────────────────

def recommend(user_prefs: dict, df: pd.DataFrame,
              feature_matrix: np.ndarray, mlb,
              top_n: int = 10,
              min_rating: float = 0.0) -> pd.DataFrame:
    """
    Returns top_n restaurant recommendations sorted by similarity score,
    with an optional minimum aggregate rating filter.
    """
    user_vec = build_user_profile(user_prefs, df, feature_matrix, mlb)
    scores   = cosine_similarity([user_vec], feature_matrix)[0]

    # Optional hard rating floor
    if min_rating > 0:
        mask = df["Aggregate rating"] >= min_rating
        scores = np.where(mask, scores, -1)

    top_indices = np.argsort(scores)[::-1][:top_n]

    results = df.iloc[top_indices][[
        "Restaurant Name", "City", "Cuisines",
        "Price range", "Aggregate rating", "Rating text",
        "Has Table booking", "Has Online delivery", "Votes"
    ]].copy()
    results["Similarity Score"] = np.round(scores[top_indices], 4)
    results = results.rename(columns={
        "Has Table booking":   "Table Booking",
        "Has Online delivery": "Online Delivery"
    })
    results["Table Booking"]   = results["Table Booking"].map({1: "✓", 0: "✗"})
    results["Online Delivery"] = results["Online Delivery"].map({1: "✓", 0: "✗"})
    results.index = range(1, len(results) + 1)
    return results


# ─────────────────────────────────────────────
# 5. EVALUATION HELPERS
# ─────────────────────────────────────────────

def evaluate_recommendations(results: pd.DataFrame,
                              user_prefs: dict,
                              verbose: bool = True) -> dict:
    """
    Lightweight quality metrics for a recommendation list.

    Metrics:
        cuisine_hit_rate  : fraction of results sharing ≥1 preferred cuisine
        avg_rating        : mean aggregate rating of recommended restaurants
        price_match_rate  : fraction matching requested price range (if given)
        delivery_match    : fraction with online delivery (if requested)
        diversity_score   : normalised number of unique cuisines in results
    """
    preferred_cuisines = set(user_prefs.get("cuisines", []))
    preferred_price    = user_prefs.get("price_range")
    wants_delivery     = user_prefs.get("online_delivery", False)

    # Cuisine hit rate
    if preferred_cuisines:
        hits = results["Cuisines"].apply(
            lambda x: any(c in x for c in preferred_cuisines)
        )
        cuisine_hit_rate = hits.mean()
    else:
        cuisine_hit_rate = None

    # Average rating
    avg_rating = results["Aggregate rating"].mean()

    # Price match
    if preferred_price:
        price_match_rate = (results["Price range"] == preferred_price).mean()
    else:
        price_match_rate = None

    # Delivery match
    if wants_delivery:
        delivery_match = (results["Online Delivery"] == "✓").mean()
    else:
        delivery_match = None

    # Diversity: unique cuisine tags / total tags across results
    all_tags = [
        c.strip()
        for row in results["Cuisines"].str.split(",")
        for c in row
    ]
    diversity_score = round(len(set(all_tags)) / max(len(all_tags), 1), 3)

    metrics = {
        "cuisine_hit_rate": cuisine_hit_rate,
        "avg_rating":       round(avg_rating, 2),
        "price_match_rate": price_match_rate,
        "delivery_match":   delivery_match,
        "diversity_score":  diversity_score,
    }

    if verbose:
        print("\n📊 Evaluation Metrics:")
        if cuisine_hit_rate is not None:
            print(f"   Cuisine Hit Rate  : {cuisine_hit_rate:.0%}")
        print(f"   Avg Rating        : {avg_rating:.2f} / 4.9")
        if price_match_rate is not None:
            print(f"   Price Match Rate  : {price_match_rate:.0%}")
        if delivery_match is not None:
            print(f"   Delivery Match    : {delivery_match:.0%}")
        print(f"   Diversity Score   : {diversity_score:.3f}  (unique tags / total tags)")

    return metrics


# ─────────────────────────────────────────────
# 6. MAIN — run test scenarios
# ─────────────────────────────────────────────

def print_results(title: str, results: pd.DataFrame):
    print(f"\n{'═'*70}")
    print(f"  {title}")
    print(f"{'═'*70}")
    display_cols = [
        "Restaurant Name", "City", "Cuisines",
        "Price range", "Aggregate rating", "Rating text",
        "Table Booking", "Online Delivery", "Similarity Score"
    ]
    print(results[display_cols].to_string(index=True))


if __name__ == "__main__":
    # ── Load & preprocess ──────────────────────────────────────────────────────
    DATA_PATH = "Dataset .csv"
    df = load_and_preprocess(DATA_PATH)

    # ── Build feature matrix ───────────────────────────────────────────────────
    feature_matrix, mlb = build_feature_matrix(df)

    # ══════════════════════════════════════════════════════════════════
    # TEST SCENARIO 1 — Italian food lover on a budget
    # ══════════════════════════════════════════════════════════════════
    user1 = {
        "cuisines":         ["Italian", "Pizza"],
        "price_range":      1,          # budget-friendly
        "min_rating":       3.5,
        "online_delivery":  True,
    }
    res1 = recommend(user1, df, feature_matrix, mlb, top_n=10, min_rating=3.5)
    print_results("SCENARIO 1 — Budget Italian / Pizza lover (wants delivery)", res1)
    evaluate_recommendations(res1, user1)

    # ══════════════════════════════════════════════════════════════════
    # TEST SCENARIO 2 — Fine-dining Japanese enthusiast
    # ══════════════════════════════════════════════════════════════════
    user2 = {
        "cuisines":       ["Japanese", "Sushi"],
        "price_range":    4,            # premium
        "table_booking":  True,
        "min_rating":     4.0,
    }
    res2 = recommend(user2, df, feature_matrix, mlb, top_n=10, min_rating=4.0)
    print_results("SCENARIO 2 — Fine-dining Japanese / Sushi (table booking)", res2)
    evaluate_recommendations(res2, user2)

    # ══════════════════════════════════════════════════════════════════
    # TEST SCENARIO 3 — North Indian food, mid-range price
    # ══════════════════════════════════════════════════════════════════
    user3 = {
        "cuisines":   ["North Indian", "Mughlai"],
        "price_range": 2,
        "min_rating":  3.0,
    }
    res3 = recommend(user3, df, feature_matrix, mlb, top_n=10, min_rating=3.0)
    print_results("SCENARIO 3 — Mid-range North Indian / Mughlai", res3)
    evaluate_recommendations(res3, user3)

    # ══════════════════════════════════════════════════════════════════
    # TEST SCENARIO 4 — Cafe & desserts, any budget
    # ══════════════════════════════════════════════════════════════════
    user4 = {
        "cuisines":        ["Cafe", "Desserts", "Bakery"],
        "online_delivery": True,
        "min_rating":      3.5,
    }
    res4 = recommend(user4, df, feature_matrix, mlb, top_n=10, min_rating=3.5)
    print_results("SCENARIO 4 — Cafe, Desserts & Bakery (delivery preferred)", res4)
    evaluate_recommendations(res4, user4)

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'═'*70}")
    print("  SUMMARY — Average Recommendation Quality across 4 Scenarios")
    print(f"{'═'*70}")
    summary_rows = []
    for label, res, prefs in [
        ("Budget Italian/Pizza",      res1, user1),
        ("Fine-dining Japanese/Sushi", res2, user2),
        ("Mid-range North Indian",    res3, user3),
        ("Cafe/Desserts/Bakery",      res4, user4),
    ]:
        m = evaluate_recommendations(res, prefs, verbose=False)
        summary_rows.append({
            "Scenario":        label,
            "Avg Rating":      m["avg_rating"],
            "Cuisine Hit %":   f"{m['cuisine_hit_rate']:.0%}" if m["cuisine_hit_rate"] is not None else "—",
            "Price Match %":   f"{m['price_match_rate']:.0%}" if m["price_match_rate"] is not None else "—",
            "Diversity Score": m["diversity_score"],
        })
    summary_df = pd.DataFrame(summary_rows).set_index("Scenario")
    print(summary_df.to_string())
    print()