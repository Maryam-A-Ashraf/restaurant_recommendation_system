# Restaurant Recommendation System

A Content-Based Restaurant Recommendation System built using Python and Scikit-Learn.

The system recommends restaurants based on user preferences such as:

- Preferred cuisines
- Budget / price range
- Online delivery availability
- Table booking availability
- Minimum rating

The recommendation engine uses feature engineering and cosine similarity to find restaurants that best match a user's profile.

---

## Dataset

Zomato Restaurant Dataset

- 9,551 restaurants
- 21 original features
- 144 unique cuisine tags

After preprocessing:

- Missing cuisine values handled
- Binary features encoded
- Numeric features normalized
- Unrated restaurants removed

Final usable restaurants:

- 7,403 restaurants

---

## Features

### Data Preprocessing

- Missing value handling
- Cuisine parsing
- Feature scaling using MinMaxScaler
- Binary encoding for:
  - Table Booking
  - Online Delivery
  - Delivering Now

### Feature Engineering

Restaurant vectors include:

- Cuisine One-Hot Encoding
- Price Range Encoding
- Table Booking
- Online Delivery
- Normalized Cost
- Normalized Rating

### Recommendation Engine

Recommendations are generated using:

- User Preference Profile
- Cosine Similarity
- Optional Minimum Rating Filtering

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn

---

## Evaluation Metrics

The system evaluates recommendation quality using:

- Cuisine Hit Rate
- Average Rating
- Price Match Rate
- Delivery Match Rate
- Diversity Score

---

## Results

### Dataset Statistics

| Metric | Value |
|----------|----------|
| Restaurants Loaded | 9,551 |
| Restaurants After Cleaning | 7,403 |
| Unique Cuisine Tags | 144 |
| Feature Matrix Size | 7,403 × 152 |

---

## Scenario 1: Budget Italian / Pizza Lover

Preferences:

- Italian
- Pizza
- Budget Price Range
- Online Delivery
- Minimum Rating 3.5

Results:

| Metric | Score |
|----------|----------|
| Cuisine Hit Rate | 100% |
| Average Rating | 4.05 |
| Price Match Rate | 70% |
| Delivery Match Rate | 40% |
| Diversity Score | 0.136 |

Top Recommendation:

**Monosoz (New Delhi)**

Similarity Score: 0.8586

---

## Scenario 2: Fine-Dining Japanese / Sushi Enthusiast

Preferences:

- Japanese
- Sushi
- Premium Budget
- Table Booking
- Minimum Rating 4.0

Results:

| Metric | Score |
|----------|----------|
| Cuisine Hit Rate | 100% |
| Average Rating | 4.37 |
| Price Match Rate | 100% |
| Diversity Score | 0.318 |

Top Recommendation:

**Guppy (New Delhi)**

Similarity Score: 0.9746

---

## Scenario 3: Mid-Range North Indian / Mughlai

Preferences:

- North Indian
- Mughlai
- Mid-Range Budget
- Minimum Rating 3.0

Results:

| Metric | Score |
|----------|----------|
| Cuisine Hit Rate | 100% |
| Average Rating | 3.04 |
| Price Match Rate | 100% |
| Diversity Score | 0.100 |

Top Recommendation:

**Shahi Zaikaa (New Delhi)**

Similarity Score: 0.965

---

## Scenario 4: Cafe, Desserts & Bakery Lover

Preferences:

- Cafe
- Desserts
- Bakery
- Online Delivery
- Minimum Rating 3.5

Results:

| Metric | Score |
|----------|----------|
| Cuisine Hit Rate | 100% |
| Average Rating | 4.16 |
| Delivery Match Rate | 80% |
| Diversity Score | 0.138 |

Top Recommendation:

**Bloomsbury's Boutique Cafe and Artisan Bakery (Kochi)**

Similarity Score: 0.8828

---

## Overall Performance

| Scenario | Avg Rating | Cuisine Hit | Price Match |
|------------|------------|------------|------------|
| Italian / Pizza | 4.05 | 100% | 70% |
| Japanese / Sushi | 4.37 | 100% | 100% |
| North Indian / Mughlai | 3.04 | 100% | 100% |
| Cafe / Desserts / Bakery | 4.16 | 100% | N/A |

### Key Observation

The recommendation engine achieved:

- 100% Cuisine Hit Rate across all test scenarios
- High average restaurant ratings
- Strong price-range matching
- Good recommendation diversity

---

## Installation

```bash
git clone https://github.com/yourusername/restaurant-recommendation-system.git

cd restaurant-recommendation-system

pip install -r requirements.txt
