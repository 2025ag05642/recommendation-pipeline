# src/validation/validate.py

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check  
import os
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s'  
)  
logger = logging.getLogger(__name__)

PROCESSED_PATH = "data/processed/"  
ERRORS_PATH = "data/errors/"

def validate_ratings(df):  
    """Validate ratings dataframe"""  
    logger.info("Validating ratings data...")  
      
    errors = []  
    valid_df = df.copy()

    # Check 1: Missing values  
    missing = df.isnull().sum()  
    if missing.any():  
        logger.warning(f"Missing values found:\n{missing}")  
        errors.append({  
            'check': 'missing_values',  
            'details': str(missing.to_dict())  
        })  
        valid_df = valid_df.dropna()

    # Check 2: Duplicates  
    dupes = df.duplicated().sum()  
    if dupes > 0:  
        logger.warning(f"Found {dupes} duplicate records")  
        errors.append({  
            'check': 'duplicates',  
            'details': f'{dupes} duplicate rows found'  
        })  
        valid_df = valid_df.drop_duplicates()

    # Check 3: Invalid ratings  
    invalid_ratings = df[  
        (df['rating'] < 0.5) |   
        (df['rating'] > 5.0)  
    ]  
    if len(invalid_ratings) > 0:  
        logger.warning(f"Found {len(invalid_ratings)} invalid ratings")  
        errors.append({  
            'check': 'invalid_ratings',  
            'details': f'{len(invalid_ratings)} ratings outside 0.5-5.0'  
        })  
        valid_df = valid_df[  
            (valid_df['rating'] >= 0.5) &   
            (valid_df['rating'] <= 5.0)  
        ]

    # Check 4: Invalid timestamps  
    invalid_ts = df[df['timestamp'] <= 0]  
    if len(invalid_ts) > 0:  
        logger.warning(f"Found {len(invalid_ts)} invalid timestamps")  
        errors.append({  
            'check': 'invalid_timestamps',  
            'details': f'{len(invalid_ts)} invalid timestamps'  
        })  
        valid_df = valid_df[valid_df['timestamp'] > 0]

    # Check 5: Data types  
    try:  
        schema = DataFrameSchema({  
            'userId': Column(int, Check.greater_than(0)),  
            'movieId': Column(int, Check.greater_than(0)),  
            'rating': Column(float,   
                Check.in_range(0.5, 5.0)),  
            'timestamp': Column(int,   
                Check.greater_than(0))  
        })  
        schema.validate(df)  
        logger.info("✅ Data types validation passed!")  
    except Exception as e:  
        logger.warning(f"Data type issues: {e}")  
        errors.append({  
            'check': 'data_types',  
            'details': str(e)  
        })

    return valid_df, errors

def validate_movies(df):  
    """Validate movies dataframe"""  
    logger.info("Validating movies data...")  
      
    errors = []  
    valid_df = df.copy()

    # Check 1: Missing values  
    missing = df.isnull().sum()  
    if missing.any():  
        logger.warning(f"Missing values:\n{missing}")  
        errors.append({  
            'check': 'missing_values',  
            'details': str(missing.to_dict())  
        })  
        valid_df = valid_df.dropna()

    # Check 2: Duplicates  
    dupes = df.duplicated().sum()  
    if dupes > 0:  
        logger.warning(f"Found {dupes} duplicates")  
        errors.append({  
            'check': 'duplicates',  
            'details': f'{dupes} duplicate rows'  
        })  
        valid_df = valid_df.drop_duplicates()

    # Check 3: Missing genres  
    no_genres = df[df['genres'] == '(no genres listed)']  
    if len(no_genres) > 0:  
        logger.warning(  
            f"Found {len(no_genres)} movies with no genres"  
        )  
        errors.append({  
            'check': 'missing_genres',  
            'details': f'{len(no_genres)} movies have no genres'  
        })

    return valid_df, errors

def check_referential_integrity(ratings, movies):  
    """Check referential integrity between ratings and movies"""  
    logger.info("Checking referential integrity...")  
      
    errors = []  
      
    # Check if all movieIds in ratings exist in movies  
    rating_movies = set(ratings['movieId'].unique())  
    movie_ids = set(movies['movieId'].unique())  
      
    orphan_movies = rating_movies - movie_ids  
    if orphan_movies:  
        logger.warning(  
            f"Found {len(orphan_movies)} movieIds in ratings "  
            f"not in movies!"  
        )  
        errors.append({  
            'check':'referential_integrity',  
            'details': f'{len(orphan_movies)} orphan movieIds'  
        })  
    else:  
        logger.info("✅ Referential integrity check passed!")  
      
    return errors

def save_error_log(all_errors):  
    """Save error log to errors folder"""  
    os.makedirs(ERRORS_PATH, exist_ok=True)  
      
    if all_errors:  
        error_df = pd.DataFrame(all_errors)  
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  
        error_path = os.path.join(
            ERRORS_PATH,
            f"validation_errors_{timestamp}.csv"
        )
        error_df.to_csv(error_path, index=False)
        logger.info(f"Error log saved to {error_path}")
    else:
        logger.info("✅ No errors found!")

def validate_all():
    """Main validation function"""
    logger.info("="*50)
    logger.info("Starting Data Validation...")
    logger.info("="*50)

    # Load data
    ratings = pd.read_parquet(
        os.path.join(PROCESSED_PATH, "ratings.parquet")
    )
    movies = pd.read_parquet(
        os.path.join(PROCESSED_PATH, "movies.parquet")
    )

    # Validate
    valid_ratings, rating_errors = validate_ratings(ratings)
    valid_movies, movie_errors = validate_movies(movies)

    # Referential integrity
    ref_errors = check_referential_integrity(  
        valid_ratings, valid_movies  
    )

    # Collect all errors
    all_errors = rating_errors + movie_errors + ref_errors

    # Save error log
    save_error_log(all_errors)

    # Summary
    logger.info("="*50)
    logger.info("Validation Summary:")
    logger.info(f"Ratings: {len(ratings):,} → {len(valid_ratings):,} valid")
    logger.info(f"Movies: {len(movies):,} → {len(valid_movies):,} valid")
    logger.info(f"Total issues found: {len(all_errors)}")
    logger.info("="*50)

    return valid_ratings, valid_movies, all_errors

if __name__ == "__main__":  
    validate_all()
