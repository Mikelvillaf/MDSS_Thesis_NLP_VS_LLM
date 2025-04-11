import os
import pytest
import pandas as pd
from scripts.data_loader import load_reviews

TEST_FILE = "data/Digital_Music.jsonl.gz"

@pytest.mark.skipif(not os.path.exists(TEST_FILE), reason="Test file not available.")
def test_load_reviews_max_rows():
    df = load_reviews(TEST_FILE, max_rows=100)
    assert isinstance(df, pd.DataFrame)
    assert len(df) <= 100

@pytest.mark.skipif(not os.path.exists(TEST_FILE), reason="Test file not available.")
def test_year_filtering():
    df = load_reviews(TEST_FILE, year_range=[2018, 2020], max_rows=500)
    assert 'year' in df.columns
    assert df['year'].between(2018, 2020).all()

@pytest.mark.skipif(not os.path.exists(TEST_FILE), reason="Test file not available.")
def test_missing_file_raises_error():
    with pytest.raises(FileNotFoundError):
        load_reviews("data/fake_file.jsonl")