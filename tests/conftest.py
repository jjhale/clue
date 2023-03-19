import os

import pytest

MAP_LOCATION = PARENT_DIR = (
    os.path.dirname(os.path.abspath(__file__)) + "/../clue/map49.csv"
)


@pytest.fixture
def map_csv_location() -> str:
    return MAP_LOCATION
