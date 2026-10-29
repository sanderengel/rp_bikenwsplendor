import argparse
from splendor.config import STUDY_AREA

def get_study_area() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--area", default=STUDY_AREA)
    return parser.parse_args().area
