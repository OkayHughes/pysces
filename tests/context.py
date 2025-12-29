import sys
import os


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def get_figdir():
  return os.path.join(os.path.dirname(os.path.abspath(__file__)), "_figures")

extensive = False
test_division_factor = 1.0 if extensive else 1000.0 
test_npts = [3, 4, 5, 6] if extensive else [3]
