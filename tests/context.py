import sys
import os


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def get_figdir():
  return os.path.join(os.path.dirname(os.path.abspath(__file__)), "_figures")