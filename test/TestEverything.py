import unittest
import sys
sys.path.append("./test")

# noinspection PyUnresolvedReferences
from DatasetTests import DatasetTests
# noinspection PyUnresolvedReferences
from MorphTests import MorphTests


if __name__ == '__main__':
    unittest.main(verbosity=2)
