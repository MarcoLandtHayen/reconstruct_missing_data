## Working on nesh with Container image py-da-tf-shap.sif:
import sys

import pytest

from reconstruct_missing_data.dummy_module import dummy_foo


# sys.path.append("./reconstruct_missing_data")

## Working on nesh with Container image reconstruct_missing_data_latest.sif:
# from reconstruct_missing_data.dummy_module import dummy_foo


def test_dummy():
    assert dummy_foo(4) == (4 + 4)
