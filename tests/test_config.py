# import os
# import warnings

# import pytest

# # NOTE setup.py and pkg_resources are DEPRECATED
# # DeprecationWarning: pkg_resources is deprecated as an API.
# # See https://setuptools.pypa.io/en/latest/pkg_resources.html
# with warnings.catch_warnings():
#     warnings.filterwarnings("ignore", category=DeprecationWarning)
#     from hat.config import DEFAULT_CONFIG, booleanify, read_config, valid_custom_config


# def test_DEFAULT_CONFIG():
#     assert DEFAULT_CONFIG


# def test_booleanify():
#     true_string_boolean = {"key": "true"}
#     false_string_boolean = {"key": "false"}
#     not_string_boolean = {"key": 123}

#     true_boolean = booleanify(true_string_boolean, "key")
#     false_boolean = booleanify(false_string_boolean, "key")

#     assert true_boolean["key"]
#     assert not false_boolean["key"]

#     with pytest.raises(ValueError):
#         _ = booleanify(not_string_boolean, "key")


# def test_valid_custom_config():
#     empty_dict = {}
#     invalid_keys = {"not_a_valid_key": 123}
#     valid_keys = list(DEFAULT_CONFIG.keys())
#     valid_key = valid_keys[0]
#     partially_complete = {valid_key: "new_value"}

#     assert valid_custom_config(empty_dict) == DEFAULT_CONFIG
#     assert valid_custom_config(invalid_keys) == DEFAULT_CONFIG
#     assert valid_custom_config(partially_complete).keys() == DEFAULT_CONFIG.keys()


# def test_read_config():
#     empty_path = ""
#     filepath_exists_but_is_not_json = os.path.abspath(__file__)

#     assert DEFAULT_CONFIG == read_config(empty_path)

#     with pytest.raises(ValueError):
#         _ = read_config(filepath_exists_but_is_not_json)
