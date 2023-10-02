import pytest

from hat.exceptions import UserError
from hat.tools.hydrostats_cli import check_inputs, parse_functions


def test_check_inputs():
    function = "kge"
    functions = "kge,rmse"
    sims = "filepath/filename.nc"
    obs = sims

    assert check_inputs(function, sims, obs)
    assert check_inputs(functions, sims, obs)

    with pytest.raises(UserError):
        check_inputs("", sims, obs)

    with pytest.raises(UserError):
        check_inputs(function, "", obs)

    with pytest.raises(UserError):
        check_inputs(function, sims, "")

    with pytest.raises(UserError):
        check_inputs(function, sims.replace(".nc", ""), obs)

    with pytest.raises(UserError):
        check_inputs(function, sims, obs.replace(".nc", ""))


def test_parse_functions():
    valid_function = "kge"
    valid_functions = "kge,rmse"

    invalid_function = "not_a_function"
    invalid_functions = "not_a_function, also_not_a_function"

    assert parse_functions(valid_function) == ["kge"]
    assert parse_functions(valid_functions) == ["kge", "rmse"]

    with pytest.raises(UserError):
        parse_functions(invalid_function)

    with pytest.raises(UserError):
        parse_functions(invalid_functions)
