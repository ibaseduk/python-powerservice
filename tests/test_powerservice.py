""" Tests for the powerservice"""
import pytest
import pandas as pd
from powerservice.trading import (
    check_if_valid_date,
    check_if_valid_time_format,
    check_start_and_end_time,
    check_time_interval,
    generate_new_random_trade_position,
)

cases = [("", False),
         ("01/04/2015", True),
         ("30/04/2015", True),
         (1, False),
         ("01/30/2015", False),
         ("30/04/bla", False)
         ]



@pytest.mark.parametrize("date,expected", cases)
def test_date_checker(date, expected):
    """ Check that only d/m/y formatted date is accepted"""
    assert check_if_valid_date(date) == expected

#
# def test_generate_new_random_trade_position_period():
#     """ Try to generate a new random trade position"""
#     new_trade = generate_new_random_trade_position(date="01/04/2015")
#     period_list = new_trade["period"]
#
#     assert period_list[0] == 1 and period_list[-1] == 24

def test_generate_new_random_trade_position_time_series_len():
    """Check that the period and volume series are of the same length"""
    new_trade = generate_new_random_trade_position(date="01/04/2015")
    period_list = new_trade["time"]
    volume_list = new_trade["volume"]

    assert len(period_list) == len(volume_list)

def test_check_if_valid_time_format():
    assert check_if_valid_time_format("12:34") == True
    assert check_if_valid_time_format("12:60") == False
    assert check_if_valid_time_format("12-34") == False

def test_check_start_and_end_time():
    # Create a DataFrame with a time series
    df = pd.DataFrame({
        'Local Time': ['23:00', '23:05', '00:00', '01:00', '02:00', '22:55']
    })
    assert check_start_and_end_time(df) == True

def test_check_time_interval():
    # Create a DataFrame with a time series
    df = pd.DataFrame({
        'Local Time': ['00:00', '00:05', '00:10', '00:15', '00:25', '00:30']
    })
    assert check_time_interval(df) == False
