"""
Module that contains the get trades functionality. This module will generate a random set of dummy positions.
"""
import os
import random
import logging
import datetime
import uuid
import numpy as np
import pandas as pd

# If you intend to use PySpark, set use_pyspark to True (1)
use_pyspark = False

# Configure logging
log_filename = "power_trades.log"
logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def check_if_valid_time_format(time_str):
    try:
        datetime.datetime.strptime(time_str, "%H:%M")
        return True
    except ValueError:
        return False


def check_start_and_end_time(dataf):
    # Check if 'Local Time' column exists and dataframe is not empty
    if "Local Time" not in dataf.columns or dataf.empty:
        return False

    # Assuming "time" column is in the format HH:MM
    start_time = dataf['Local Time'].iloc[0]
    end_time = dataf['Local Time'].iloc[-1]

    # Convert start_time and end_time to datetime objects
    start_time = datetime.datetime.strptime(start_time, '%H:%M').time()
    end_time = datetime.datetime.strptime(end_time, '%H:%M').time()

    # Check if start time is 23:00 and end time is 22:55 (previous day)
    if start_time == datetime.time(23, 0) and end_time == datetime.time(22, 55):
        return True
    else:
        return False


def check_time_interval(dataf):
    # Check if the time intervals are 5 minutes apart
    time_intervals = pd.to_datetime(dataf['Local Time'], format='%H:%M')
    expected_intervals = pd.date_range(start=time_intervals.min(), end=time_intervals.max(), freq='5min')
    return time_intervals.equals(expected_intervals)


def check_if_valid_date(date: str):
    """
    Verify that the date format matches d/m/y
    :param date: str date in d/m/y format
    :return: True or False
    """
    date_format = "%d/%m/%Y"

    """ Warning to any non python devs reading this code..
        In Python the only way to test a valid date is with a try catch. Yep, it sux.
    """
    if not isinstance(date, str):
        return False

    try:
        datetime.datetime.strptime(date, date_format)
        valid_date = True
    except ValueError:
        valid_date = False

    return valid_date


def random_nan(x):
    """
    Replace x with a nan, if the random number == 1
    """
    if random.randrange(0, 15) == 1:
        x = np.nan

    return x


def generate_new_random_trade_position(date: str):
    """ Generates a new random trade position with the date, period sequence and volume sequence
    :param date: Date in d/m/y format
    :return: dict with data
    """

    period_list = [random_nan(i.strftime("%H:%M")) for i in pd.date_range("00:00", "23:59", freq="5min").time]
    volume = [random_nan(x) for x in random.sample(range(0, 500), len(period_list))]

    open_trade_position = {"date": date,
                           "time": period_list,
                           "volume": volume,
                           "id": uuid.uuid4().hex
                           }

    return open_trade_position


def get_trades(date: str):
    """
    Generate some random number of open trade positions
    :param date: date in d/m/y format
    :return:
    """

    if not check_if_valid_date(date=date):
        error_msg = "The supplied date {} is invalid.Please supply a date in the format d/m/Y.".format(date)
        logging.error(error_msg)
        raise ValueError(error_msg)

    # a randomly chosen number of open trades
    number_of_open_trades = random.randint(1, 101)
    logging.info("Generated" + str(number_of_open_trades) + " open trades randomly.")

    open_trades_list = []
    # Generate a list of open trade dicts
    for open_trade in range(0, number_of_open_trades):
        open_trades_list.append(generate_new_random_trade_position(date=date))

    # Create a Pandas DataFrame from the trade data
    trade_df = pd.DataFrame(open_trades_list)

    # Handle missing values by filling them with zeros
    trade_df['volume'].fillna(0, inplace=True)
    return trade_df


def parse_time(time_str):
    if pd.notna(time_str):
        try:
            return datetime.datetime.strptime(time_str, "%H:%M").time()
        except ValueError:
            pass
    return None


def fill_missing_values(volume_series):
    # Forward fill missing values in the volume series
    return volume_series.ffill()


def flatten_and_parse_time(trade_df):
    # Explode the "time" and "volume" columns to create multiple rows for each element in the lists
    exploded_df = trade_df.explode(['time', 'volume'])

    # Apply the parse_time function element-wise to the "time" column
    exploded_df['parsed_time'] = exploded_df['time'].apply(lambda x: parse_time(x))

    # Filter out rows with invalid or NaN parsed_time values
    exploded_df = exploded_df.dropna(subset=['parsed_time'])

    # Create a new datetime column by combining the date with parsed_time
    exploded_df['full_datetime'] = pd.to_datetime(exploded_df['date'] + ' ' + exploded_df['parsed_time'].astype(str))

    # Adjust the datetime values to start from 23:00 (11 pm) on the previous day
    exploded_df['full_datetime'] = exploded_df['full_datetime'] - pd.DateOffset(hours=1)

    # Fill missing values in the "volume" column with sensible values (e.g., forward fill)
    exploded_df['volume'] = fill_missing_values(exploded_df['volume'])

    # Group the data by hour and sum the trade volumes
    aggregated_df = exploded_df.groupby(exploded_df['full_datetime'].dt.hour)['volume'].sum().reset_index()

    # Rename columns for clarity
    aggregated_df.rename(columns={'full_datetime': 'Local Time', 'volume': 'Volume'}, inplace=True)

    # Convert the "Local Time" column to datetime format
    aggregated_df['Local Time'] = pd.to_datetime(aggregated_df['Local Time'], format='%H')

    # Format the "Local Time" column as HH:MM (24-hour format)
    aggregated_df['Local Time'] = aggregated_df['Local Time'].dt.strftime('%H:%M')

    return aggregated_df


# Define a function to generate data profiling report
def generate_data_profiling_report(dataf, filename):
    profiling_report = pd.DataFrame({
        "Column Name": dataf.columns,
        "Data Type": dataf.dtypes,
        "Total Rows": len(dataf),
        "Missing Values": dataf.isnull().sum(),
        "Unique Values": dataf.nunique()
    })

    profiling_report.to_csv(filename, index=False)


# Define a function to generate data quality report
def generate_data_quality_report(dataf, filename):
    data_quality_report = pd.DataFrame({
        "Check": ["Correct Time Format", "Start and End Time", "Missing Values", "Time Interval Check"],
        "Status": ["Pass" if check_passes else "Fail" for check_passes in [
            all(dataf['Local Time'].apply(check_if_valid_time_format)),
            check_start_and_end_time(dataf),
            not dataf.isnull().values.any(),
            check_time_interval(dataf)
        ]]
    })

    data_quality_report.to_csv(filename, index=False)


if __name__ == '__main__':
    # Configure log file handler
    log_filename = "power_trades.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logging.getLogger().addHandler(file_handler)

    logging.info("Starting data extraction process.")

    try:
        trades = get_trades(date='01/03/2022')
        logging.info("Data extraction completed successfully.")

        trades = get_trades(date='01/03/2022')
        aggregated_trade_data = flatten_and_parse_time(trades)

        # Get the current local time for the extraction
        extraction_time = datetime.datetime.now().strftime('%Y%m%d_%H%M')

        print(aggregated_trade_data)

        # Define the CSV filename with the current local time
        csv_filename = f'PowerPosition_{extraction_time}.csv'

        # Specify the location where the CSV should be saved (you can pass this as a parameter)
        output_location = r'C:/Users/Ali/PycharmProjects/Quizes/python-powerservice1'
        # Create the full path for saving the CSV file
        csv_path = os.path.join(output_location, csv_filename)

        # Save the aggregated data to a CSV file with the desired format and filename
        aggregated_trade_data.to_csv(csv_path, index=False, header=True)

        # If using PySpark, also save the data in Delta format (optional)
        if use_pyspark:  # The condition to check if PySpark is used
            # from delta import DeltaTable
            # from pyspark.sql import SparkSession

            # Initialize a Spark session
            # spark = SparkSession.builder.appName("PowerPositionDelta").getOrCreate()

            # Create a PySpark DataFrame from the Pandas DataFrame
            # pyspark_df = spark.createDataFrame(aggregated_trade_data)

            # Save the PySpark DataFrame in Delta format
            # delta_table = DeltaTable.forPath(spark, 'path/to/delta/output/directory')
            # pyspark_df.write.format("delta").mode("append").save("path/to/delta/output/directory")
            pass

        # Generate data profiling report filename
        data_profiling_filename = csv_path.replace(".csv", "_data_profiling.csv")

        # Generate data quality report filename
        data_quality_filename = csv_path.replace(".csv", "_data_quality.csv")

        # Generate data profiling report
        generate_data_profiling_report(aggregated_trade_data, data_profiling_filename)

        # Generate data quality report
        generate_data_quality_report(aggregated_trade_data, data_quality_filename)


    except Exception as e:
        logging.error("An error occurred during data extraction: %s", str(e))
        # Handle the error as needed

    logging.info("Data processing finished.")
