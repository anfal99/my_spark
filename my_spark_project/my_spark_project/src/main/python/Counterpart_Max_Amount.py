from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def create_spark_session(app_name="CounterpartyMaxAmount"):
    """
    Create a Spark session with configurable settings.
    """
    spark = SparkSession.builder.appName(app_name) \
        .config("spark.sql.shuffle.partitions", "4") \
        .getOrCreate()
    return spark


def load_data(spark, data_path):
    """
    Load data from a CSV file into a Spark DataFrame.
    """
    df = spark.read.csv(data_path, header=True, inferSchema=True)
    return df


def find_max_counterparty_amount(df):
    """
    Find the counterparty with the most amount of money transferred for each user.
    """
    result_df = df.groupBy("user_id", "counterparty_id") \
        .agg(F.sum("amount").alias("total_amount")) \
        .withColumn("max_amount", F.max("total_amount").over(Window.partitionBy("user_id")))

    result_df = result_df.filter(result_df["total_amount"] == result_df["max_amount"]) \
        .select("user_id", "counterparty_id", "total_amount")

    return result_df


def main():
    data_path = "data/Transaction.csv"

    try:
        # Create a Spark session
        spark = create_spark_session()

        # Load the data
        df = load_data(spark, data_path)

        # Find the max counterparty amounts
        result_df = find_max_counterparty_amount(df)

        # Show the final result
        result_df.show()

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()

