import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format, dayofweek
from pyspark.sql.types import TimestampType, DateType


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS CREDS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS CREDS']['AWS_SECRET_ACCESS_KEY']
input_data = config['IO SOURCE']['input_data']
output_data = config['IO SOURCE']['output_data']


def create_spark_session():
    """"
    Get/ create spark session
    
    Arguments: 
        None
    Returns: 
        Spark session
    """"
    
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark):
    """
    Extract and load song data from S3 then transform it songs and artists parquet files in S3. 
    Songs parquet files are partitioned by year and artist_id.
    
    Arguments:
        Spark session
        
    Returns:
        None
    """
    
    # get filepath to song data file
    song_data = input_data + "song_data/*/*/*/*.json"
    
    # read song data file
    df = spark.read.json(song_data)

    # extract columns to create songs table
    songs_table = df.select("song_id", "title", "artist_id", "year", "duration").drop_duplicates()
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.partitionBy("year", "artist_id").mode('overwrite').parquet(output_data + "songs/songs.parquet")

    # extract columns to create artists table
    artists_table = df.select("artist_id", "artist_name", "artist_location", "artist_latitude", "artist_longitude").drop_duplicates()
    
    # write artists table to parquet files
    artists_table.write.mode('overwrite').parquet(output_data + "artists/artists.parquet")


def process_log_data(spark):
    """
    Extract and load log data from S3 then transform it users, times, and songplays parquet files in S3. 
    Times and songplays parquet files are partitioned by year and month
    
    Arguments:
        Spark session
    
    Returns:
        None
    """
    
    # get filepath to log data file
    log_data = input_data + "log_data/*/*/*.json"

    # read log data file
    df = spark.read.json(log_data)
    
    # filter by actions for song plays
    df = df.filter(df.page == "NextSong")

    # extract columns for users table    
    users_table = df.select("userId", "firstName", "lastName", "gender", "level")
    
    # write users table to parquet files
    users_table.write.mode('overwrite').parquet(output_data + "users/users.parquet")

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x:datetime.fromtimestamp(x/1000), TimestampType())
    df = df.withColumn("timestamp", get_timestamp(df.ts))
    
    # create datetime column from original timestamp column
    get_datetime = udf(lambda x:datetime.fromtimestamp(x/1000), DateType())
    df = df.withColumn("datetime", get_datetime(df.ts))
    
    # extract columns to create time table
    time_table = df.select(col("timestamp").alias("start_time"), 
                           hour(df.timestamp).alias("hour"), 
                           dayofmonth(df.timestamp).alias("day"), 
                           weekofyear(df.timestamp).alias("week"),
                           month(df.timestamp).alias("month"),
                           year(df.timestamp).alias("year"),
                           dayofweek(df.timestamp).alias("weekday")).drop_duplicates()
    
    # write time table to parquet files partitioned by year and month
    time_table.write.partitionBy("year", "month").mode('overwrite').parquet(output_data + "times/times.parquet")

    # read in song data to use for songplays table
    song_df = spark.read.parquet(output_data + "songs/songs.parquet")

    # read in artist data to use for songplays table
    artist_df = spark.read.parquet(output_data + "artists/artists.parquet")
    
    # extract columns from joined song, artist, and log datasets to create songplays table
    joinedSongArtist_df = song_df.join(artist_df, ['artist_id'])
    songplays_table = df.join(joinedSongArtist_df, (df.song == joinedSongArtist_df.title) & (df.artist == joinedSongArtist_df.artist_name) & (df.length == joinedSongArtist_df.duration), "left").select(col("timestamp").alias("start_time"), year("timestamp").alias("year"), month("timestamp").alias("month"), col("userId").alias("user_id"), df.level, joinedSongArtist_df.song_id, col("artist_id"), col("sessionId").alias("session_id"), df.location, col("userAgent").alias("user_agent")).drop_duplicates()
    
    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.partitionBy("year", "month").mode('overwrite').parquet(output_data + "songplays/songplays.parquet")


def main():
    spark = create_spark_session()
    
    process_song_data(spark)    
    process_log_data(spark)


if __name__ == "__main__":
    main()
