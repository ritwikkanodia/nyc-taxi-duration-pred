import pandas as pd
import numpy as np
from datetime import timedelta
import datetime as dt

def remove_outliers(df):
    # Refer to EDA notebook for the reasoning for choosing these specific filters
    df = df.query('trip_duration < 5900')
    df = df.query('passenger_count > 0')
    df = df.query('pickup_latitude > -100')
    df = df.query('pickup_latitude < 50')
    df['trip_duration'] = np.log(df['trip_duration'].values)

    return df

def encode_categorical_data(df, test):

    df = pd.concat([df, pd.get_dummies(df['store_and_fwd_flag'])], axis=1)
    test = pd.concat([test, pd.get_dummies(test['store_and_fwd_flag'])], axis=1)
    df = df.drop(['store_and_fwd_flag'], axis=1)

    df = pd.concat([df, pd.get_dummies(df['vendor_id'])], axis=1)
    test = pd.concat([test, pd.get_dummies(test['vendor_id'])], axis=1)
    df = df.drop(['vendor_id'], axis=1)

    return df, test

def convert_obj_to_ts(df, test):

    df['pickup_datetime'] = pd.to_datetime(df.pickup_datetime)
    test['pickup_datetime'] = pd.to_datetime(test.pickup_datetime)

    df = df.drop(['dropoff_datetime'], axis=1)

    return df, test

def create_date_features(df):

    df['month'] = df.pickup_datetime.dt.month
    df['week'] = df.pickup_datetime.dt.week
    df['weekday'] = df.pickup_datetime.dt.weekday
    df['hour'] = df.pickup_datetime.dt.hour
    df['minute'] = df.pickup_datetime.dt.minute
    df['minute_oftheday'] = df['hour'] * 60 + df['minute']
    df.drop(['minute'], axis=1, inplace=True)

    return df

def ft_haversine_distance(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def create_distance_features(df):

    df['distance'] = ft_haversine_distance(
                            df['pickup_latitude'].values,
                            df['pickup_longitude'].values, 
                            df['dropoff_latitude'].values,
                            df['dropoff_longitude'].values
                        )
    return df

def ft_degree(lat1, lng1, lat2, lng2):

    AVG_EARTH_RADIUS = 6371
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))

def create_direction_features(df):
    df['direction'] = ft_degree(
                            df['pickup_latitude'].values,
                            df['pickup_longitude'].values,
                            df['dropoff_latitude'].values,
                            df['dropoff_longitude'].values
                        )
    return df

def data_pre_feat_engg(df):

    df = df.query('distance < 200')
    df['speed'] = df.distance / df.trip_duration
    df = df.query('speed < 30')
    df = df.drop(['speed'], axis=1)
    y = df["trip_duration"]
    df = df.drop(["trip_duration"], axis=1)
    df = df.drop(['id'], axis=1)
    X = df
    
    return X, y


def main():

    df = pd.read_csv('../input/nyc-taxi-trip-duration/train.zip')
    test = pd.read_csv('../input/nyc-taxi-trip-duration/test.zip')

    df = remove_outliers(df)
    df, test = encode_categorical_data(df, test)
    df, test = convert_obj_to_ts(df, test)
    df, test = create_date_features(df), create_date_features(test)

    df.drop(['pickup_datetime'], axis=1, inplace=True)

    df, test = create_distance_features(df), create_distance_features(test)
    df, test = create_direction_features(df), create_direction_features(test)
    
    fr1 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_1.csv', usecols=['id', 'total_distance', 'total_travel_time',  'number_of_steps', ])
    fr2 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_2.csv', usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])
    test_street_info = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_test.csv',
                                   usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])
    
    train_street_info = pd.concat((fr1, fr2))
    df = df.merge(train_street_info, how='left', on='id')
    test = test.merge(test_street_info, how='left', on='id')

    X, y = data_pre_feat_engg(df)
    

    return X, y, test


if __name__ == '__main__':

    X, y, test = main()
    # print(X.head(10))
    # print(y.head(10))