from sklearn.decomposition import PCA
import pandas
import joblib
import numpy
import lightgbm
from sklearn.cluster import MiniBatchKMeans
import datetime as dt

Radius_of_Earth = 6371  # in km

# Calculating Distance between two points identified by their latitude and longitude
def calculate_haversine_distance(latitude1, longitude1, latitude_2, longitude_2):

    latitude1, longitude1, latitude_2, longitude_2 = map(numpy.radians, (latitude1, longitude1, latitude_2, longitude_2))
    longitude_difference = longitude_2 - longitude1
    latitude_difference = latitude_2 - latitude1
    depth = numpy.sin(latitude_difference * 0.5) ** 2 + numpy.cos(latitude1) * numpy.cos(latitude_2) * numpy.sin(longitude_difference * 0.5) ** 2
    height = 2 * Radius_of_Earth * numpy.arcsin(numpy.sqrt(depth))
    return height

def calculate_rmsle(predicted_value, real_value):
    total = 0.0
    for i in range(len(predicted_value)):
        real_value = real_value[i]
        predicted_value = predicted_value[i]
        if predicted_value < 0:
            predicted_value = 0
        if real_value < 0:
            real_value = 0
        real_value = numpy.log(real_value + 1)
        predicted_value = numpy.log(predicted_value + 1)
        total = total + (predicted_value - real_value) ** 2
    return (total / len(predicted_value)) ** 0.5

def array_bearing(latitude1, longitude1, latitude2, longitude2):
    longitude_difference_in_radian = numpy.radians(longitude2 - longitude1)
    latitude1, longitude1, latitude2, longitude2 = map(numpy.radians, (latitude1, longitude1, latitude2, longitude2))
    height = numpy.sin(longitude_difference_in_radian) * numpy.cos(latitude2)
    length = numpy.cos(latitude1) * numpy.sin(latitude2) - numpy.sin(latitude1) * numpy.cos(latitude2) * numpy.cos(longitude_difference_in_radian)
    return numpy.degrees(numpy.arctan2(height, length))

def generate_bagged_set(values, labels, seed_set, approximators, value_t, label_t=None):
    prediction_on_bagged_set = numpy.array([0.0 for d in range(0, value_t.shape[0])])
    for i in range(0, approximators):
        parameters = {
                  'objective': 'fair',
                  'metric': 'rmse',
                  'fair_c': 1.5,
                  'feature_fraction': 0.7,
                  'learning_rate': 0.2,
                  'verbose': 0,
                  'num_leaves': 75,
                  'bagging_freq': 1,
                  'bagging_fraction': 0.95,
                  'bagging_seed': seed_set + i,
                  'feature_fraction_seed': seed_set + i,
                  'min_data_in_leaf': 10,
                  'max_bin': 255,
                  'max_depth': 10,
                  'reg_alpha': 20,
                  'boosting': 'gbdt',
                  'reg_lambda': 20,
                  'num_threads': 45,
                  'lambda_l2': 20,
                  }
        num_boost_round = 5000
        lightgbm_train = lightgbm.Dataset(values, numpy.log1p(labels), free_raw_data=False)
        if type(label_t) != type(None):
            lightgbm_cv = lightgbm.Dataset(value_t, numpy.log1p(label_t), free_raw_data=False, reference=lightgbm_train)
            lightgbm_model = lightgbm.train(parameters, lightgbm_train, num_boost_round=num_boost_round,
                              valid_sets=lightgbm_cv,
                              verbose_eval=True)
        else:
            lightgbm_model = lightgbm.train(parameters, lightgbm_train, num_boost_round=num_boost_round)
            lightgbm_cv = lightgbm.Dataset(value_t, free_raw_data=False)
        predictions = numpy.expm1(lightgbm_model.predict(value_t))
        prediction_on_bagged_set += predictions
        print("Execution complete: " + str(i))
    prediction_on_bagged_set /= approximators
    return prediction_on_bagged_set

def calculate_manhattan_difference(latitutde1, longitude1, latitude2, longitude2):
    difference_1 = calculate_haversine_distance(latitutde1, longitude1, latitude2, longitude1)
    difference_2 = calculate_haversine_distance(latitutde1, longitude1, latitutde1, longitude2)
    return difference_2 + difference_1

def date_time_conversion(train_data_frame, test_data_frame):
    test_data_frame['pickup_datetime'] = pandas.to_datetime(test_data_frame.pickup_datetime)
    train_data_frame['pickup_datetime'] = pandas.to_datetime(train_data_frame.pickup_datetime)
    test_data_frame.loc[:, 'pickup_date'] = test_data_frame['pickup_datetime'].dt.date
    train_data_frame.loc[:, 'pickup_date'] = train_data_frame['pickup_datetime'].dt.date
    train_data_frame['dropoff_datetime'] = pandas.to_datetime(train_data_frame.dropoff_datetime)

    test_data_frame.loc[:, 'pickup_weekday'] = test_data_frame['pickup_datetime'].dt.weekday
    test_data_frame.loc[:, 'pickup_hour_weekofyear'] = test_data_frame['pickup_datetime'].dt.weekofyear
    test_data_frame.loc[:, 'pickup_hour'] = test_data_frame['pickup_datetime'].dt.hour
    test_data_frame.loc[:, 'pickup_minute'] = test_data_frame['pickup_datetime'].dt.minute
    test_data_frame.loc[:, 'pickup_dt'] = (test_data_frame['pickup_datetime'] - train_data_frame['pickup_datetime'].min()).dt.total_seconds()
    test_data_frame.loc[:, 'pickup_week_hour'] = test_data_frame['pickup_weekday'] * 24 + test_data_frame['pickup_hour']

    train_data_frame.loc[:, 'pickup_weekday'] = train_data_frame['pickup_datetime'].dt.weekday
    train_data_frame.loc[:, 'pickup_hour_weekofyear'] = train_data_frame['pickup_datetime'].dt.weekofyear
    train_data_frame.loc[:, 'pickup_hour'] = train_data_frame['pickup_datetime'].dt.hour
    train_data_frame.loc[:, 'pickup_minute'] = train_data_frame['pickup_datetime'].dt.minute
    train_data_frame.loc[:, 'pickup_dt'] = (train_data_frame['pickup_datetime'] - train_data_frame['pickup_datetime'].min()).dt.total_seconds()
    train_data_frame.loc[:, 'pickup_week_hour'] = train_data_frame['pickup_weekday'] * 24 + train_data_frame['pickup_hour']

    train_data_frame.loc[:, 'pickup_dayofyear'] = train_data_frame['pickup_datetime'].dt.dayofyear
    test_data_frame.loc[:, 'pickup_dayofyear'] = test_data_frame['pickup_datetime'].dt.dayofyear

def calculation_of_distance(train_data_frame, test_data_frame):
    test_data_frame.loc[:, 'distance_haversine'] = calculate_haversine_distance(test_data_frame['pickup_latitude'].values,
                                                                                test_data_frame['pickup_longitude'].values,
                                                                                test_data_frame['dropoff_latitude'].values,
                                                                                test_data_frame['dropoff_longitude'].values)
    train_data_frame.loc[:, 'distance_haversine'] = calculate_haversine_distance(train_data_frame['pickup_latitude'].values,
                                                                                 train_data_frame['pickup_longitude'].values,
                                                                                 train_data_frame['dropoff_latitude'].values,
                                                                                 train_data_frame['dropoff_longitude'].values)
def calculation_of_bearing(train_data_frame, test_data_frame):
    test_data_frame.loc[:, 'direction'] = array_bearing(test_data_frame['pickup_latitude'].values, test_data_frame['pickup_longitude'].values,
                                                        test_data_frame['dropoff_latitude'].values, test_data_frame['dropoff_longitude'].values)
    train_data_frame.loc[:, 'direction'] = array_bearing(train_data_frame['pickup_latitude'].values, train_data_frame['pickup_longitude'].values,
                                                         train_data_frame['dropoff_latitude'].values, train_data_frame['dropoff_longitude'].values)

def calculate_difference_in_distance(train_data_frame, test_data_frame):
    test_data_frame.loc[:, 'distance_dummy_manhattan'] = calculate_manhattan_difference(test_data_frame['pickup_latitude'].values,
                                                                                        test_data_frame['pickup_longitude'].values,
                                                                                        test_data_frame['dropoff_latitude'].values,
                                                                                        test_data_frame['dropoff_longitude'].values)
    train_data_frame.loc[:, 'distance_dummy_manhattan'] = calculate_manhattan_difference(train_data_frame['pickup_latitude'].values,
                                                                                         train_data_frame['pickup_longitude'].values,
                                                                                         train_data_frame['dropoff_latitude'].values,
                                                                                         train_data_frame['dropoff_longitude'].values)


def perform_prinicipal_component_analysis(train_data_frame, test_data_frame):
    principal_component_analysis = PCA().fit(co_ordinates)

    test_data_frame['pickup_pca0'] = principal_component_analysis.transform(test_data_frame[['pickup_latitude', 'pickup_longitude']])[:, 0]
    test_data_frame['pickup_pca1'] = principal_component_analysis.transform(test_data_frame[['pickup_latitude', 'pickup_longitude']])[:, 1]
    test_data_frame['dropoff_pca0'] = principal_component_analysis.transform(test_data_frame[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
    test_data_frame['dropoff_pca1'] = principal_component_analysis.transform(test_data_frame[['dropoff_latitude', 'dropoff_longitude']])[:, 1]
    train_data_frame['pickup_pca0'] = principal_component_analysis.transform(train_data_frame[['pickup_latitude', 'pickup_longitude']])[:, 0]
    train_data_frame['pickup_pca1'] = principal_component_analysis.transform(train_data_frame[['pickup_latitude', 'pickup_longitude']])[:, 1]
    train_data_frame['dropoff_pca0'] = principal_component_analysis.transform(train_data_frame[['dropoff_latitude', 'dropoff_longitude']])[:,
                            0]
    train_data_frame['dropoff_pca1'] = principal_component_analysis.transform(train_data_frame[['dropoff_latitude', 'dropoff_longitude']])[:,
                            1]

    test_data_frame.loc[:, 'pca_manhattan'] = numpy.abs(test_data_frame['dropoff_pca1'] - test_data_frame['pickup_pca1']) + numpy.abs(
        test_data_frame['dropoff_pca0'] - test_data_frame['pickup_pca0'])
    train_data_frame.loc[:, 'pca_manhattan'] = numpy.abs(train_data_frame['dropoff_pca1'] - train_data_frame['pickup_pca1']) + numpy.abs(
        train_data_frame['dropoff_pca0'] - train_data_frame['pickup_pca0'])

def perform_clustering(train_data_frame, test_data_frame):
    co_ordinates = numpy.vstack((train[['pickup_latitude', 'pickup_longitude']].values,
                                 train[['dropoff_latitude', 'dropoff_longitude']].values,
                                 test[['pickup_latitude', 'pickup_longitude']].values,
                                 test[['dropoff_latitude', 'dropoff_longitude']].values))

    temp_index = numpy.random.permutation(len(co_ordinates))[:600000]
    perform_k_means_clustering = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(co_ordinates[temp_index])
    test_data_frame.loc[:, 'pickup_cluster'] = perform_k_means_clustering.predict(test_data_frame[['pickup_latitude', 'pickup_longitude']])
    test_data_frame.loc[:, 'dropoff_cluster'] = perform_k_means_clustering.predict(test_data_frame[['dropoff_latitude', 'dropoff_longitude']])
    train_data_frame.loc[:, 'pickup_cluster'] = perform_k_means_clustering.predict(train_data_frame[['pickup_latitude', 'pickup_longitude']])
    train_data_frame.loc[:, 'dropoff_cluster'] = perform_k_means_clustering.predict(
        train_data_frame[['dropoff_latitude', 'dropoff_longitude']])

def refine_lat_lon(train, test):
    test.loc[:, 'center_latitude'] = (test['pickup_latitude'].values + test['dropoff_latitude'].values) / 2
    test.loc[:, 'center_longitude'] = (test['pickup_longitude'].values + test['dropoff_longitude'].values) / 2
    train.loc[:, 'center_latitude'] = (train['pickup_latitude'].values + train['dropoff_latitude'].values) / 2
    train.loc[:, 'center_longitude'] = (train['pickup_longitude'].values + train['dropoff_longitude'].values) / 2

def save_result_to_file(predictions):
    print("Saving results to csv file...")
    save_to_file = "submission_" + outset + ".csv"
    print("Submission being written to %s" % save_to_file)

    file = open(save_to_file, "w")
    file.write("id,trip_duration\n")
    for i in range(0, len(predictions)):
        estimation = predictions[i]
        file.write("%s,%f\n" % (((id_test_set[i]), estimation)))
    file.close()



temp = True
if temp:
    test = pandas.read_csv('../input/nyc-taxi-trip-duration/test.zip')
    train = pandas.read_csv('../input/nyc-taxi-trip-duration/train.zip')
    test_1 = test.copy()

    date_time_conversion(train, test)
    calculation_of_distance(train, test)
    calculate_difference_in_distance(train, test)
    calculation_of_bearing(train, test)
    refine_lat_lon(train, test)
    perform_prinicipal_component_analysis(train, test)
    perform_clustering(train, test)

    fastest_route_1 = pandas.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_1.csv',
                                      usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps', ])
    fastest_route_2 = pandas.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_2.csv',
                                      usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])

    information_on_street = pandas.concat((fastest_route_1, fastest_route_2))
    information_on_street_test = pandas.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_test.csv',
                                                 usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])

    test = test.merge(information_on_street_test, how='left', on='id')
    train = train.merge(information_on_street, how='left', on='id')


    train['log_trip_duration'] = numpy.log(train['trip_duration'].values + 1)

    feature_names = list(train.columns)

    features_not_for_train = ["log_trip_duration", "id", "dropoff_datetime", "trip_duration", "date",
                               'pickup_datetime', "pickup_date"]
    feature_names = [f for f in train.columns if f not in features_not_for_train]
    print('We have %i features.' % len(feature_names))
    train[feature_names].count()

    train['store_and_fwd_flag'] = train['store_and_fwd_flag'].map(lambda x: 0 if x == 'N' else 1)

    test['store_and_fwd_flag'] = test['store_and_fwd_flag'].map(lambda x: 0 if x == 'N' else 1)

    result = numpy.array(train['trip_duration'].values)
    value = train[feature_names].values
    id_test_set = numpy.array(test['id'].values)
    value_test = test[feature_names].values

    joblib.dump((value, value_test, result, id_test_set), "pikcles.pkl")
else:
    value, value_test, result, id_test_set = joblib.load("pikcles.pkl")

print(" The shape of train is ", value.shape)
print(" The final shape of X_test is", value_test.shape)
print(" THe final shape of y is", result.shape)

seed_count = 1
file_path = ''
outset = "lightgbm"
number_of_approximators = 40

predictions = generate_bagged_set(value, result, seed_count, number_of_approximators, value_test, label_t=None)
predictions = numpy.array(predictions)

save_result_to_file(predictions)

print("Completed Execution")