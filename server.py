import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from PIL import Image, ImageOps
from scipy.spatial import cKDTree
from skimage.measure import ransac
from skimage.transform import AffineTransform
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import tensorflow as tf
from flask import Flask, render_template, request
import glob
import tensorflow_hub as hub
import re
import codecs
import csv
from sklearn.utils import check_random_state
from sklearn.linear_model._ransac import _dynamic_max_trials


app = Flask(__name__)


def num_sort(test_string): 
    return list(map(int, re.findall(r'\d+', test_string)))[0]


def get_resized_db_image_paths(destfolder='./static/images/resized'):
    db = (list(glob.iglob(os.path.join(destfolder, '*.[Jj][Pp][Gg]'))))
    db.sort(key=num_sort)
    return db


# Preloaded data
locations_agg = np.load('./static/images/feature/locations.npy')
descriptors_agg = np.load('./static/images/feature/descriptors.npy')
accumulated_indexes_boundaries = np.load('./static/images/feature/accumulated_indexes_boundaries.npy')
db_images = get_resized_db_image_paths()
base = get_resized_db_image_paths('./static/images/database')
d_tree = cKDTree(descriptors_agg)
building_descs = []
f = codecs.open('./static/images/buildings.csv', 'rU', encoding='utf-8-sig')
reader = csv.reader(f)
for utf8_row in reader:
    building_descs.append(utf8_row[0])


def resize(image, new_width=256, new_height=256):
  image = ImageOps.fit(image, (new_width, new_height), Image.ANTIALIAS)
  return image

def run_delf(image):
  np_image = np.array(image)
  float_image = tf.image.convert_image_dtype(np_image, tf.float32)

  return delf(
      image=float_image,
      score_threshold=tf.constant(100.0),
      image_scales=tf.constant([0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0]),
      max_feature_num=tf.constant(1000))


def ransac(data, model_class, min_samples, residual_threshold,
           is_data_valid=None, is_model_valid=None,
           max_trials=100, stop_sample_num=np.inf, stop_residuals_sum=0,
           stop_probability=1, random_state=None, initial_inliers=None):
    best_model = None
    best_inlier_num = 0
    best_inlier_residuals_sum = np.inf
    best_inliers = None

    random_state = check_random_state(random_state)

    # in case data is not pair of input and output, male it like it
    if not isinstance(data, (tuple, list)):
        data = (data, )
    num_samples = len(data[0])

    if not (0 < min_samples < num_samples):
        return None, None

    if residual_threshold < 0:
        raise ValueError("`residual_threshold` must be greater than zero")

    if max_trials < 0:
        raise ValueError("`max_trials` must be greater than zero")

    if not (0 <= stop_probability <= 1):
        raise ValueError("`stop_probability` must be in range [0, 1]")

    if initial_inliers is not None and len(initial_inliers) != num_samples:
        raise ValueError("RANSAC received a vector of initial inliers (length %i)"
                         " that didn't match the number of samples (%i)."
                         " The vector of initial inliers should have the same length"
                         " as the number of samples and contain only True (this sample"
                         " is an initial inlier) and False (this one isn't) values."
                          % (len(initial_inliers), num_samples))

    # for the first run use initial guess of inliers
    spl_idxs = (initial_inliers if initial_inliers is not None else random_state.choice(num_samples, min_samples, replace=False))

    for num_trials in range(max_trials):
        # do sample selection according data pairs
        samples = [d[spl_idxs] for d in data]
        # for next iteration choose random sample set and be sure that no samples repeat
        spl_idxs = random_state.choice(num_samples, min_samples, replace=False)

        # optional check if random sample set is valid
        if is_data_valid is not None and not is_data_valid(*samples):
            continue

        # estimate model for current random sample set
        sample_model = model_class()

        success = sample_model.estimate(*samples)
        # backwards compatibility
        if success is not None and not success:
            continue

        # optional check if estimated model is valid
        if is_model_valid is not None and not is_model_valid(sample_model, *samples):
            continue

        sample_model_residuals = np.abs(sample_model.residuals(*data))
        # consensus set / inliers
        sample_model_inliers = sample_model_residuals < residual_threshold
        sample_model_residuals_sum = np.sum(sample_model_residuals ** 2)

        # choose as new best model if number of inliers is maximal
        sample_inlier_num = np.sum(sample_model_inliers)
        if (
            # more inliers
            sample_inlier_num > best_inlier_num
            # same number of inliers but less "error" in terms of residuals
            or (sample_inlier_num == best_inlier_num
                and sample_model_residuals_sum < best_inlier_residuals_sum)
        ):
            best_model = sample_model
            best_inlier_num = sample_inlier_num
            best_inlier_residuals_sum = sample_model_residuals_sum
            best_inliers = sample_model_inliers
            dynamic_max_trials = _dynamic_max_trials(best_inlier_num,
                                                     num_samples,
                                                     min_samples,
                                                     stop_probability)
            if (best_inlier_num >= stop_sample_num
                or best_inlier_residuals_sum <= stop_residuals_sum
                or num_trials >= dynamic_max_trials):
                break

    # estimate final model using all inliers
    if best_inliers is not None and any(best_inliers):
        # select inliers for each data array
        data_inliers = [d[best_inliers] for d in data]
        best_model.estimate(*data_inliers)
    else:
        best_model = None
        best_inliers = None

    return best_model, best_inliers


def get_locations_2_use(image_db_index, k_nearest_indices, accumulated_indexes_boundaries, query_image_locations, locations_agg):
    image_accumulated_indexes = image_index_2_accumulated_indexes(image_db_index, accumulated_indexes_boundaries)
    locations_2_use_query = []
    locations_2_use_db = []
    for i, row in enumerate(k_nearest_indices):
        for acc_index in row:
            if acc_index in image_accumulated_indexes:
                locations_2_use_query.append(query_image_locations[i])
                locations_2_use_db.append(locations_agg[acc_index])
                break
    return np.array(locations_2_use_query), np.array(locations_2_use_db)


def image_index_2_accumulated_indexes(index, accumulated_indexes_boundaries):
    '''
    Image index to accumulated/aggregated locations/descriptors pair indexes.
    '''
    if index > len(accumulated_indexes_boundaries) - 1:
        return None
    accumulated_index_start = None
    accumulated_index_end = None
    if index == 0:
        accumulated_index_start = 0
        accumulated_index_end = accumulated_indexes_boundaries[index]
    else:
        accumulated_index_start = accumulated_indexes_boundaries[index-1]
        accumulated_index_end = accumulated_indexes_boundaries[index]
    return np.arange(accumulated_index_start,accumulated_index_end)


delf = hub.load('model').signatures['default']


def AP_():
    hit = 0.0
    score = 0.0
    loop = 0.0
    true = 0

    f = open('AP.txt', 'r+')
    f = f.readlines()
    for i in range(len(f)):
        if f[i] == f[1]:
            hit += 1.0
            loop += 1.0
            score += hit/loop
            true+=1
        else: loop+=1
    return (round((score/true),2)), true


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["query_img"]
        print(file)
        image = Image.open(file.stream)
        upload_img_path = "static/upload/" + file.filename
        
        # db_images = get_resized_database_image_paths()
        image.save(upload_img_path)

        resized_image = resize(image)
        
        result = run_delf(resized_image)
        
        query_image_locations, query_image_descriptors = result['locations'], result['descriptors']
        print(len(query_image_descriptors[0]))
        distance_threshold = 0.8
        # K nearest neighbors
        K = 10
        distances, indices = d_tree.query(
            query_image_descriptors, distance_upper_bound=distance_threshold, k = K, n_jobs=-1)


        unique_indices = np.array(list(set(indices.flatten())))

        unique_indices.sort()
        if unique_indices[-1] == descriptors_agg.shape[0]:
            unique_indices = unique_indices[:-1]

        unique_image_indexes = np.array(
            list(set([np.argmax([np.array(accumulated_indexes_boundaries)>index]) 
                    for index in unique_indices])))

        inliers_counts = []

        for index in unique_image_indexes:
            locations_2_use_query, locations_2_use_db = get_locations_2_use(index, indices, accumulated_indexes_boundaries, query_image_locations, locations_agg)
            # Perform geometric verification using RANSAC.
            _, inliers = ransac(
                (locations_2_use_db, locations_2_use_query), # source and destination coordinates
                AffineTransform,
                min_samples=5,
                residual_threshold=20,
                max_trials=50)
            # If no inlier is found for a database candidate image, we continue on to the next one.
            if inliers is None or len(inliers) == 0:
                continue
            # the number of inliers as the score for retrieved images.
            inliers_counts.append({"index": index, "inliers": sum(inliers)})
        

        top_match = sorted(inliers_counts, key=lambda k: k['inliers'], reverse=True)
        score = []
        for i in range(len(top_match)):
            index = top_match[i]
            index = index['index']
            score.append((building_descs[index], db_images[index], base[index]))


        with open('AP.txt', 'w') as f:
            for item, _, _ in score:
                f.write("%s\n" % item)

        ap = AP_()

        map = open("MAP.txt","a")
        map.write(str(ap[0]) + "\n")
        map.close()

        Map = 0.0
        ind = 1
        with open('MAP.txt', 'r') as m:
            for line in m:
                try:
                    num = float(line)
                    Map += num
                except ValueError:
                    print('{} is not a number!'.format(line))
                ind += 1
        Map = round(Map / (ind - 1), 2)

        return render_template("search.html", query_path=upload_img_path, scores=score, ap=ap, map=Map)
    else:
        return render_template("search2.html")


if __name__ == "__main__":
    app.run()

