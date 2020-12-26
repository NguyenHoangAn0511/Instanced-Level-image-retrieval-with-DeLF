import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from scipy.spatial import cKDTree
from skimage.feature import plot_matches
from skimage.transform import AffineTransform
from six import BytesIO
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import silence_tensorflow.auto
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_hub as hub
from six.moves.urllib.request import urlopen
from sklearn.utils import check_random_state
import glob
from itertools import accumulate
from sklearn.linear_model._ransac import _dynamic_max_trials
np.random.seed(10)
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import re
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import numba
from numba import jit

def image_input_fn(image_files):
    filename_queue = tf.train.string_input_producer(
        image_files, shuffle=False)
    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)
    image_tf = tf.image.decode_jpeg(value, channels=3)
    return tf.image.convert_image_dtype(image_tf, tf.float32)

# import csv
# import codecs
# building_descs = []
# f = codecs.open('./images/buildings.csv', 'rU', encoding='utf-8-sig')
# reader = csv.reader(f)
# for utf8_row in reader:
#     building_descs.append(utf8_row[0])

"""## Resize all database images"""

def resize_image(srcfile, destfile='static/upload/query.jpg', new_width=256, new_height=256):
    # pil_image = Image.open(srcfile)
    pil_image = ImageOps.fit(srcfile, (new_width, new_height), Image.ANTIALIAS)
    pil_image_rgb = pil_image.convert('RGB')
    pil_image_rgb.save(destfile)
    return destfile



def resize_images_folder(srcfolder, destfolder='./static/images/resized', new_width=256, new_height=256):
    os.makedirs(destfolder,exist_ok=True)
    for srcfile in glob.iglob(os.path.join('./static/images/database', '*.[Jj][Pp][Gg]')):
        src_basename = os.path.basename(srcfile)
        destfile=os.path.join(destfolder,src_basename)
        srcfile = Image.open(srcfile)
        resize_image(srcfile, destfile, new_width, new_height)
    return destfolder


def num_sort(test_string): 
    return list(map(int, re.findall(r'\d+', test_string)))[0] 


def get_resized_db_image_paths(destfolder='./static/images/resized'):
    db = (list(glob.iglob(os.path.join(destfolder, '*.[Jj][Pp][Gg]'))))
    db.sort(key=num_sort)
    return db


# module_outputs = m(module_inputs, as_dict=True)

# image_tf = image_input_fn(db_images)

# with tf.train.MonitoredSession() as sess:
#   results_dict = {}  # Stores the locations and their descriptors for each image
#   for image_path in db_images:
#     image = sess.run(image_tf)
#     results_dict[image_path] = sess.run(
#         [module_outputs['locations'], module_outputs['descriptors']],
#         feed_dict={image_placeholder: image})



def compute_locations_and_descriptors(image_path):
    # tf.reset_default_graph()
    # tf.logging.set_verbosity(tf.logging.ERROR)

    m = hub.Module('model')
    image_placeholder = tf.placeholder(
        tf.float32, shape=(None, None, 3), name='input_image')
    module_inputs = {
        'image': image_placeholder,
        'score_threshold': 100.0,
        'image_scales': [0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0],
        'max_feature_num': 1000,
    }

    module_outputs = m(module_inputs, as_dict=True)
    image_tf = image_input_fn([image_path])

    with tf.train.MonitoredSession() as sess:
        image = sess.run(image_tf)
        return sess.run(
            [module_outputs['locations'], module_outputs['descriptors']],
            feed_dict={image_placeholder: image})


# @st.cache
# def preprocess_query_image(query_image):
#     '''
#     Resize the query image and return the resized image path.
#     '''
#     # query_temp_folder_name = 'query_temp_folder'
#     # query_temp_folder = os.path.join(os.path.dirname(query_image), query_temp_folder_name)
#     # os.makedirs(query_temp_folder,exist_ok=True)
#     query_basename = os.path.basename(query_image)
#     # destfile=os.path.join(query_temp_folder,query_basename)
#     resized_image = resize_image(query_image, 'images/query/query')
#     return resized_image


@jit(nopython=True)
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


# Commented out IPython magic to ensure Python compatibility.

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

