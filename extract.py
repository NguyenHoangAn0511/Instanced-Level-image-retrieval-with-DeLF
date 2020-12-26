from match import *

if __name__ == "__main__":
    resize_images_folder('./static/images/database')
    db_images = get_resized_db_image_paths()
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

    image_tf = image_input_fn(db_images)

    with tf.train.MonitoredSession() as sess:
        results_dict = {}  # Stores the locations and their descriptors for each image
        for image_path in db_images:
            print(image_path)
            image = sess.run(image_tf)
            results_dict[image_path] = sess.run(
                [module_outputs['locations'], module_outputs['descriptors']],
                feed_dict={image_placeholder: image})


    # Extract here before save
    locations_agg = np.concatenate([results_dict[img][0] for img in db_images])
    descriptors_agg = np.concatenate([results_dict[img][1] for img in db_images])
    accumulated_indexes_boundaries = list(accumulate([results_dict[img][0].shape[0] for img in db_images]))


    # Save then load
    np.save('./static/images/feature/locations.npy', locations_agg)
    np.save('./static/images/feature/descriptors.npy', descriptors_agg)
    np.save('./static/images/feature/accumulated_indexes_boundaries.npy', accumulated_indexes_boundaries)