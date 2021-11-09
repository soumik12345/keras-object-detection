from keras_object_detection import benchmarks


shapes_benchmark = benchmarks.ShapesBenchMark()
n_data_samples = 128
# shapes_benchmark.make_dataset(n_data_samples=n_data_samples)
shapes_benchmark.set_label_map()
tfrecord_dir = shapes_benchmark.create_tfrecords(
    val_split=0.2, samples_per_shard=64
)
print(tfrecord_dir)