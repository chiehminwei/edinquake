import tensorflow as tf
from collections import OrderedDict
from os import listdir
from os.path import isfile
import os
mypath = "/Users/Jimmy/Downloads/test"
onlyfiles = [os.path.join(mypath, f) for f in listdir(mypath) if isfile(os.path.join(mypath, f))]


def create_float_feature(values):
	f = tf.train.Feature(float_list=tf.train.FloatList(value=values))
	return f

def create_string_feature(values):
	f = tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))
	return f

writer = tf.python_io.TFRecordWriter(os.path.join("/Users/Jimmy/", "test.tf_record"))
for file in sorted(onlyfiles):
	seg_id = file.strip().split('/')[-1]
	seg_id = seg_id.encode('utf-8')
	# print(seg_id)
	
	with open(file, 'r') as f:
		lines = []
		try:
			lines = f.readlines()[-128:]
		except:
			continue
		acoustic_signals = []
		for line in lines:
			try:
				line = float(line.strip())
			except:
				print(file)
				print(line)
			acoustic_signals.append(line)
		length_mask = [1.0] * 128

		features = OrderedDict()
		features["inputs"] = create_float_feature(acoustic_signals)
		features["length_mask"] = create_float_feature(length_mask)
		features["seg_id"] = create_string_feature(seg_id)
		
		tf_example = tf.train.Example(features=tf.train.Features(feature=features))
		writer.write(tf_example.SerializeToString())		




# class EdinquakeProcessor(DataProcessor):
# 	"""Processor for the edinquake data set."""
# 	def __init__(self, max_seq_length):
# 		self.max_seq_length = max_seq_length

# 	def get_train_examples(self, data_dir):
# 		input_file = os.path.join(data_dir, "quakes_128_real")
# 		output_file = os.path.join(data_dir, "train.tf_record")
# 		writer = tf.python_io.TFRecordWriter(output_file)

# 		def create_float_feature(values):
# 			f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
# 			return f

# 		with tf.gfile.Open(input_file, "r") as f:
# 			# examples = []
# 			for (i, line) in enumerate(f):
# 				if i % 10000 == 0:
# 					tf.logging.info("Writing example %d" % (i))

# 				acoustic_signals, labels = line.split('\t')
# 				acoustic_signals = [float(signal) for signal in acoustic_signals.split(',')]
# 				labels = [float(label) for label in labels.split(',')]
# 				length_mask = [1.0] * self.max_seq_length
# 				guid = "train-%d" % (i)

# 				example = InputExample(guid, acoustic_signals, labels, length_mask)
# 				features = OrderedDict()
# 				features["inputs"] = create_float_feature(example.acoustic_signals)
# 				features["labels"] = create_float_feature(example.labels)
# 				features["length_mask"] = create_float_feature(example.length_mask)

# 				tf_example = tf.train.Example(features=tf.train.Features(feature=features))
# 				writer.write(tf_example.SerializeToString())		
