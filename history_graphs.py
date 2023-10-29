import matplotlib.pyplot as plt
import pickle
import numpy as np
import seaborn as sns
import pandas as pd

def plot_confusion_matrix(confmat, title='Confusion matrix'):
	
	con_mat_norm = np.around(confmat.astype('float') / confmat.sum(axis=1)[:, np.newaxis], decimals=2)
	con_mat_norm_df = pd.DataFrame(con_mat_norm)
	con_mat_df = pd.DataFrame(confmat)

	figure = plt.figure(figsize=(16, 8))

	ax = figure.add_subplot(122)
	sns.heatmap(con_mat_norm_df, annot=True,cmap=plt.cm.Blues)
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	names = ['Foliage', 'Stem', 'Ground', 'Undergrowth']
	ax.set_xticklabels(names)
	ax.set_yticklabels(names)
	plt.title('Normalised ' + title)

	ax = figure.add_subplot(121)

	sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	names = ['Foliage', 'Stem', 'Ground', 'Undergrowth']
	ax.set_xticklabels(names)
	ax.set_yticklabels(names)
	plt.title(title)

	plt.show()

def plot_result(history, item, window_width: None):
	try: history = history.history
	except: pass

	print(window_width)

	if window_width is not None:
		title = "Train and Validation {} Over Epochs with Window Width {}".format(item, window_width)
	else:
		title = "Train and Validation {} Over Epochs".format(item)

	plt.plot(history[item], label=item)
	plt.plot(history["val_" + item], label="val_" + item)
	plt.xlabel("Epochs")
	plt.ylabel(item)
	plt.title(title, fontsize=14)
	plt.legend()
	plt.grid()
	plt.ylim(0, 1)
	plt.show()

def gen_training_graphs(val = False):
	nums = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
	numLabels = ["0.5m", "1.0m", "1.5m", "2.0m", "2.5m", "3.0m", "3.5m", "4.0m", "4.5m"]
	dirs = ["trees_{}".format(num) for num in nums]

	# List all attributes for each history
	if val == False:
		attributes = ["loss", "sparse_cat_acc", "meanIoU", "FoliageIoU", "StemIoU", "GroundIoU", "UndergrowthIoU"]
		attributeLabels = ["Training Loss", "Training Accuracy", "Training mean IoU", "Training Foliage IoU", "Training Stem IoU", "Training Ground IoU", "Training Undergrowth IoU"]
		IoULabels= ["Training Foliage IoU", "Training Stem IoU", "Training Ground IoU", "Training Undergrowth IoU"]
		IoUattributes = ["FoliageIoU", "StemIoU", "GroundIoU", "UndergrowthIoU"]
		title = "Training "
	elif val == True:
		attributes = ["val_loss", "val_sparse_cat_acc", "val_meanIoU", "val_FoliageIoU", "val_StemIoU", "val_GroundIoU", "val_UndergrowthIoU"]
		IoUattributes = ["val_FoliageIoU", "val_StemIoU", "val_GroundIoU", "val_UndergrowthIoU"]
		title = "Validation "

	title = "Training and Validation "

	val_attributes = ["val_loss", "val_sparse_cat_acc", "val_meanIoU", "val_FoliageIoU", "val_StemIoU", "val_GroundIoU", "val_UndergrowthIoU"]
	val_IoUattributes = ["val_FoliageIoU", "val_StemIoU", "val_GroundIoU", "val_UndergrowthIoU"]
	val_attributeLabels = ["Validation Loss", "Validation Accuracy", "Validation mean IoU", "Validation Foliage IoU", "Validation Stem IoU", "Validation Ground IoU", "Validation Undergrowth IoU"]
	val_IoUattributeLabels = ["Validation Foliage IoU", "Validation Stem IoU", "Validation Ground IoU", "Validation Undergrowth IoU"]

	# Create a dictionary to store the final values for each attribute
	final_values = {attribute: [] for attribute in attributes}
	final_val_values = {attribute: [] for attribute in val_attributes}

	# List final values of each attribute across all logs
	for attribute in attributes:
		# print("{}: ".format(attribute))
		for dir in dirs:
			with open('./logs/{}/trainHistoryDict'.format(dir), 'rb') as f:
				history = pickle.load(f)
				final_value = history[attribute][-1]
				final_val_value = history[val_attributes[attributes.index(attribute)]][-1]
				final_values[attribute].append(final_value)
				final_val_values[val_attributes[attributes.index(attribute)]].append(final_val_value)
				# print("{}: {}".format(dir, final_value))
		# print("\n")

	# Plot the final values for IoU
	colours = ['lime', 'red', 'blue', 'cyan']
	for attribute in IoUattributes:
		colour = colours[IoUattributes.index(attribute)]
		plt.plot(nums, final_values[attribute], label=IoULabels[IoUattributes.index(attribute)], color=colour)
		plt.plot(nums, final_val_values[val_IoUattributes[IoUattributes.index(attribute)]], label=val_IoUattributeLabels[IoUattributes.index(attribute)], linestyle='dashed', color=colour)
	plt.xlabel("Window Width")
	plt.ylabel("IoU")
	plt.title("{}IoU for Varying Window Widths".format(title), fontsize=14)
	# Legend in the bottom right
	plt.legend(loc='lower right')
	plt.grid()
	plt.ylim(0, 1)
	plt.show()

	# Plot the final values for mean IoU only
	plt.plot(nums, final_values[attributes[2]], label=attributeLabels[2], color = 'blueviolet')
	plt.plot(nums, final_val_values[val_attributes[2]], label=val_attributeLabels[2], linestyle='dashed', color = 'blueviolet')
	plt.xlabel("Window Width")
	plt.ylabel("IoU")
	plt.title("{}Mean IoU for Varying Window Widths".format(title), fontsize=14)
	plt.legend()
	plt.grid()
	plt.ylim(0, 1)
	plt.show()

	# Plot the final values for sparse categorical accuracy
	plt.plot(nums, final_values[attributes[1]], label=attributeLabels[1], color='steelblue')
	plt.plot(nums, final_val_values[val_attributes[1]], label=val_attributeLabels[1], linestyle='dashed', color='steelblue')
	plt.xlabel("Window Width")
	plt.ylabel("Accuracy")
	plt.title("{}Accuracy for Varying Window Widths".format(title), fontsize=14)
	plt.legend()
	plt.grid()
	plt.ylim(0, 1)
	plt.show()

	# Plot the final values for loss
	plt.plot(nums, final_values[attributes[0]], label=attributeLabels[0], color='olivedrab')
	plt.plot(nums, final_val_values[val_attributes[0]], label=val_attributeLabels[0], linestyle='dashed', color='olivedrab')
	plt.xlabel("Window Width")
	plt.ylabel("Cross-Entropy")
	plt.title("{}Loss for Varying Window Widths".format(title), fontsize=14)
	plt.legend()
	plt.grid()
	plt.ylim(0, 1)
	plt.show()

	# Plot all the meanIoUs together on one graph across all epochs without validation data
	for dir in dirs:
		with open('./logs/{}/trainHistoryDict'.format(dir), 'rb') as f:
			history = pickle.load(f)
			plt.plot(history[attributes[2]], label=numLabels[dirs.index(dir)])

	plt.xlabel("Epochs")
	plt.ylabel("IoU")
	plt.title("Training Mean IoU Across Epochs".format(title), fontsize=14)
	plt.legend()
	plt.grid()
	plt.ylim(0.5, 1)
	plt.show()

	for dir in dirs:
		with open('./logs/{}/trainHistoryDict'.format(dir), 'rb') as f:
			history = pickle.load(f)
			plt.plot(history[attributes[0]], label=numLabels[dirs.index(dir)])

	plt.xlabel("Epochs")
	plt.ylabel("Cross-Entropy")
	plt.title("Training Loss Across Epochs".format(title), fontsize=14)
	plt.legend()
	plt.grid()
	plt.ylim(0, 1)
	plt.show()

	for dir in dirs:
		with open('./logs/{}/trainHistoryDict'.format(dir), 'rb') as f:
			history = pickle.load(f)
			plt.plot(history[attributes[1]], label=numLabels[dirs.index(dir)])

	plt.xlabel("Epochs")
	plt.ylabel("Accuracy")
	plt.title("Training Accuracy Across Epochs".format(title), fontsize=14)
	plt.legend()
	plt.grid()
	plt.ylim(0, 1)
	plt.show()

def gen_testing_graphs_and_confmats(test_data_nums):
	# Define the order the results appear in
	metric_list = ['loss', 'sparse_cat_acc', 'meanIoU', 'FoliageIoU', 'StemIoU', 'GroundIoU', 'UndergrowthIoU']

	# Define the directory containing the testHistoryDict files
	test_dirs = ['./logs/test_history/testHistory_{}'.format(num) for num in test_data_nums]

	# Create a list of window_widths from test_data_nums
	window_widths = [float(num.replace('_', '.')) for num in test_data_nums]

	# Create empty lists
	accuracies = []
	meanIoUs = []
	losses = []
	FoliageIous = []
	StemIoUs = []
	GroundIoUs = []
	UndergrowthIoUs = []

	# Load the accuracies for each window width from each testHistoryDict file
	for i in range(len(test_dirs)):
		with open(test_dirs[i], 'rb') as f:
			history = pickle.load(f)

			# Append the accuracy and meanIoU to the lists based on their position in metric_list
			accuracies.append(history[metric_list.index('sparse_cat_acc')])
			meanIoUs.append(history[metric_list.index('meanIoU')])
			losses.append(history[metric_list.index('loss')])
			FoliageIous.append(history[metric_list.index('FoliageIoU')])
			StemIoUs.append(history[metric_list.index('StemIoU')])
			GroundIoUs.append(history[metric_list.index('GroundIoU')])
			UndergrowthIoUs.append(history[metric_list.index('UndergrowthIoU')])


	# Plot the accuracy
	plt.plot(window_widths, accuracies, label='Accuracy', color = 'steelblue')
	plt.xlabel('Window Width')
	plt.ylabel('Accuracy')
	plt.title('Accuracy across Window Widths on Testing Data', fontsize=14)
	plt.grid()
	plt.ylim(0, 1)
	plt.show()

	# Plot the IoUs
	plt.plot(window_widths, meanIoUs, label='mean IoU', color='blueviolet')
	plt.plot(window_widths, FoliageIous, label='Foliage IoU', color='lime')
	plt.plot(window_widths, StemIoUs, label='Stem IoU', color='red')
	plt.plot(window_widths, GroundIoUs, label='Ground IoU', color='blue')
	plt.plot(window_widths, UndergrowthIoUs, label='Undergrowth IoU', color='cyan')
	plt.xlabel('Window Width')
	plt.ylabel('IoU')
	plt.title('IoU across Window Widths on Testing Data', fontsize=14)
	plt.legend(loc = 'lower right')
	plt.grid()
	plt.ylim(0.0, 1)
	plt.show()

	# Plot the loss
	plt.plot(window_widths, losses, label='Loss', color='olivedrab')
	plt.xlabel('Window Width')
	plt.ylabel('Loss')
	plt.title('Loss across Window Widths on Testing Data', fontsize=14)
	plt.grid()
	plt.ylim(0, 1)
	plt.show()

	# Make an empty dictionary to store the stats with the data nums as the keys
	stats = {}

	# Plot confusion matrices
	for i in range(len(test_data_nums)):
		with open('./logs/test_history/confmat_{}'.format(test_data_nums[i]), 'rb') as f:
			confmat = pickle.load(f)
			con_mat_norm = np.around(confmat.astype('float') / confmat.sum(axis=1)[:, np.newaxis], decimals=2)
			plot_confusion_matrix(confmat, title='Confusion Matrix for {}m Window Width'.format(window_widths[i]))
			# Calculate per class TP, FP, FN, TN
			foliage_values = calculate_stats(confmat, con_mat_norm, 0)
			stem_values = calculate_stats(confmat, con_mat_norm, 1)
			ground_values = calculate_stats(confmat, con_mat_norm, 2)
			undergrowth_values = calculate_stats(confmat, con_mat_norm, 3)
			# Construct dictionary for these values
			values = {'foliage': foliage_values, 'stem': stem_values, 'ground': ground_values, 'undergrowth': undergrowth_values}
			# Add to stats dictionary
			stats[test_data_nums[i]] = values
			# Print the stats
			# print("Stats for {}m Window Width:".format(window_widths[i]))
			# print("Foliage: {}".format(foliage_values))
			# print("Stem: {}".format(stem_values))
			# print("Ground: {}".format(ground_values))
			# print("Undergrowth: {}".format(undergrowth_values))

	# Save stats dictionary
	# with open('./logs/test_history/stats', 'wb') as f:
	#     pickle.dump(stats, f)

def calculate_stats(confmat, confmat_norm, class_num):
	TP = confmat[class_num, class_num]
	FP = np.sum(confmat[:, class_num]) - TP
	FN = np.sum(confmat[class_num, :]) - TP
	TN = np.sum(confmat) - TP - FP - FN

	TP_N = confmat_norm[class_num, class_num]
	FP_N = np.sum(confmat_norm[:, class_num]) - TP_N
	FN_N = np.sum(confmat_norm[class_num, :]) - TP_N
	TN_N = np.sum(confmat_norm) - TP_N - FP_N - FN_N

	# Construct a dictionary
	values = {'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN, 'TP_N': TP_N, 'FP_N': FP_N, 'FN_N': FN_N, 'TN_N': TN_N}
	return values

def gen_confmat_derived_graphs():
	# Using the saved confusion matrix stats for both normalised and not, calculate and plot per class IoU and overall
	# as well as accuracy

	# Load the confusion matrix stats for each window width
	test_data_nums = ['0_5', '1_0', '1_5', '2_0', '2_5', '3_0', '3_5', '4_0', '4_5']
	window_widths = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5 ,4.0, 4.5]
	with open("./logs/test_history/stats", "rb") as f:
		stats = pickle.load(f)

	classes = ['foliage', 'stem', 'ground', 'undergrowth']

	foliage_ious = []
	stem_ious = []
	ground_ious = []
	undergrowth_ious = []
	mean_ious = []
	accuracies = []

	normalised_foliage_ious = []
	normalised_stem_ious = []
	normalised_ground_ious = []
	normalised_undergrowth_ious = []
	normalised_mean_ious = []

	for num in test_data_nums:
		# Calculate IoU for each class
		for class_name in classes:
			#IoU
			stats[num][class_name]['IoU'] = stats[num][class_name]['TP'] / (stats[num][class_name]['TP'] + stats[num][class_name]['FN'] + stats[num][class_name]['FP'])
			#Normalised IoU
			stats[num][class_name]['NormalisedIoU'] = stats[num][class_name]['TP_N'] / (stats[num][class_name]['TP_N'] + stats[num][class_name]['FN_N'] + stats[num][class_name]['FP_N'])
		IoU = [stats[num]['foliage']['IoU'], stats[num]['stem']['IoU'], stats[num]['ground']['IoU'], stats[num]['undergrowth']['IoU']]
		normalised_IoU = [stats[num]['foliage']['NormalisedIoU'], stats[num]['stem']['NormalisedIoU'], stats[num]['ground']['NormalisedIoU'], stats[num]['undergrowth']['NormalisedIoU']]
		# print("IoU for window width {}: {}".format(num, IoU))

		# Calculate accuracy
		stats[num]['Accuracy'] = (stats[num]['foliage']['TP'] + stats[num]['stem']['TP'] + stats[num]['ground']['TP'] + stats[num]['undergrowth']['TP']) / (stats[num]['foliage']['TP'] + stats[num]['stem']['TP'] + stats[num]['ground']['TP'] + stats[num]['undergrowth']['TP'] + stats[num]['foliage']['FN'] + stats[num]['stem']['FN'] + stats[num]['ground']['FN'] + stats[num]['undergrowth']['FN'])
		# Normalised accuracy
		stats[num]['NormalisedAccuracy'] = (stats[num]['foliage']['TP_N'] + stats[num]['stem']['TP_N'] + stats[num]['ground']['TP_N'] + stats[num]['undergrowth']['TP_N']) / (stats[num]['foliage']['TP_N'] + stats[num]['stem']['TP_N'] + stats[num]['ground']['TP_N'] + stats[num]['undergrowth']['TP_N'] + stats[num]['foliage']['FN_N'] + stats[num]['stem']['FN_N'] + stats[num]['ground']['FN_N'] + stats[num]['undergrowth']['FN_N'])	
		# print("Accuracy for window width {}: {}".format(num, stats[num]['Accuracy']))

		# Calculate mean IoU
		stats[num]["MeanIoU"] = sum(IoU) / len(IoU)
		stats[num]["NormalisedMeanIoU"] = sum(normalised_IoU) / len(normalised_IoU)
		
		# print("Mean IoU for window width {}: {}".format(num, stats[num]["MeanIoU"]))

		# Store IoUs and accuracy

		foliage_ious.append(stats[num]['foliage']['IoU'])
		stem_ious.append(stats[num]['stem']['IoU'])
		ground_ious.append(stats[num]['ground']['IoU'])
		undergrowth_ious.append(stats[num]['undergrowth']['IoU'])
		mean_ious.append(stats[num]["MeanIoU"])
		accuracies.append(stats[num]['Accuracy'])

		normalised_foliage_ious.append(stats[num]['foliage']['NormalisedIoU'])
		normalised_stem_ious.append(stats[num]['stem']['NormalisedIoU'])
		normalised_ground_ious.append(stats[num]['ground']['NormalisedIoU'])
		normalised_undergrowth_ious.append(stats[num]['undergrowth']['NormalisedIoU'])
		normalised_mean_ious.append(stats[num]["NormalisedMeanIoU"])

		# # Plot IoU for each class
		# plt.bar(['Foliage', 'Stem', 'Ground', 'Undergrowth'], IoU)
		# plt.xlabel("Class")
		# plt.ylabel("IoU")
		# plt.title("IoU for window width {}".format(num))
		# plt.grid()
		# plt.ylim(0, 1)
		# plt.show()

	# print("Done calculating IoUs and accuracies")
	# Plot IoUs all together
	plt.plot(window_widths, mean_ious, label='Mean', color='b')
	plt.plot(window_widths, foliage_ious, label='Foliage', color='g')
	plt.plot(window_widths, stem_ious, label='Stem', color='r')
	plt.plot(window_widths, ground_ious, label='Ground', color='c')
	plt.plot(window_widths, undergrowth_ious, label='Undergrowth', color='m')
	plt.xlabel("Window Width")
	plt.ylabel("IoU")
	plt.title("IoU Across Window Widths")
	plt.legend()
	plt.grid()
	plt.ylim(0, 1)
	plt.show()

	plt.plot(window_widths, normalised_mean_ious, label='Mean', color='blueviolet')
	plt.plot(window_widths, normalised_foliage_ious, label='Foliage', color='lime')
	plt.plot(window_widths, normalised_stem_ious, label='Stem', color='red')
	plt.plot(window_widths, normalised_ground_ious, label='Ground', color='blue')
	plt.plot(window_widths, normalised_undergrowth_ious, label='Undergrowth', color='cyan')
	plt.xlabel("Window Width")
	plt.ylabel("Normalised IoU")
	plt.title("Normalised IoU Across Window Widths")
	plt.legend()
	plt.grid()
	plt.ylim(0, 1)
	plt.show()


if __name__ == '__main__':
	# List all log directories to use
	# gen_training_graphs(val = False)
	test_data_nums = ['0_5', '1_0', '1_5', '2_0', '2_5', '3_0', '3_5', '4_0', '4_5']

	gen_testing_graphs_and_confmats(test_data_nums)
	# gen_confmat_derived_graphs()
