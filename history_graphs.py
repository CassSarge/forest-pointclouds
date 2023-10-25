import matplotlib.pyplot as plt
import pickle

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

def gen_graphs(val = False):
	nums = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
	dirs = ["trees_{}".format(num) for num in nums]

	# List all attributes for each history
	if val == False:
		attributes = ["loss", "sparse_cat_acc", "meanIoU", "FoliageIoU", "StemIoU", "GroundIoU", "UndergrowthIoU"]
		IoUattributes = ["meanIoU", "FoliageIoU", "StemIoU", "GroundIoU", "UndergrowthIoU"]
		title = "Training "
	elif val == True:
		attributes = ["val_loss", "val_sparse_cat_acc", "val_meanIoU", "val_FoliageIoU", "val_StemIoU", "val_GroundIoU", "val_UndergrowthIoU"]
		IoUattributes = ["val_meanIoU", "val_FoliageIoU", "val_StemIoU", "val_GroundIoU", "val_UndergrowthIoU"]
		title = "Validation "

	# title = "Training and Validation "

	val_attributes = ["val_loss", "val_sparse_cat_acc", "val_meanIoU", "val_FoliageIoU", "val_StemIoU", "val_GroundIoU", "val_UndergrowthIoU"]
	val_IoUattributes = ["val_meanIoU", "val_FoliageIoU", "val_StemIoU", "val_GroundIoU", "val_UndergrowthIoU"]


	# Create a dictionary to store the final values for each attribute
	final_values = {attribute: [] for attribute in attributes}
	final_val_values = {attribute: [] for attribute in val_attributes}

	# List final values of each attribute across all logs
	for attribute in attributes:
		print("{}: ".format(attribute))
		for dir in dirs:
			with open('./logs/{}/trainHistoryDict'.format(dir), 'rb') as f:
				history = pickle.load(f)
				final_value = history[attribute][-1]
				final_val_value = history[val_attributes[attributes.index(attribute)]][-1]
				final_values[attribute].append(final_value)
				final_val_values[val_attributes[attributes.index(attribute)]].append(final_val_value)
				print("{}: {}".format(dir, final_value))
		print("\n")

	# Plot the final values for IoU
	colours = ['b', 'g', 'r', 'c', 'm']
	for attribute in IoUattributes:
		colour = colours[IoUattributes.index(attribute)]
		plt.plot(nums, final_values[attribute], label=attribute, color=colour)
		# plt.plot(nums, final_val_values[val_IoUattributes[IoUattributes.index(attribute)]], label=val_IoUattributes[IoUattributes.index(attribute)], linestyle='dashed', color=colour)
	plt.xlabel("Window Width")
	plt.ylabel("IoU")
	plt.title("Final {}IoU Values Across Experiments".format(title), fontsize=14)
	plt.legend()
	plt.grid()
	plt.ylim(0, 1)
	plt.show()

	# Plot the final values for sparse categorical accuracy
	plt.plot(nums, final_values[attributes[1]], label=attributes[1])
	# plt.plot(nums, final_val_values[val_attributes[1]], label=val_attributes[1], linestyle='dashed')
	plt.xlabel("Window Width")
	plt.ylabel("Accuracy")
	plt.title("Final {}Accuracy Values Across Experiments".format(title), fontsize=14)
	plt.legend()
	plt.grid()
	plt.ylim(0, 1)
	plt.show()

	# Plot the final values for loss
	plt.plot(nums, final_values[attributes[0]], label=attributes[0])
	# plt.plot(nums, final_val_values[val_attributes[0]], label=val_attributes[0], linestyle='dashed')
	plt.xlabel("Window Width")
	plt.ylabel("Cross-Entropy")
	plt.title("Final {}Loss Values Across Experiments".format(title), fontsize=14)
	plt.legend()
	plt.grid()
	plt.ylim(0, 1)
	plt.show()

	# # Generate plots for each log showing how each attribute changed over the 50 epochs, including window width
	# for dir in dirs:
	#     with open('./logs/{}/trainHistoryDict'.format(dir), 'rb') as f:
	#         history = pickle.load(f)
	#         # plot_result(history, "loss", window_width=dir[-3:])
	#         # plot_result(history, "sparse_cat_acc", window_width=dir[-3:])
	#         plot_result(history, "meanIoU", window_width=dir[-3:])
	#         # plot_result(history, "FoliageIoU", window_width=dir[-3:])
	#         # plot_result(history, "StemIoU", window_width=dir[-3:])
	#         # plot_result(history, "GroundIoU", window_width=dir[-3:])
	#         # plot_result(history, "UndergrowthIoU", window_width=dir[-3:])

	# Plot all the meanIoUs together on one graph across all epochs without validation data
	for dir in dirs:
		with open('./logs/{}/trainHistoryDict'.format(dir), 'rb') as f:
			history = pickle.load(f)
			plt.plot(history[attributes[2]], label=dir)

	plt.xlabel("Epochs")
	plt.ylabel("IoU")
	plt.title("{}Mean IoU Across Epochs".format(title), fontsize=14)
	plt.legend()
	plt.grid()
	plt.ylim(0, 1)
	plt.show()

	for dir in dirs:
		with open('./logs/{}/trainHistoryDict'.format(dir), 'rb') as f:
			history = pickle.load(f)
			plt.plot(history[attributes[0]], label=dir)

	plt.xlabel("Epochs")
	plt.ylabel("Cross-Entropy")
	plt.title("{}Loss Across Epochs".format(title), fontsize=14)
	plt.legend()
	plt.grid()
	plt.ylim(0, 1)
	plt.show()

	for dir in dirs:
		with open('./logs/{}/trainHistoryDict'.format(dir), 'rb') as f:
			history = pickle.load(f)
			plt.plot(history[attributes[1]], label=dir)

	plt.xlabel("Epochs")
	plt.ylabel("Accuracy")
	plt.title("{}Accuracy Across Epochs".format(title), fontsize=14)
	plt.legend()
	plt.grid()
	plt.ylim(0, 1)
	plt.show()

def graph_from_confmats():
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
		print("IoU for window width {}: {}".format(num, IoU))

		# Calculate accuracy
		stats[num]['Accuracy'] = (stats[num]['foliage']['TP'] + stats[num]['stem']['TP'] + stats[num]['ground']['TP'] + stats[num]['undergrowth']['TP']) / (stats[num]['foliage']['TP'] + stats[num]['stem']['TP'] + stats[num]['ground']['TP'] + stats[num]['undergrowth']['TP'] + stats[num]['foliage']['FN'] + stats[num]['stem']['FN'] + stats[num]['ground']['FN'] + stats[num]['undergrowth']['FN'])
		# Normalised accuracy
		stats[num]['NormalisedAccuracy'] = (stats[num]['foliage']['TP_N'] + stats[num]['stem']['TP_N'] + stats[num]['ground']['TP_N'] + stats[num]['undergrowth']['TP_N']) / (stats[num]['foliage']['TP_N'] + stats[num]['stem']['TP_N'] + stats[num]['ground']['TP_N'] + stats[num]['undergrowth']['TP_N'] + stats[num]['foliage']['FN_N'] + stats[num]['stem']['FN_N'] + stats[num]['ground']['FN_N'] + stats[num]['undergrowth']['FN_N'])	
		print("Accuracy for window width {}: {}".format(num, stats[num]['Accuracy']))


		# Calculate mean IoU
		stats[num]["MeanIoU"] = sum(IoU) / len(IoU)
		stats[num]["NormalisedMeanIoU"] = sum(normalised_IoU) / len(normalised_IoU)
		
		print("Mean IoU for window width {}: {}".format(num, stats[num]["MeanIoU"]))

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

	print("Done calculating IoUs and accuracies")
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

	plt.plot(window_widths, normalised_mean_ious, label='Mean', color='b')
	plt.plot(window_widths, normalised_foliage_ious, label='Foliage', color='g')
	plt.plot(window_widths, normalised_stem_ious, label='Stem', color='r')
	plt.plot(window_widths, normalised_ground_ious, label='Ground', color='c')
	plt.plot(window_widths, normalised_undergrowth_ious, label='Undergrowth', color='m')
	plt.xlabel("Window Width")
	plt.ylabel("Normalised IoU")
	plt.title("Normalised IoU Across Window Widths")
	plt.legend()
	plt.grid()
	plt.ylim(0, 1)
	plt.show()


if __name__ == '__main__':
	# List all log directories to use
#    gen_graphs(val = True)
	graph_from_confmats()