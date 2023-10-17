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
    nums = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
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

    # Create a dictionary to store the final values for each attribute
    final_values = {attribute: [] for attribute in attributes}

    # List final values of each attribute across all logs
    for attribute in attributes:
        print("{}: ".format(attribute))
        for dir in dirs:
            with open('./logs/{}/trainHistoryDict'.format(dir), 'rb') as f:
                history = pickle.load(f)
                final_value = history[attribute][-1]
                final_values[attribute].append(final_value)
                print("{}: {}".format(dir, final_value))
        print("\n")

    # Plot the final values for IoU
    for attribute in IoUattributes:
        plt.plot(nums, final_values[attribute], label=attribute)
    plt.xlabel("Window Width")
    plt.ylabel("IoU")
    plt.title("Final {}IoU Values Across Experiments".format(title), fontsize=14)
    plt.legend()
    plt.grid()
    plt.ylim(0, 1)
    plt.show()

    # Plot the final values for sparse categorical accuracy
    plt.plot(nums, final_values[attributes[1]], label=attributes[1])
    plt.xlabel("Window Width")
    plt.ylabel("Accuracy")
    plt.title("Final {}Accuracy Values Across Experiments".format(title), fontsize=14)
    plt.legend()
    plt.grid()
    plt.ylim(0, 1)
    plt.show()

    # Plot the final values for loss
    plt.plot(nums, final_values[attributes[0]], label=attributes[0])
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

if __name__ == '__main__':
    # List all log directories to use
   gen_graphs(val = True)