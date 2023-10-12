import matplotlib.pyplot as plt
import pickle

def plot_result(history, item):
    try: history = history.history
    except: pass

    plt.plot(history[item], label=item)
    plt.plot(history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    # List all log directories to use
    nums = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    dirs = ["trees_{}".format(num) for num in nums]

    # List all attributes for each history
    attributes = ["loss", "sparse_cat_acc", "meanIoU", "FoliageIoU", "StemIoU", "GroundIoU", "UndergrowthIoU"]

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

    # Plot the final values for each attribute
    for attribute in attributes:
        plt.plot(nums, final_values[attribute], label=attribute)
    plt.xlabel("Window Width")
    plt.ylabel("Final Value")
    plt.title("Final Values of Metrics Across Experiments", fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()