from basic_treedata import train, setup
from sliding_window import create_records_using_window

if __name__ == '__main__':

    window_width = float(input("Enter window width: "))

    create_records_using_window(window_width)
    
    config = setup(window_width)

    train(config)