# Created by Tomas Kliment, with virtual interpreter based on Python 3.12
# The goal of this program is customer segmentation based on available data


from controller.MainController import Controller


if __name__ == "__main__":

    files = ["customer-spending-1.csv", "customer-spending-2.csv", "customer-spending-3.csv"]
    ctr: Controller = Controller(files)
    ctr.do_it()
