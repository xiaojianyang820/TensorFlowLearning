import csv


if __name__ == '__main__':
    agnews_train = csv.reader(open('data/ag/train.csv'))
    for line in agnews_train:
        print(line)
