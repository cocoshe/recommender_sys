import pandas as pd
import os


class converter:
    """
    convert *.dat to *.csv
    """

    def __init__(self):
        self.path = 'dataset' + os.sep

    def convert(self):
        print('Converting user data...')
        self._convert_user_data()
        print('Converting movies data...')
        self._convert_movies_data()
        print('Converting rating data...')
        self._convert_rating_data()

    def _convert_user_data(self, file_path='users.dat'):
        f = pd.read_table(self.path + file_path, sep='::', engine='python',
                          names=['UserId', 'Gender', 'Age', 'Occupation', "Zip-code"])
        f.to_csv(self.path + 'users.csv', index=False)

    def _convert_rating_data(self, file_path='ratings.dat'):
        f = pd.read_table(self.path + file_path, sep='::', engine='python',
                          names=['UserId', 'MovieId', 'Rating', 'Timestamp'])
        f.to_csv(self.path + 'ratings.csv', index=False)

    def _convert_movies_data(self, file_path='movies.dat'):
        f = pd.read_table(self.path + file_path, sep='::', engine='python',
                          names=['MovieId', 'Title', 'Genres'], encoding='ISO-8859-1')
        f.to_csv(self.path + 'movies.csv', index=False)


if __name__ == '__main__':
    convert = converter()
    convert.convert()
