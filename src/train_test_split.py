class print_tts(object):
    def __init__(self, X1_train, X1_val, X1_test, X1_rfe_train, X1_rfe_test, y1_train, y1_val, y1_test):
        self.X1_train = X1_train
        self.X1_val = X1_val
        self.X_test = X1_test
        self.X1_rfe_train = X1_rfe_train
        self.X1_rfe_test = X1_rfe_test
        self.y1_train = y1_train
        self.y1_val = y1_val
        self.y1_test = y1_test

    def tts(X1_train, X1_val, X1_test, y1_train, y1_val, y1_test):
        '''Print the shape of train test split of dataset'''
        print(f'Shape of the X1_train {X1_train.shape}')
        print(f'Shape of the X1_val {X1_val.shape}')
        print(f'Shape of the X1_test {X1_test.shape}')

        print(f'Shape of the y1_train {y1_train.shape}')
        print(f'Shape of the y1_val {y1_val.shape}')
        print(f'Shape of the y1_test {y1_test.shape}')

    def tts_rfe(X1_train, X1_test, X1_rfe_train, X1_rfe_test, y1_train, y1_test):
        '''Print the shape of train test split of dataset after Recursive Feature Elimination'''
        print("Train size: {}".format(len(X1_train)))
        print("Test size: {}".format(len(X1_test)))                                                                            
        print("Train size RFE: {}".format(len(X1_rfe_train)))
        print("Test size RFE: {}".format(len(X1_rfe_test)))
        print("Train size: {}".format(len(y1_train)))
        print("Test size: {}".format(len(y1_test)))