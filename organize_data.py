from random import shuffle
import os
import shutil
import random
import copy 



class Organization:
    def __init__(self, main_root="data", data_folder="all_data", percent_valid = 0.0, percent_test=0.3, valid_folder="valid_data", test_folder= "testing_data", train_folder="training_data"):
        self.main_root = main_root
        self.all_data_root = os.path.join(self.main_root, data_folder) 
        self.valid_root = os.path.join(self.main_root, valid_folder) 
        self.test_root = os.path.join(self.main_root, test_folder) 
        self.train_root = os.path.join(self.main_root, train_folder) 
        self.percent_valid = percent_valid
        self.percent_test = percent_test
        self.percent_train = 1.0 - (percent_valid + percent_test)
        self.folder_main_list = [os.path.join(self.all_data_root,x) for x in os.listdir(self.all_data_root)]
        self.folder_name_list = os.listdir(self.all_data_root)
        self.folder_valid_list = [os.path.join(self.valid_root,x) for x in os.listdir(self.all_data_root)]
        self.folder_test_list = [os.path.join(self.test_root,x) for x in os.listdir(self.all_data_root)]
        self.folder_train_list = [os.path.join(self.train_root,x) for x in os.listdir(self.all_data_root)]


    def create_roots(self):
        if not os.path.exists(self.valid_root):
            os.makedirs(self.valid_root)
        if not os.path.exists(self.test_root):
            os.makedirs(self.test_root)
        if not os.path.exists(self.train_root):
            os.makedirs(self.train_root)
        for x in self.folder_valid_list : 
            if not os.path.exists(x):
                os.makedirs(x)   
        for x in self.folder_test_list : 
            if not os.path.exists(x):
                os.makedirs(x)
        for x in self.folder_train_list : 
            if not os.path.exists(x):
                os.makedirs(x)

    def dispatch_from_all_data_to_train_test_valid(self):
        for folder_name, main_folder, valid_folder, test_folder, train_folder in zip(self.folder_name_list, self.folder_main_list, self.folder_valid_list, self.folder_test_list, self.folder_train_list):
            file_main_list = [os.path.join(main_folder,x) for x in os.listdir(main_folder)]
            copy_main_file_list = copy.copy(file_main_list)
            file_name_main_list = [os.path.join(folder_name, x) for x in os.listdir(main_folder)]
            copy_file_name_main_list = copy.copy(file_name_main_list)
            valid_file_list_len = int(self.percent_valid * len(file_main_list))
            testing_file_list_len = int(self.percent_test * len(file_main_list))
            if testing_file_list_len == 0 :
                testing_file_list_len = 1
            training_file_list_len = len(file_main_list) - (valid_file_list_len + testing_file_list_len)

            valid_list_file = []
            valid_list_file_name = []
            for i in range(valid_file_list_len):
                index_rand = int(random.random()*len(copy_main_file_list))
                valid_list_file.append(copy_main_file_list.pop(index_rand))
                valid_list_file_name.append(copy_file_name_main_list(index_rand))


            test_list_file = []
            test_list_file_name = []
            for i in range(testing_file_list_len):
                index_rand = int(random.random()*len(copy_main_file_list))
                test_list_file.append(copy_main_file_list.pop(index_rand))
                test_list_file_name.append(copy_file_name_main_list.pop((index_rand)))


            
            train_list_file = copy_main_file_list
            train_list_file_name = copy_file_name_main_list

            for test_old_file, train_old_file, in zip( test_list_file, train_list_file):
                shutil.copy(test_old_file, test_folder)
                shutil.copy(train_old_file, train_folder)
