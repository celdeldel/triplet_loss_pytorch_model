from random import shuffle
import os
import shutil
import random
import copy 

main_root = "2data/data"


valid_data_root =  "valid_data"
if not os.path.exists(valid_data_root):
    os.makedirs(valid_data_root)
print("valid_data_root")
print(valid_data_root)
test_data_root =  "testing_data"
if not os.path.exists(test_data_root):
    os.makedirs(test_data_root)
print("test_data_root")
print(test_data_root)
train_data_root = "training_data"
if not os.path.exists(train_data_root):
    os.makedirs(train_data_root)
print("train_data_root")
print(train_data_root)

percent_valid = 0.0
percent_test = 0.333333334
percent_train = 1.0 - (percent_valid + percent_test)




folder_main_list = [os.path.join(main_root,x) for x in os.listdir(main_root)]
folder_name_list = os.listdir(main_root)
print("folder main list")
print(folder_main_list)
print("folder folder_name_list")
print(folder_name_list)

folder_valid_list = [os.path.join(valid_data_root,x) for x in os.listdir(main_root)]
for x in folder_valid_list : 
    if not os.path.exists(x):
        os.makedirs(x)
print("folder_valid_list")
print(folder_valid_list)
folder_test_list = [os.path.join(test_data_root,x) for x in os.listdir(main_root)]
for x in folder_test_list : 
    if not os.path.exists(x):
        os.makedirs(x)
print("folder_test_list")
print(folder_test_list)
folder_train_list = [os.path.join(train_data_root,x) for x in os.listdir(main_root)]
for x in folder_train_list : 
    if not os.path.exists(x):
        os.makedirs(x)
print("folder_train_list")
print(folder_train_list)

for folder_name, main_folder, valid_folder, test_folder, train_folder in zip(folder_name_list, folder_main_list, folder_valid_list, folder_test_list, folder_train_list):
    file_main_list = [os.path.join(main_folder,x) for x in os.listdir(main_folder)]
    copy_main_file_list = copy.copy(file_main_list)
    print("copy_main_file_list")
    print(copy_main_file_list)

    file_name_main_list = [os.path.join(folder_name, x) for x in os.listdir(main_folder)]
    copy_file_name_main_list = copy.copy(file_name_main_list)
    print("copy_file_name_main_list")



    valid_file_list_len = int(percent_valid * len(file_main_list))
    testing_file_list_len = int(percent_test * len(file_main_list))
    if testing_file_list_len == 0 :
        testing_file_list_len = 1
    training_file_list_len = len(file_main_list) - (valid_file_list_len + testing_file_list_len)

    print('valid len')
    print(valid_file_list_len)
    print('test len')
    print(testing_file_list_len)
    print('train len')
    print(training_file_list_len)

    valid_list_file = []
    valid_list_file_name = []
    for i in range(valid_file_list_len):
        index_rand = int(random.random()*len(copy_main_file_list))
        valid_list_file.append(copy_main_file_list.pop(index_rand))
        valid_list_file_name.append(copy_file_name_main_list(index_rand))
    print("valid list ")
    print(valid_list_file)
    print("valid name list")
    print(valid_list_file_name)


    test_list_file = []
    test_list_file_name = []
    for i in range(testing_file_list_len):
        index_rand = int(random.random()*len(copy_main_file_list))
        test_list_file.append(copy_main_file_list.pop(index_rand))
        test_list_file_name.append(copy_file_name_main_list.pop((index_rand)))
    print("test list ")
    print(test_list_file)
    print("test list file name")
    print(test_list_file_name)


    
    train_list_file = copy_main_file_list
    train_list_file_name = copy_file_name_main_list
    print("train_list_file")
    print(train_list_file)
    print("train list file name")
    print(train_list_file_name)

    print("***********"*20)
    print(test_list_file)
    print(train_list_file)

    for test_old_file, train_old_file, in zip( test_list_file, train_list_file):

        
        print("test old file ")
        print(test_old_file)
        print("new folder")
        print(test_folder)
        shutil.copy(test_old_file, test_folder)
        print("train old file")
        print(train_old_file)
        print("new folder")
        print(train_folder)
        shutil.copy(train_old_file, train_folder)


"""
    for a in l1_valid:
        shutil.copy(os.path.join(all_root_0, a),valid_root_0)
    for a in l1_test:
        shutil.copy(os.path.join(all_root_0, a), test_root_0)
    for a in l1_train:
        shutil.copy(os.path.join(all_root_0, a), train_root_0)

"""
