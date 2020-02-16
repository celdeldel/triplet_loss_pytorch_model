import os
import shutil
from statistics import mean
import random

main_root = '2data/data'

def mean_by_folder(root):
    list_name = os.listdir(root)
    amount_by_folder = []
    for x in list_name:
        amount_by_folder.append(len(os.listdir(os.path.join(main_root, x))))
    return mean(amount_by_folder)

def min_by_folder(root):
    list_name = os.listdir(root)
    amount_by_folder = []
    for x in list_name:
        amount_by_folder.append(len(os.listdir(os.path.join(main_root, x))))
    return {"name": list_name[amount_by_folder.index(min(amount_by_folder))], "min" : min(amount_by_folder)}

def folder_erase_elements(root, m):
    list_name = os.listdir(root)
    for x in list_name:
        #print(list_name)
        folder_path = os.path.join(main_root, x)
        if len(os.listdir(folder_path)) < m:
            print("suppression du dossier suivant")
            print(folder_path)
            print(len(os.listdir(folder_path))) 
            shutil.rmtree(folder_path)
            continue
        if len(os.listdir(folder_path)) >m :
            print("old len")
            print(len(os.listdir(folder_path)))
            random_multi_delete(folder_path, len(os.listdir(folder_path))-m)
            print("new len")
            print(len(os.listdir(folder_path)))

#def amount_more_than_folder(root):
    

def random_multi_delete(folder_path, amount):
    list_folder = os.listdir(folder_path)
    random.shuffle(list_folder)
    delete_list = list_folder[0:amount]
    print(len(delete_list))
    for file_name in delete_list:
        file_path = os.path.join(folder_path,file_name)
        #print(file_path)
        os.remove(file_path)



if __name__ == '__main__':
    folder_erase_elements(main_root, 30)
    print(mean_by_folder(main_root))


