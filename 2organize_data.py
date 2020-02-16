import os
import shutil
from statistics import mean
import random



class Organization2:
    def __init__(self, main_root):
        self.main_root = main_root



    def mean_by_folder(self, root):
        list_name = os.listdir(root)
        amount_by_folder = []
        for x in list_name:
            amount_by_folder.append(len(os.listdir(os.path.join(self.main_root, x))))
        return mean(amount_by_folder)

    def min_by_folder(self, root):
        list_name = os.listdir(root)
        amount_by_folder = []
        for x in list_name:
            amount_by_folder.append(len(os.listdir(os.path.join(self.main_root, x))))
        return {"name": list_name[amount_by_folder.index(min(amount_by_folder))], "min" : min(amount_by_folder)}

    def folder_erase_elements(self, root, m):
        list_name = os.listdir(root)
        for x in list_name:
            #print(list_name)
            folder_path = os.path.join(self.main_root, x)
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

    

    def random_multi_delete(self, folder_path, amount):
        list_folder = os.listdir(folder_path)
        random.shuffle(list_folder)
        delete_list = list_folder[0:amount]
        print(len(delete_list))
        for file_name in delete_list:
            file_path = os.path.join(folder_path,file_name)
            #print(file_path)
            os.remove(file_path)


