import os 
import glob

path = "/mmlabworkspace/WorkSpaces/danhnt/tuyensh/tungphan/ABINetpp/ABINet-PP/datasets/vintext_origin/unseen_test_images"
labels = os.listdir("/mmlabworkspace/WorkSpaces/danhnt/tuyensh/tungphan/ABINetpp/ABINet-PP/datasets/vintext_origin/unseen_test_images")
for file in labels:
    c = 0
    for i in file[2:]:
        if i == "0":
            c +=1
        else:
            break
    c += 2
    new_name = file[c:]
    print(file)
    print(new_name)
    os.rename(os.path.join(path,file), os.path.join(path, new_name))