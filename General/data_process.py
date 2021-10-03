# download all keel dataset
# the file structure is e.g. data >  imb_IRlowerThan9 >
# file name 1 (need to loop) > file name + "-5-fold".zip (need to be comporess)

# return path : data >  imb_IRlowerThan9 >
# file name (need to loop) > file name + "file name-5-fold"  process into list (use append)

import os
import zipfile
import pandas as pd

# input : e.g. data >  imb_IRlowerThan9
# turn the dat file into excel
# discover zip file & extract the file
# return the extract file path


def extractFile(imbalDataset):  # extract zip file and return the zip file directory
    temp = []
    for element in imbalDataset:
        for subdir, dirs, files in os.walk("../data/"+element):
            temp.append(subdir)
            for filename in files:
                filepath = subdir + os.sep + filename
                # print(filepath)
                if filepath.endswith(".zip") and "5-fold" in filepath and '.dat' not in filepath:
                    zip_list(filepath, subdir)
    return temp


def zip_list(file_path, file_position):
    # extract file_path 的 zip file
    # put it into file_position
    zf = zipfile.ZipFile(file_path, 'r')
    zf.extractall(file_position)


def single(filepath):
    '''参数：txt文件路径，功能将其转换成excel文件'''
    # tempfile = filepath.split("/")
    # print('/'.join(tempfile[:-1]), "  dfsdfsdfsdf")
    # print(os.getcwd())
    # os.chdir(filepath)
    # print("csdc", os.getcwd())

    data = []
    newname = filepath.replace(".txt", ".xlsx")
    print("+++++++++++++++++++++++++++++")
    print(newname)
    fp = open(filepath, "r", encoding="utf-8")
    for line in fp:  # 设置文件对象并读取每一行文件
        # print(type(line)) # str
        line = line.split('|@|')
        data.append(line)

    # 抓出屬性
    attr = []
    for i in data:
        if("attribute" in i[0]):
            i = i[0].split(" ")
            attr.append(i[1])

    # 抓出資料
    realdata = []
    datastart = False
    for i in data:
        if(datastart is True):
            i = i[0].split(",")
            """
            for j in i:
                if("\n" in j):
                    print(j)
                    realdata[]
            """
            realdata.append(i)
        if("@data" in i[0]):
            datastart = True
    print(len(realdata))

    dd = pd.DataFrame(data=realdata, columns=attr,
                      index=list(range(1, len(realdata)+1)))

    dd.to_excel(newname)
    row_num = len(data)
    col_num = len(data[0])  # 如果是空文件此处报错

    """
    # 步骤2：创建工作簿对象workbook
    workbook = xlwt.Workbook(encoding='utf-8')
    # 步骤3：创建单页对象sheet
    sheet = workbook.add_sheet('测试单页1')
    # 步骤5：写入内容数据
    # 步骤5-1：外层for循环控制行数
    for rowIndex in range(0, row_num):
    # 步骤5-2：内层for循环控制列数
        for colIndex in range(col_num):
        # 步骤5-3：写入内容数据
            sheet.write(rowIndex, colIndex, data[rowIndex][colIndex])
    # 步骤6：保存工作簿
    workbook.save(newname)
    """


def dat_txt(files_path):  # folder of every dataset the input will be a list
    '''將 .dat 改成 .txt'''
    origin = os.getcwd()
    for filepath in files_path:
        files = os.listdir(filepath)
        for file in files:
            portion = os.path.splitext(file)
            if portion[1] == ".dat":  # 如果后缀是.dat
                # 重新组合文件名和后缀名
                #print("dsd", file)
                newname = portion[0] + ".txt"
                #print("now", os.getcwd())
                os.chdir(origin+"/"+filepath)
                os.rename(file, newname)
                os.chdir(origin)


def get_txtpath(files_path):  # folder of every dataset input will be a list
    '''obtain every txt router'''
    # files = os.listdir('.')
    # print('files',files)
    data = []

    for filepath in files_path:
        # print(filepath)
        files = os.listdir(filepath)

        for filename in files:
            portion = os.path.splitext(filename)
            if portion[1] == ".txt":
                #txtpath = os.getcwd() + os.sep + filename
                txtpath = filepath + os.sep + filename
                data.append(txtpath)

    #print("++++++++++++  data +++++++++++++++")
    # print(data)
    return data


def get_excel(imbalDataset):  # in order to get all excel folder
    excelfolder = []
    for element in imbalDataset:
        # os.chdir("../data")
        for subdir, dirs, files in os.walk("..\data"):
            # print(subdir)
            tempelement = element+'\\'
            if(tempelement in subdir):
                excelfolder.append(subdir)
                # print(subdir)
    return excelfolder


if __name__ == "__main__":
    imbalDataset = ["imb_IRhigherThan9p3"]  # imb_IRlowerThan9
    # temp = []
    allpath = extractFile(imbalDataset)
    print(allpath)
    dat_txt(allpath)
    txtpath = get_txtpath(allpath)
    get_excel(imbalDataset)

    for index, element in enumerate(txtpath):
        # print("cdsc", element)
        single(element)
