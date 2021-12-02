#!/usr/bin/env python3

import pandas as pd
import xlsxwriter
import glob
import sys
import os


def writeIt(DFs, names):

    writer = pd.ExcelWriter('../output/WITHHELD.xlsx', engine='xlsxwriter')

    for df, name in zip(DFs, names):
        df.to_excel(writer, sheet_name = name)
        print(name)

    writer.save()


def main():

    # BASELINE 5-fold leave one out cross-validation
    # featureDirList = [("/tmp/tmp.1Uu3jV6Anl", "rasta"),
    #                   ("/tmp/tmp.2CKHTckEZ3", "ams"),
    #                   ("/tmp/tmp.OnCYm1VXui", "mfcc"),
    #                   ("/tmp/tmp.Wy3D2NOZ86", "combined"),
    #                   ("/tmp/tmp.TcRZJvM59O", "auditory"),
    #                   ("/tmp/tmp.Psc7g4V77e", "compare")]

    # Using withheld data
    featureDirList = [("/tmp/tmp.5LyuZwbVlX", "rasta"),
                      ("/tmp/tmp.WVvOwfh1Z8", "ams"),
                      ("/tmp/tmp.Sx7EPflQq2", "mfcc"),
                      ("/tmp/tmp.3TbF62kUeS", "combined"),
                      ("/tmp/tmp.7NmQXNGAif", "auditory"),
                      ("/tmp/tmp.YlVcGcjskQ", "compare")]

    DFs = list()
    names = list()
    for fDir, algorithm in featureDirList:
        IDs = glob.glob("{}/reports/*.csv".format(fDir))
        for id in IDs:
            name = algorithm + "-" + os.path.basename(id).split('.')[0]
            df = pd.read_csv(id)
            
            DFs.append(df)
            names.append(name)

    writeIt(DFs, names)


if __name__ == "__main__":

    main()
