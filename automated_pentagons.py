import re
import cv2
import os
import numpy as np
import tkinter.filedialog
from tkinter import *
import shutil
import corner_dets_methods
import datetime
import tkinter as tk
from time import sleep
from math import trunc
from tkinter import ttk
import csv

good_total = 0

class AutoApp:
    def __init__(self, master):
        self.master = master
        master.wm_title("Automation")

        self.corner_num = 500 #Do not limit corners
        self.quality = 0.1  #Accepts best 90% of corners found
        self.distance = 10 #Will not detect corners with in 10px of another detected corner
        self.detection_threshold = 0.00
        self.line_threshold = .95
        self.corner_total = 0
        self.line_total = 0
        self.det_core = 0
        self.count = 0
        self.line_thresh = 10
        self.line_length = 10
        self.line_gap = 4

        self.cornersList = []
        self.linesList = []
        self.gapList = []
        self.distilled_lines = []
        self.bounding_boxes = []
        self.box_image = None

        self.path = tkinter.filedialog.askopenfilename()
        #self.path = os.path.dirname(self.path)
        #self.path = r'C:\Users\KinectProcessing\Desktop\sample_pents_output'
        #self.paths = self.make_list_dir(top_path)

        self.directory = os.path.dirname(self.path)

        self.label = Label(master, text="Automating Pentagon output")
        self.label.grid(row=0, column=0, sticky="N")

        self.progress_var = DoubleVar()

        self.progress = ttk.Progressbar(master, length=100)
        self.progress.grid(row=1, column=0, sticky="NE")
        self.progress.after(1, self.select_pentagon())

        self.master.destroy()

    def make_list_dir(self, path):
        for _,dirs,_ in os.walk(path):
            return dirs

    def detection_funct(self, path):
        detections, found_dets, output_image, bounding_boxes, box_image = corner_dets_methods.find_contours(path, self.corner_num, self.quality, self.distance, self.detection_threshold)
        return detections, found_dets, output_image, bounding_boxes, box_image

    def select_pentagon(self):
        global good_total
        file_list = self.make_file_list()
        #print(len(file_list))
        step = (100 / len(file_list))
        for file in file_list:
            if file.endswith('.png'):
                self.image_path = file
                #print(self.image_path)
                #print(file)
                im = cv2.imread(file)
                #cv2.imshow("t", im)
                #cv2.waitKey()
                gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
                det, found_details, output_image, bounding_boxes, box_image = self.detection_funct(file)
                if found_details != []:
                    self.corner_total, self.line_total = self.corner_details(gray, found_details[0])
                    self.gapList = corner_dets_methods.gapCorners(im, self.cornersList)
                    self.bounding_boxes = bounding_boxes
                    self.box_image = box_image
                    #if self.corner_total == 12:
                    #    good_total += 1
                    self.save_cmd(file, output_image)
                self.progress.step(step)
                self.progress.update()
                #print(good_total)

    def make_file_list(self):
        f = []
        # for (_,dirs,_) in os.walk(self.path):
        #     for d in dirs:
        #         base_path = os.path.join(self.path, d)
        for (_, _, file_names) in os.walk(self.directory):
            for i in file_names:
                if i.endswith(".png"):
                    fullfill = (os.path.join(self.directory, i))
                    f.append(fullfill)
        #print(f)
        return f

    def corner_details(self, image, detections):
        self.cornersList, self.linesList, self.distilled_lines = corner_dets_methods.cornerMeths(image,
            [detections],
            self.corner_num,
            self.quality,
            self.distance,
            self.line_thresh,
            self.line_length,
            self.line_gap)
        line_count = len(self.linesList)
        corner_count = len(self.cornersList)
        return corner_count, line_count

    def name_save_files(self, originalFileName):
        saveFileList = []
        baselineTerms = ["BL", "Baseline", "Base"]
        fuTerms = ["FU", "F-U"]
        try:
            originalFileName = (os.path.split(originalFileName)[-1])
            print(originalFileName)
            regex = "^[0-9]{8}\s+"
            if re.match(regex, originalFileName):
                removeString = re.search('img-(.*)png', originalFileName)
                file = originalFileName.replace("img-" + removeString.group(1) + "png", '')
                dateFind = re.search('[0-9]+-[0-9]+-[0-9]+', file)
                file = file.replace(dateFind.group(0), '')
                result = re.search(regex, file)
                projID = result.group(0).strip()
                # baseline
                #print(file)
                if any(x in file for x in baselineTerms):
                    saveFileList.append(projID + "_" + "0" + "_Output.txt")
                    saveFileList.append(projID + "_" + "0" + "_Corner.csv")
                    saveFileList.append(projID + "_" + "0" + "_Line.csv")
                    saveFileList.append(projID + "_" + "0" + "_Gap.csv")
                    saveFileList.append(projID + "_" + "0" + "_Figure.jpg")
                    saveFileList.append(projID + "_" + "0" + "_Inside_contours.csv")
                    saveFileList.append(projID + "_" + "0" + "_Bounding_Boxes.csv")
                    saveFileList.append(projID + "_" + "0" + "_Box_image.jpg")


                elif (x in file for x in fuTerms):
                    fuYear = file.replace(projID, '')
                    if ("FU" in file):
                        fuYear = fuYear.replace("FU", '')
                    if ("F-U" in file):
                        fuYear = fuYear.replace("F-U", '')
                    fuYear = fuYear.strip()
                    if (fuYear != ""):
                        saveFileList.append(projID + "_" + fuYear + "_Output.txt")
                        saveFileList.append(projID + "_" + fuYear + "_Corner.csv")
                        saveFileList.append(projID + "_" + fuYear + "_Line.csv")
                        saveFileList.append(projID + "_" + fuYear + "_Gap.csv")
                        saveFileList.append(projID + "_" + fuYear + "_Figure.jpg")
                        saveFileList.append(projID + "_" + fuYear + "_Inside_contours.csv")
                        saveFileList.append(projID + "_" + fuYear + "_Bounding_Boxes.csv")
                        saveFileList.append(projID + "_" + fuYear + "_Box_image.jpg")


                else:
                    saveFileList.append(projID + "_" + "ERROR" + "_Output.txt")
                    saveFileList.append(projID + "_" + "ERROR" + "_Corner.csv")
                    saveFileList.append(projID + "_" + "ERROR" + "_Line.csv")
                    saveFileList.append(projID + "_" + "ERROR" + "_Gap.csv")
                    saveFileList.append(projID + "_" + "ERROR" + "_Figure.jpg")
                    saveFileList.append(projID + "_" + "ERROR" + "_Inside_contours.csv")
                    saveFileList.append(projID + "_" + "ERROR" + "_Bounding_Boxes.csv")
                    saveFileList.append(projID + "_" + "ERROR" + "_Box_image.jpg")


            # print(saveFileList)
            return saveFileList
        except:
            return ["Error_output.txt", "Error_Corner.csv", "Error_Line.csv", "Error_Gap", "Error_Figure", "Error_Inside_contours.csv", "Error_Bounding_Boxes.csv", "Error_Box_image.jpg"]



    def save_cmd(self, infile, output_image):
        try:
            base, ext = os.path.splitext(infile)
            #save_file = r"C:\Users\KinectProcessing\Documents\HistoricalPentAnalysis"
            save_file = os.path.join(os.getcwd(), f"Pent_Output_{self.line_thresh}_{self.line_length}_{self.line_gap}")
            if not os.path.exists(save_file):
                os.makedirs(save_file)
            # os.path.join(os.path.dirname(infile),base)
            #print(save_file)
            # with open(save_file, "w") as text_file:
            #     text_file.write("MMSE Pentagon GUI Version=1\n")
            #     text_file.write("Process date = {0}\n".format(datetime.datetime.now()))
            #     text_file.write("corner count =  {0}\n".format(self.corner_total))
            #     text_file.write("line count =  {0}\n".format(self.line_total))
            #     text_file.write("min corner count =  {0}\n".format(self.corner_num))
            #     text_file.write("corner list =  {0}\n".format(self.cornersList))
            #     text_file.write("lines list =  {0}\n".format(self.linesList))
            save_file_list = self.name_save_files(infile)
            print(save_file_list[0])
               ## Output Textfiles
            try:
                with open(os.path.join(save_file, save_file_list[0]), "w") as text_file:
                    text_file.write("MMSE Pentagon GUI Version=1\n")
                    text_file.write("Process date = {0}\n".format(datetime.datetime.now()))
                    text_file.write("Original filename = " + infile + "\n")
                    text_file.write("corner count =  {0}\n".format(str(len(self.cornersList))))
                    text_file.write("line count =  {0}\n".format(str(len(self.linesList))))
                    text_file.write("corner list =  {0}\n".format(str(self.cornersList)))
                    text_file.write("lines list =  {0}\n".format(str(self.linesList)))

                # save cornerlist
                with open(os.path.join(save_file, save_file_list[1]), "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(self.cornersList)

                # save Line.csv
                with open(os.path.join(save_file,save_file_list[2]), "w", newline="") as f:
                    writer = csv.writer(f)
                    for i in self.linesList:
                        writer.writerows([i[0], i[1]])
                
                #save Gap.csv
                with open(os.path.join(save_file,save_file_list[3]), "w", newline="") as f:
                    writer = csv.writer(f)
                    for i in self.gapList:
                        writer.writerows([i[0], i[1]])
                
                #save Figure.jpg
                cv2.imwrite(os.path.join(save_file,save_file_list[4]),output_image)

                #INSIDE CONTORURTS
                # inside_cnts = corner_dets_methods.inside_contours(output_image)
                # with open(os.path.join(save_file,save_file_list[5]), "w", newline="") as f:
                #     writer = csv.writer(f)
                #     writer.writerows(inside_cnts)
                
                #BOUNDING BOXES
                print("bounding boxes")
                with open(os.path.join(save_file,save_file_list[6]), "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(self.bounding_boxes)

                #BOX IMAGE
                print("box image")
                cv2.imwrite(os.path.join(save_file,save_file_list[7]),self.box_image)

            except Exception as e: print(e)
        except:
            print("Error processing: " + infile)



if __name__ == '__main__':
    root = Tk()
    my_gui = AutoApp(root)
    root.mainloop()

print("COMPLETE")
