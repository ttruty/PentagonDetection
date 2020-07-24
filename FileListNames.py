import os,re

fileCount = 0

baselineTerms = ["BL", "Baseline", "Base"]
fuTerms = ["FU", "F-U"]
baselineCount = 0
fuCount = 0

dirWorkable = r"C:\Users\KinectProcessing\Documents\DigiPen Studies\Scanned_workablepentagons"
# dirError = r"C:\Users\KinectProcessing\Documents\DigiPen Studies\Scanned_Problem_pents"
for root, dirs, files in os.walk(dirWorkable):
    for file in files:
        try:
            #print(file)
            regex = "^[0-9]{8}\s+"
            if re.match(regex, file):
                removeString = re.search('img-(.*)png', file)
                file = file.replace("img-" + removeString.group(1) + "png", '')
                dateFind = re.search('[0-9]+-[0-9]+-[0-9]+', file)
                file = file.replace(dateFind.group(0), '')
                result = re.search(regex, file)
                projID = result.group(0).strip()
                #baseline
                #print(file)
                if any(x in file for x in baselineTerms):
                    print(projID + "_" + "0" + "_Corner")
                    print(projID + "_" + "0" + "_Line")
                    print(projID + "_" + "0" + "_Gap")
                    baselineCount+=1
                elif (x in file for x in fuTerms):
                    fuYear = file.replace(projID, '')
                    if ("FU" in file):
                        fuYear = fuYear.replace("FU", '')
                    if ("F-U" in file):
                        fuYear = fuYear.replace("F-U", '')
                    fuYear = fuYear.strip()
                    if (fuYear != ""):
                        print(projID + "_" + fuYear + "_Corner")
                        print(projID + "_" + fuYear + "_Line")
                        print(projID + "_" + fuYear + "_Gap")
                        fuCount += 1
        except:
             pass

print(baselineCount)
print(fuCount)

