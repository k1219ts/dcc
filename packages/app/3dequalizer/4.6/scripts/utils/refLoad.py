# 3DE4.script.name:    refLoad
# 3DE4.script.version: v1.5
# 3DE4.script.gui.button:   Manual Tracking Controls::RefLoad, align-bottom-right,70,20
# 3DE4.script.gui:    Main Window::Dexter
# 3DE4.script.comment: Imports multiple reference jpg images. First select the folder, then select the images you wish to import.

#James Porter
#v1.0 (Sept '15)  - Initial release for work. Includes flop reference images in nuke.
#v1.5 (June '16)  - For public release. Have stripped any mention of work's pipeline. 
#                   Have also removed Nuke flop part (that was a work releated thing)
#                   Tidier code than the previous version. (Which makes a change for me!)

import tde4
import os
import collections

tde4.clearConsole()


'''
if you have a pipline, and have environment variables (eg project), you can assign the folderPath variable to the project variables. 
Ask a grown up to help if you're not sure. Otherwise, I've left it to / depending which environment you're using (win,linux,osx)

'''

if 'linux' in sys.platform:
    folderPath = os.environ['HOME']

elif 'darwin' in sys.platform:
    folderPath = os.environ['HOME']
else:
    if 'win' in sys.platform:
        folderPath = os.environ['TEMP']



def importImg(FILE_IN,folder):
    lens_id = tde4.getFirstLens()
    referencePhoto = len(FILE_IN)
    
    if lens_id ==None:
        print "No lens found in the scene. Making a lens for you. Remove this after with the correct lens for your image"
        tde4.createLens()
        
    for ref in range(referencePhoto):
        filePath = "%s/%s" %(folder,FILE_IN[ref])
        camera = tde4.createCamera("REF_FRAME")
        tde4.setCameraPath(camera,filePath)
        tde4.setCameraLens(camera,lens_id)
        tde4.setCameraName(camera,FILE_IN[ref])
        
    print "**********************************************"
    print "* Import complete. Now import correct lenses *"
    print "**********************************************"
    
    
    
def selectAllPoints(r,widget,action):
    count = tde4.getListWidgetNoItems(r,'img')
    for i in range(count):
        tde4.setListWidgetItemSelectionFlag(r,'img',i,1)
        
def selectNoPoints(r,widget,action):
    count = tde4.getListWidgetNoItems(r,'img')
    for i in range(count):
        tde4.setListWidgetItemSelectionFlag(r,'img',i,0)    
        
        
def listImgs(folder):
    
    i       = 0
    imgs    = collections.defaultdict(list)
    imgFile = []
    
    files = os.listdir(folder)
    files.sort
    
    for file in files:
        #You get th picture here, if you're using PNG or tiffs, you can adjust this script to add extra file extensions.
        if file.endswith(".jpg") or file.endswith(".JPG") or file.endswith(".jpeg") or file.endswith(".JPEG") or file.endswith(".png"):
            imgs[i] = file
            i+=1 
            #I'm going old skool with the i = i+1 thing. I had errors with enumerate.
            
            
    #Now populate a requester with a list of imgs in requester 2!
    
    r = tde4.createCustomRequester()
    tde4.addListWidget(r,'img','Import',350,400)
    tde4.addButtonWidget(r,'selectAll','Select All')
    tde4.addButtonWidget(r,'selectNone','Select None')
    #tde4.setWidgetOffsets(r,'selectAll',55,10,10,10) #That was for another time. Another era....
    #tde4.setWidgetOffsets(r,'selectNone',95,10,10,10)
    
    tde4.setWidgetCallbackFunction(r,'selectAll',"selectAllPoints")
    tde4.setWidgetCallbackFunction(r,'selectNone',"selectNoPoints")
    
    for each in range(i):
        tde4.insertListWidgetItem(r,'img',str(imgs[each]),each)
    requester = tde4.postCustomRequester(r,'Select images you wish to import',450,530,"OK","Cancel")
    
    if requester == 1:
        FILE_IN = []
        noOfImgs = tde4.getListWidgetNoItems(r,'img')
        
        #Make sure they actually select something
        
        for n in range(noOfImgs):
            if tde4.getListWidgetItemSelectionFlag(r,'img',n) == 1:
                selection = tde4.getListWidgetItemUserdata(r,'img',n)
                FILE_IN.append(imgs[selection])
                
        if not FILE_IN:
            tde4.postQuestionRequester(r,"Nothing selected","OK")
            listImgs(folder)
        else:
            importImg(FILE_IN,folder)
    else:
        print "Cancelled"
    
    



def main():
    req = tde4.createCustomRequester()
    tde4.addFileWidget(req,"folderName","Select Folder...","*",folderPath)
    ret = tde4.postCustomRequester(req,"Load Multiple Reference Images",700,100,"  OK  ", "Cancel")
    
    if ret==1:
        #TODO: Sort out correct error handling with try/except. For now, it'll do
        folder = tde4.getWidgetValue(req,'folderName')
        if os.path.isdir(folder):
            listImgs(folder)
        else:
            tde4.postQuestionRequester(req,"Just select a folder!!! If you've pasted the path in, make sure there's no spaces at the beginnin/end","OK")
            main()

    else:
        print "Cancelled"


main()
