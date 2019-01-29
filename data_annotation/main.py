#-------------------------------------------------------------------------------
# Name:        Object bounding box label tool
# Purpose:     Label object bboxes for ImageNet Detection data
#
#-------------------------------------------------------------------------------
from __future__ import division
from tkinter import *
from PIL import Image, ImageTk
import os
import glob
import random
import json

# colors for the bboxes
COLORS = ['red', 'blue', 'yellow', 'pink', 'green', 'black']
# image sizes for the examples
SIZE = 256, 256
NORMAL = 1
SLIGHT_ANORMAL = 2
SEVERE_ANORMAL = 3

class LabelTool():
    def __init__(self, master):
        # set up the main frame
        self.width = 960
        self.height = 780
        self.img_start_x = 0
        self.img_start_y = 0
        self.parent = master
        self.parent.title("LabelTool")
        self.parent.geometry("{}x{}".format(self.width, self.height))
        self.frame = Frame(self.parent)
        self.frame.pack(fill=BOTH, expand=1)
        self.parent.resizable(width = FALSE, height = FALSE)

        # initialize global state
        self.imageDir = ''
        self.imageList= []
        self.egDir = ''
        self.egList = []
        self.outDir = ''
        self.cur = 0
        self.total = 0
        self.category = 0
        self.imagename = ''
        self.labelfilename = ''
        self.tkimg = None
        self.listbox = Listbox(self.frame, width=22, height=12)

        # initialize mouse state
        self.STATE = {}
        self.STATE['click'] = 0
        self.STATE['x'], self.STATE['y'] = 0, 0

        # reference to bbox
        self.bboxIdList = []
        self.bboxId = None
        self.bboxList = []
        self.hl = None
        self.vl = None

        # reference to class
        self.classes = [] # classes information
        self.class_btns = [] # class btns
        self.class_points = [] # class points on the picture
        self.current_class = None

        # task info
        self.MODE_TRANS = 0
        self.MODE_DEGREE = 1
        self.task_type = None
        self.task_btns = []

        # ----------------- GUI stuff ---------------------
        # dir entry & load
        self.label = Label(self.frame, text = "Image Dir:")
        self.label.grid(row = 0, column = 0, sticky = E)
        self.entry = Entry(self.frame)
        self.entry.grid(row = 0, column = 1, sticky = W+E)
        self.ldBtn = Button(self.frame, text = "Load", command = self.loadDir)
        self.ldBtn.grid(row = 0, column = 2, sticky = W+E)
        self.label_dir = ""
        if not os.path.exists(r'Labels'):
            os.mkdir(r'Labels')
        self.json_data = {}

        # main panel for labeling
        self.mainPanel = Canvas(self.frame, cursor='tcross')
        self.mainPanel.bind("<Button-1>", self.mouseClick)
        self.mainPanel.bind("<Motion>", self.mouseMove)
        self.parent.bind("<Escape>", self.cancelBBox)  # press <Espace> to cancel current bbox
        self.parent.bind("s", self.cancelBBox)
        self.parent.bind("a", self.prevImage) # press 'a' to go backforward
        self.parent.bind("d", self.nextImage) # press 'd' to go forward
        self.mainPanel.grid(row = 1, column = 1, rowspan = 4, sticky = W+N)

        # hint
        self.hintPanel = Frame(self.frame)
        self.hintPanel.grid(row=1, column=2, sticky=W)
        self.hintLabel = Label(self.hintPanel, text="提示文字:")
        self.hintLabel.grid(row=0, column=0, sticky=N)
        self.hintText = Label(self.hintPanel, justify="right", wraplength=200)
        self.hintText.grid(row=1, column=1, sticky=N)

        # trans area
        self.draw_task_buttons()

        # control panel for image navigation
        self.ctrPanel = Frame(self.frame)
        self.ctrPanel.grid(row = 5, column = 1, columnspan = 2, sticky = W+E)
        self.prevBtn = Button(self.ctrPanel, text='<< 上一张', width = 10, command = self.prevImage)
        self.prevBtn.pack(side = LEFT, padx = 5, pady = 3)
        self.nextBtn = Button(self.ctrPanel, text='下一张 >>', width = 10, command = self.nextImage)
        self.nextBtn.pack(side = LEFT, padx = 5, pady = 3)
        self.progLabel = Label(self.ctrPanel, text = "Progress:     /    ")
        self.progLabel.pack(side = LEFT, padx = 5)
        self.tmpLabel = Label(self.ctrPanel, text = "前往图片 No.")
        self.tmpLabel.pack(side = LEFT, padx = 5)
        self.idxEntry = Entry(self.ctrPanel, width = 5)
        self.idxEntry.pack(side = LEFT)
        self.goBtn = Button(self.ctrPanel, text = 'Go', command = self.gotoImage)
        self.goBtn.pack(side = LEFT)
        self.saveBtn = Button(self.ctrPanel, text='保存修改', width=10, command=self.saveCurrent)
        self.saveBtn.pack(side=LEFT)
        self.btnOrigColor = self.saveBtn.cget("background")

        self.disp = Label(self.ctrPanel, text='')
        self.disp.pack(side = RIGHT)

        self.frame.columnconfigure(1, weight = 1)
        self.frame.rowconfigure(4, weight = 1)
        self.json_file_name = ""

    def loadDir(self, dbg = False):
        if not dbg:
            s = self.entry.get()
            self.parent.focus()
            self.category = int(s)
        else:
            s = r'D:\workspace\python\labelGUI'
        # get image list
        self.imageDir = os.path.join(r'.\Images', '%03d' %(self.category))
        self.labelDir = os.path.join(r'.\Labels', '%03d' % self.category)
        if not os.path.exists(self.labelDir):
            os.mkdir(self.labelDir)
        self.label_dir = "./Labels/%03d/" %self.category
        if not os.path.exists(self.label_dir):
            os.mkdir(self.label_dir)
        self.imageList = glob.glob(os.path.join(self.imageDir, '*.jpg'))

        if len(self.imageList) == 0:
            print ('No .JPEG images found in the specified dir!')
            return

        # default to the 1st image in the collection
        self.cur = 1
        self.total = len(self.imageList)

        # load example bboxes
        # self.egDir = os.path.join(r'./Examples', '%03d' %(self.category))
        # if not os.path.exists(self.egDir):
        #     return
        filelist = glob.glob(os.path.join(self.egDir, '*.jpg'))
        self.tmp = []
        self.egList = []
        random.shuffle(filelist)
        for (i, f) in enumerate(filelist):
            if i == 3:
                break
            im = Image.open(f)
            r = min(SIZE[0] / im.size[0], SIZE[1] / im.size[1])
            new_size = int(r * im.size[0]), int(r * im.size[1])
            self.tmp.append(im.resize(new_size, Image.ANTIALIAS))
            self.egList.append(ImageTk.PhotoImage(self.tmp[-1]))
            self.egLabels[i].config(image = self.egList[-1], width = SIZE[0], height = SIZE[1])

        self.new_picture()
        print ('%d images loaded from %s' %(self.total, s))
    
    def new_picture(self):
        self.task_btn_clear()
        self.draw_task_buttons()
        self.task_type = None
        self.loadImage()
    
    def task_btn_clear(self):
        if self.task_btns:
            for btn in self.task_btns:
                btn.destroy()
        self.task_btns = []
        
    def draw_task_buttons(self):
        self.operatePanel = Frame(self.frame)
        self.operatePanel.grid(row=1, column=2, sticky=W)

        self.transBtn = Button(self.operatePanel, text='标注转化区', command=self.task_trans)
        self.transBtn.grid(row=1, column=1, sticky=W)
        self.operatePanel.grid(row=2, column=2, sticky=W)
        self.degreeBtn = Button(self.operatePanel, text='标注类别', command=self.task_class)
        self.degreeBtn.grid(row=1, column=3, sticky=W)

    def task_trans(self):
        self.task_type = self.MODE_TRANS
        self.changeButtonColor()
        self.lb1 = Label(self.frame, text = '已标注的转化区')
        self.lb1.grid(row = 3, column = 2,  sticky = W+N)
        self.task_btns.append(self.lb1)
        self.listbox = Listbox(self.frame, width = 22, height = 12)
        self.listbox.grid(row = 4, column = 2, sticky = N)
        self.task_btns.append(self.listbox)
        self.btnDel = Button(self.frame, text = '删除', command = self.delBBox)
        self.btnDel.grid(row = 5, column = 2, sticky = W+E+N)
        self.task_btns.append(self.btnDel)
        self.btnClear = Button(self.frame, text = '清空所有', command = self.clearBBox)
        self.btnClear.grid(row = 6, column = 2, sticky = W+E+N)
        self.task_btns.append(self.btnClear)
        self.loadImage()
        # clear panel
        for idx in range(len(self.class_points)):
            self.mainPanel.delete(self.class_points[idx])
        self.listbox.delete(0, self.listbox.size())
        for tmp in self.json_data["bbox"]:
            tmpId = self.mainPanel.create_rectangle(tmp[0], tmp[1], tmp[2], tmp[3], width=2,
                                                    outline=COLORS[(len(self.json_data["bbox"])-1) % len(COLORS)])
            self.bboxIdList.append(tmpId)
            self.listbox.insert(END, '(%d, %d) -> (%d, %d)' % (tmp[0], tmp[1], tmp[2], tmp[3]))

    def task_class(self):
        self.task_type = self.MODE_DEGREE
        self.changeButtonColor()
        self.lb1 = Label(self.frame, text='已标注的类别')
        self.lb1.grid(row=3, column=2, sticky=W + N)
        self.task_btns.append(self.lb1)
        self.listbox = Listbox(self.frame, width=22, height=12)
        self.listbox.grid(row=4, column=2, sticky=N)
        self.task_btns.append(self.listbox)
        self.btnDel = Button(self.frame, text='删除', command=self.delClass)
        self.btnDel.grid(row=5, column=2, sticky=W + E + N)
        self.task_btns.append(self.btnDel)
        self.btnClear = Button(self.frame, text='清空所有', command=self.clearClass)
        self.btnClear.grid(row=6, column=2, sticky=W + E + N)
        self.task_btns.append(self.btnClear)
        self.loadImage()
        ## clear panel
        for idx in range(len(self.bboxIdList)):
            self.mainPanel.delete(self.bboxIdList[idx])
        self.listbox.delete(0, self.listbox.size())
        for coor in self.json_data["class"]:
            [current_x, current_y, cls] = coor
            self.listbox.insert(END, '%d, %d, %d' % (current_x, current_y, cls))
            x1, y1 = (current_x - 5), (current_y - 5)
            x2, y2 = (current_x + 5), (current_y + 5)
            if cls == NORMAL:
                tmpId = self.mainPanel.create_oval(x1, y1, x2, y2, fill='green')
            elif cls == SLIGHT_ANORMAL:
                tmpId = self.mainPanel.create_oval(x1, y1, x2, y2, fill='yellow')
            else :
                tmpId = self.mainPanel.create_oval(x1, y1, x2, y2, fill='red')
            self.class_points.append(tmpId)


    def loadImage(self):
        # load image
        imagepath = self.imageList[self.cur - 1]
        self.img = Image.open(imagepath)
        self.tkimg = ImageTk.PhotoImage(self.img)
        self.mainPanel.config(width = max(self.tkimg.width(), 400), height = max(self.tkimg.height(), 400))
        self.mainPanel.create_image(self.img_start_x, self.img_start_y, image = self.tkimg, anchor=NW)
        self.progLabel.config(text = "%04d/%04d" %(self.cur, self.total))
        self.imagename = os.path.split(imagepath)[-1].split('.')[0]
        self.json_file_name = os.path.join(self.label_dir, self.imagename + '.json')
        if not os.path.exists(self.json_file_name):
            with open(self.json_file_name, 'w+') as file:
                file.write(json.dumps({"bbox": [], "class": []}))
                file.close()
        with open(self.json_file_name, 'r+') as file:
            self.json_data = json.loads(file.read())
            file.close()


    def saveImage(self, finished=True):
        import json
        with open(self.json_file_name, 'w+') as f:
            f.write(json.dumps(self.json_data))
            f.close()
        # print('Image No. %d saved' % (self.cur))
        self.deleteJson()

        if self.task_type != None and finished:
            self.clearClass()
            self.clearBBox()

    def saveCurrent(self):
        if self.json_file_name != '':
            import json
            with open(self.json_file_name, 'w+') as f:
                f.write(json.dumps(self.json_data))
                f.close()
            # print('Image No. %d saved' % (self.cur))
            self.deleteJson()

    def deleteJson(self):
        delete_flag = True
        for key in self.json_data:
            if self.json_data[key]:
                delete_flag = False
        if delete_flag and self.json_file_name:
            os.remove(self.json_file_name)

    def changeButtonColor(self):
        if self.task_type == self.MODE_TRANS:
            self.degreeBtn.configure(bg=self.btnOrigColor)
            self.transBtn.configure(bg='red')
        elif self.task_type == self.MODE_DEGREE:
            self.transBtn.configure(bg=self.btnOrigColor)
            self.degreeBtn.configure(bg='red')
        else:
            self.transBtn.configure(bg=self.btnOrigColor)
            self.degreeBtn.configure(bg=self.btnOrigColor)


    def mouseClick(self, event):
        if self.task_type == self.MODE_TRANS:
            if self.STATE['click'] == 0:
                self.STATE['x'], self.STATE['y'] = event.x, event.y
            else:
                x1, x2 = min(self.STATE['x'], event.x), max(self.STATE['x'], event.x)
                y1, y2 = min(self.STATE['y'], event.y), max(self.STATE['y'], event.y)
                self.bboxIdList.append(self.bboxId)
                self.bboxId = None
                self.listbox.insert(END, '(%d, %d) -> (%d, %d)' %(x1, y1, x2, y2))
                self.json_data["bbox"].append([x1, y1, x2, y2])
            self.STATE['click'] = 1 - self.STATE['click']
        elif self.task_type == self.MODE_DEGREE:
            current_x, current_y = event.x, event.y
            labels = ['正常', '轻微异常', '严重异常']
            def on_click_label1():
                self.json_data["class"].append((current_x, current_y, NORMAL))
                self.listbox.insert(END, '%d, %d, 正常' %(current_x, current_y))
                x1, y1 = (current_x - 5), (current_y - 5)
                x2, y2 = (current_x + 5), (current_y + 5)
                tmpId = self.mainPanel.create_oval(x1, y1, x2, y2, fill='green')
                self.class_points.append(tmpId)
                for btn in self.class_btns:
                    btn.destroy()
                self.class_btns = []

            def on_click_label2():
                self.json_data["class"].append((current_x, current_y, SLIGHT_ANORMAL))
                self.listbox.insert(END, '%d, %d, 轻微异常' %(current_x, current_y))
                x1, y1 = (current_x - 5), (current_y - 5)
                x2, y2 = (current_x + 5), (current_y + 5)
                tmpId = self.mainPanel.create_oval(x1, y1, x2, y2, fill='yellow')
                self.class_points.append(tmpId)
                for btn in self.class_btns:
                    btn.destroy()
                self.class_btns = []

            def on_click_label3():
                self.json_data["class"].append((current_x, current_y, SEVERE_ANORMAL))
                self.listbox.insert(END, '%d, %d, 严重异常' %(current_x, current_y))
                x1, y1 = (current_x - 5), (current_y - 5)
                x2, y2 = (current_x + 5), (current_y + 5)
                tmpId = self.mainPanel.create_oval(x1, y1, x2, y2, fill='red')
                self.class_points.append(tmpId)
                for btn in self.class_btns:
                    btn.destroy()
                self.class_btns = []

            btn1 = Button(self.frame, text=labels[0], command=on_click_label1)
            btn1.place(x=current_x+5, y=current_y, height=30)
            self.class_btns.append(btn1)
            btn2 = Button(self.frame, text=labels[1], command=on_click_label2)
            btn2.place(x=current_x + 5, y=current_y + 30, height=30)
            self.class_btns.append(btn2)
            btn3 = Button(self.frame, text=labels[2], command=on_click_label3)
            btn3.place(x=current_x + 5, y=current_y + 30 * 2, height=30)
            self.class_btns.append(btn3)
        self.saveImage(False)

    def mouseMove(self, event):
        self.disp.config(text = 'x: %d, y: %d' %(event.x, event.y))
        if self.tkimg:
            if self.hl:
                self.mainPanel.delete(self.hl)
            self.hl = self.mainPanel.create_line(0, event.y, self.tkimg.width(), event.y, width = 2)
            if self.vl:
                self.mainPanel.delete(self.vl)
            self.vl = self.mainPanel.create_line(event.x, 0, event.x, self.tkimg.height(), width = 2)
        if 1 == self.STATE['click']:
            if self.bboxId:
                self.mainPanel.delete(self.bboxId)
            self.bboxId = self.mainPanel.create_rectangle(self.STATE['x'], self.STATE['y'], event.x, event.y, width=2,
                                                          outline=COLORS[len(self.json_data["bbox"]) % len(COLORS)])

    def cancelBBox(self, event):
        if 1 == self.STATE['click']:
            if self.bboxId:
                self.mainPanel.delete(self.bboxId)
                self.bboxId = None
                self.STATE['click'] = 0

    def delBBox(self):
        sel = self.listbox.curselection()
        if len(sel) != 1:
            return
        idx = int(sel[0])
        self.mainPanel.delete(self.bboxIdList[idx])
        self.bboxIdList.pop(idx)
        # self.bboxList.pop(idx)
        if self.listbox.size():
            self.listbox.delete(idx)
        self.json_data["bbox"].pop(idx)

    def clearBBox(self):
        for idx in range(len(self.bboxIdList)):
            self.mainPanel.delete(self.bboxIdList[idx])
        self.listbox.delete(0, self.listbox.size())
        self.json_data["bbox"] = []

    def clearClass(self):
        for idx in range(len(self.class_points)):
            self.mainPanel.delete(self.class_points[idx])
        self.listbox.delete(0, self.listbox.size())
        self.json_data["class"] = []

    def delClass(self):
        sel = self.listbox.curselection()
        if len(sel) != 1:
            return
        idx = int(sel[0])
        self.mainPanel.delete(self.class_points[idx])
        self.class_points.pop(idx)
        if self.listbox.size():
            self.listbox.delete(idx)
        self.json_data["class"].pop(idx)

    def prevImage(self, event = None):
        self.saveImage()
        if self.cur > 1:
            self.cur -= 1
            self.new_picture()
        else:
            self.cur = self.total
            self.new_picture()

    def nextImage(self, event = None):
        self.saveImage()
        self.task_btn_clear()
        self.task_type = None
        if self.cur < self.total:
            self.cur += 1
            self.new_picture()
        else:
            self.cur = 1
            self.new_picture()

    def gotoImage(self):
        self.saveImage()
        idx = int(self.idxEntry.get())
        if 1 <= idx and idx <= self.total:
            self.cur = idx
            self.new_picture()


if __name__ == '__main__':
    root = Tk()
    tool = LabelTool(root)
    root.resizable(width =  True, height = True)
    root.mainloop()