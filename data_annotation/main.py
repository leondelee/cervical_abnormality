#!/usr/bin/env python
# coding: utf-8

# In[27]:


# -------------------------------------------------------------------------------
# Name:        Object bounding box label tool
# Purpose:     Label object bboxes for ImageNet Detection data
#
# -------------------------------------------------------------------------------
from __future__ import division
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
import os
import glob
import random
import json
from functools import partial


# colors for the bboxes
COLORS = ['green', 'yellow', 'red', 'black', 'white', 'BurlyWood', 'LimeGreen']
CLS_NAME = ["正常", "轻微异常", "严重异常", "转化区", "ECC低级别", "ECC高级别", "癌变"]
# image sizes for the examples
SIZE = (1400, 960)
RESIZE_RATE = 0.8
NORMAL = 1
SLIGHT_ANORMAL = 2
SEVERE_ANORMAL = 3
PRE_SIZE = 100
# available image formats
FORMATS = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]


class LabelTool():
    def __init__(self, master):
        # set up the main frame
        self.width = SIZE[0]
        self.height = SIZE[1]
        self.img_start_x = 0
        self.img_start_y = 0
        self.parent = master
        self.parent.title("LabelTool")
        self.parent.geometry("{}x{}".format(self.width, self.height))
        self.frame = Frame(self.parent)
        self.frame.pack(fill=BOTH, expand=1)
        self.parent.resizable(width=FALSE, height=FALSE)

        # initialize global state
        self.imageDir = ''
        self.imageList = []
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
        self.classes = []  # classes information
        self.class_btns = []  # class btns
        self.class_points = []  # class points on the picture
        self.current_class = None
        
        # reference to class
        self.pre = []  # classes information
        self.pre_btns = []  # class btns
        self.pre_points = []  # class points on the picture
        self.current_class = None
        
        # task info
        self.MODE_TRANS = 0
        self.MODE_DEGREE = 1
        self.MODE_PRE = 2
        self.task_type = None
        self.task_btns = []

        # ----------------- GUI stuff ---------------------
        # dir entry & load
        self.label = Label(self.frame, text="Image Dir:")
        self.label.grid(row=0, column=0, sticky=E)
        self.cbx = ttk.Combobox(self.frame, width=12, height=8, textvariable=int)
        self.cbx.grid(row=0, column=1, sticky=W + E)
        self.pwd = os.getcwd()
        self.mylist = os.listdir(self.pwd + '.\Images')
        self.cbx["values"] = self.mylist
        self.ldBtn = Button(self.frame, text="Load", command=self.loadDir)
        self.ldBtn.grid(row=0, column=2, sticky=W + E)
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
        self.parent.bind("a", self.prevImage)  # press 'a' to go backforward
        self.parent.bind("d", self.nextImage)  # press 'd' to go forward
        self.mainPanel.grid(row=1, column=1, rowspan=4, sticky=W + N)

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
        self.ctrPanel.grid(row=5, column=1, columnspan=2, sticky=W + E)
        self.prevBtn = Button(self.ctrPanel, text='<< 上一张', width=10, command=self.prevImage)
        self.prevBtn.pack(side=LEFT, padx=5, pady=3)
        self.nextBtn = Button(self.ctrPanel, text='下一张 >>', width=10, command=self.nextImage)
        self.nextBtn.pack(side=LEFT, padx=5, pady=3)
        self.progLabel = Label(self.ctrPanel, text="Progress:     /    ")
        self.progLabel.pack(side=LEFT, padx=5)
        self.tmpLabel = Label(self.ctrPanel, text="前往图片 No.")
        self.tmpLabel.pack(side=LEFT, padx=5)
        self.idxEntry = Entry(self.ctrPanel, width=5)
        self.idxEntry.pack(side=LEFT)
        self.goBtn = Button(self.ctrPanel, text='Go', command=self.gotoImage)
        self.goBtn.pack(side=LEFT)
        self.saveBtn = Button(self.ctrPanel, text='保存修改', width=10, command=self.saveCurrent)
        self.saveBtn.pack(side=LEFT)
        self.deleteBtn = Button(self.ctrPanel, text='删除\撤销删除', width=10, command=self.DeleteCurrent)
        self.deleteBtn.pack(side=LEFT)
        self.eccBtn = Button(self.ctrPanel, text='ECC', width=10, command=self.TypeECC)
        self.eccBtn.pack(side=LEFT)
        self.btnOrigColor = self.saveBtn.cget("background")

        # change label
        self.changelabelBtn = Button(self.ctrPanel, text='修改类别', width=10, command=self.change_label)
        self.changelabelBtn.pack(side=LEFT)

        self.disp = Label(self.ctrPanel, text='')
        self.disp.pack(side=LEFT)

        self.frame.columnconfigure(1, weight=1)
        self.frame.rowconfigure(4, weight=1)
        self.json_file_name = ""

    def change_label(self):
        sel = self.listbox.curselection()
        if len(sel) != 1:
            return
        idx = int(sel[0])
        tmp = self.json_data["class"][idx]
        tmpId = self.mainPanel.create_rectangle(tmp[0], tmp[1], tmp[2], tmp[3], width=2,
                                                outline='DarkGoldenrod')
        x = tmp[2]
        y = tmp[3]

        def change(new_label):
            self.json_data["class"][idx][-1] = new_label
            for btn in self.class_btns:
                btn.destroy()
            self.class_btns = []
            self.saveImage(False)
            self.task_class()

        for new_label, clss in enumerate(CLS_NAME):
            btn = Button(self.frame, text=CLS_NAME[new_label], command=partial(change, new_label))
            btn.place(x=x + 5, y=y + 30 * new_label, height=30)
            self.class_btns.append(btn)

    def loadDir(self, dbg=False):
        if not dbg:
            s = self.cbx.get()
            self.parent.focus()
            self.category = int(s)
        else:
            s = r'D:\workspace\python\labelGUI'
        # get image list
        self.imageDir = os.path.join(r'.\Images', '%03d' % self.category)
        self.labelDir = os.path.join(r'.\Labels', '%03d' % self.category)
        if not os.path.exists(self.labelDir):
            os.mkdir(self.labelDir)
        self.label_dir = "./Labels/%03d/" % self.category
        if not os.path.exists(self.label_dir):
            os.mkdir(self.label_dir)
        # load Images
        self.imageList = []
        for format in FORMATS:
            images = glob.glob(os.path.join(self.imageDir, format))
            if len(images):
                self.imageList.extend(images)
        if len(self.imageList) == 0:
            print('No Images found in the specified dir!')
            return
        self.imageList.sort(key=lambda x: int(x.split("\\")[-1].split(".")[0]))

        # default to the 1st image in the collection
        self.cur = 1
        self.total = len(self.imageList)

        # load example bboxes
        self.egDir = os.path.join(r'./Images', '%03d' % (self.category))
        if not os.path.exists(self.egDir):
            return
        filelist = glob.glob(os.path.join(self.egDir, '*.jpg'))
        self.tmp = []
        self.egList = []
        random.shuffle(filelist)

        self.new_picture()
        print('%d Images loaded from %s' % (self.total, s))

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

        self.PreBtn = Button(self.operatePanel, text='预处理', command=self.task_Pre)
        self.PreBtn.grid(row=1, column=1, sticky=W)
        self.operatePanel.grid(row=2, column=2, sticky=W)
        self.degreeBtn = Button(self.operatePanel, text='标注类别', command=self.task_class)
        self.degreeBtn.grid(row=1, column=3, sticky=W)
    
    def task_Pre(self):
        self.saveCurrent()
        self.task_type = self.MODE_PRE
        self.lb1 = Label(self.frame, text='裁剪区')
        self.lb1.grid(row=3, column=2, sticky=W + N)
        self.task_btns.append(self.lb1)
        self.listbox = Listbox(self.frame, width=22, height=12)
        self.listbox.grid(row=4, column=2, sticky=N)
        self.task_btns.append(self.listbox)
        self.btnDel = Button(self.frame, text='删除', command=self.delPre)
        self.btnDel.grid(row=5, column=2, sticky=W + E + N)
        self.task_btns.append(self.btnDel)
        self.btnClear = Button(self.frame, text='清空所有', command=self.clearPre)
        self.btnClear.grid(row=6, column=2, sticky=W + E + N)
        self.task_btns.append(self.btnClear)
        self.loadImage()
        ## clear panel
        for idx in range(len(self.bboxIdList)):
            self.mainPanel.delete(self.bboxIdList[idx])
        self.listbox.delete(0, self.listbox.size())
        for coor in self.json_data["pre"]:
            [current_x, current_y] = coor
            current_x = current_x*self.tkimg.width()
            current_y = current_y*self.tkimg.height()
            self.listbox.insert(END, "{}, {}".format(current_x, current_y))
            x1, y1 = (current_x - PRE_SIZE), (current_y - PRE_SIZE)
            x2, y2 = (current_x + PRE_SIZE), (current_y + PRE_SIZE)
            tmpId = self.mainPanel.create_rectangle(x1, y1, x2,y2, width=2,
                                                     outline='black')
            self.pre_points.append(tmpId)


    def task_class(self):
        self.saveCurrent()
        self.task_type = self.MODE_DEGREE
        self.changeButtonColor()
        self.lb1 = Label(self.frame, text='已标注的转化区')
        self.lb1.grid(row=3, column=2, sticky=W + N)
        self.task_btns.append(self.lb1)
        self.listbox = Listbox(self.frame, width=22, height=12)
        self.listbox.grid(row=4, column=2, sticky=N)
        self.task_btns.append(self.listbox)
        self.btnDel = Button(self.frame, text='删除', command=self.delBBox)
        self.btnDel.grid(row=5, column=2, sticky=W + E + N)
        self.task_btns.append(self.btnDel)
        self.btnClear = Button(self.frame, text='清空所有', command=self.clearBBox)
        self.btnClear.grid(row=6, column=2, sticky=W + E + N)
        self.task_btns.append(self.btnClear)
        self.loadImage()
        if self.json_data["available"]==0:
            self.deleteBtn['text']='撤销删除'
        else :
            self.deleteBtn['text']='删除'
        if self.json_data["ECC"]==1:
            self.eccBtn.configure(bg='red')
        else :
            self.eccBtn.configure(bg=self.btnOrigColor)
        self.eccBtn.configure(bg='red')


        # clear panel
        for idx in range(len(self.class_points)):
            self.mainPanel.delete(self.class_points[idx])
        self.listbox.delete(0, self.listbox.size())
        for tmp in self.json_data["class"]:
            tmpId = self.mainPanel.create_rectangle(tmp[0], tmp[1], tmp[2], tmp[3], width=2,
                                                    outline=COLORS[int(tmp[4])])
            self.bboxIdList.append(tmpId)
            self.listbox.insert(END,
                                '(%d, %d) -> (%d, %d), %s' % (tmp[0], tmp[1], tmp[2], tmp[3], CLS_NAME[int(tmp[4])]))

    def loadImage(self):
        # load image
        imagepath = self.imageList[self.cur - 1]
        self.img = Image.open(imagepath)
        new_size = (int(self.img.size[0] * RESIZE_RATE), int(self.img.size[1] * RESIZE_RATE))
        self.img = self.img.resize(new_size, Image.ANTIALIAS)
        print(self.img.size)
        self.tkimg = ImageTk.PhotoImage(self.img)
        self.mainPanel.config(width=max(self.tkimg.width(), 400), height=max(self.tkimg.height(), 400))
        self.mainPanel.create_image(self.img_start_x, self.img_start_y, image=self.tkimg, anchor=NW)
        self.progLabel.config(text="%04d/%04d" % (self.cur, self.total))
        self.imagename = os.path.split(imagepath)[-1].split('.')[0]
        self.json_file_name = os.path.join(self.label_dir, self.imagename + '.json')
        if not os.path.exists(self.json_file_name):
            with open(self.json_file_name, 'w+') as file:
                file.write(json.dumps({"bbox": [], "class": [],"pre":[],"available":1,"ECC":0}))
                file.close()
        with open(self.json_file_name, 'r+') as file:
            self.json_data = json.loads(file.read())
            if not "available" in str(self.json_data):
                self.json_data["available"]=1
            if not "ECC" in str(self.json_data):
                self.json_data["ECC"]=0
            if self.json_data["ECC"] == 1:
                self.eccBtn.configure(bg='red')
            else:
                self.eccBtn.configure(bg=self.btnOrigColor)
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
            self.clearPre()

    def saveCurrent(self):
        if self.json_file_name != '':
            import json
            with open(self.json_file_name, 'w+') as f:
                f.write(json.dumps(self.json_data))
                f.close()
            # print('Image No. %d saved' % (self.cur))
            self.deleteJson()
            
    def  DeleteCurrent(self):
        if self.json_data["available"]==0:
            self.deleteBtn['text']='删除'
            self.json_data["available"]=1
        else :
            self.deleteBtn['text']='撤销删除'
            self.json_data["available"]=0


    def  TypeECC(self):
        if self.json_data["ECC"]==0:
            self.json_data["ECC"]=1
            self.eccBtn.configure(bg='red')
        else :
            self.eccBtn.configure(bg=self.btnOrigColor)
            self.json_data["ECC"]=0
        
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
            # self.transBtn.configure(bg='red')
        elif self.task_type == self.MODE_DEGREE:
            # self.transBtn.configure(bg=self.btnOrigColor)
            self.degreeBtn.configure(bg='red')
        else:
            # self.transBtn.configure(bg=self.btnOrigColor)
            self.degreeBtn.configure(bg=self.btnOrigColor)

    def mouseClick(self, event):
        if self.task_type == self.MODE_DEGREE:
            if self.STATE['click'] == 0:
                self.STATE['x'], self.STATE['y'] = event.x, event.y
            else:
                x1, x2 = min(self.STATE['x'], event.x), max(self.STATE['x'], event.x)
                y1, y2 = min(self.STATE['y'], event.y), max(self.STATE['y'], event.y)

                def on_click_label(idx):
                    self.listbox.insert(END, '({}, {}) -> ({}, {}), {}'.format(x1, y1, x2, y2, CLS_NAME[idx]))
                    self.json_data["class"].append([x1, y1, x2, y2, idx])
                    self.task_class()
                    for btn in self.class_btns:
                        btn.destroy()
                    self.class_btns = []

                for idx, clss in enumerate(CLS_NAME):
                    btn = Button(self.frame, text=CLS_NAME[idx], command=partial(on_click_label, idx))
                    btn.place(x=x2 + 5, y=y2 + 30 * idx, height=30)
                    self.class_btns.append(btn)

                self.saveImage(False)
            #                 self.bboxIdList.append(self.bboxId)
            #                 self.bboxId = None
            #                 self.listbox.insert(END, '(%d, %d) -> (%d, %d)' %(x1, y1, x2, y2))
            #                 self.json_data["bbox"].append([x1, y1, x2, y2])
            self.STATE['click'] = 1 - self.STATE['click']

        elif self.task_type == self.MODE_PRE:
            current_x, current_y = event.x, event.y
            self.json_data["pre"].append([current_x/self.tkimg.width(), current_y/self.tkimg.height()])
            self.listbox.insert(END, '%d, %d' %(current_x, current_y))
            x1, y1 = (current_x - PRE_SIZE), (current_y - PRE_SIZE)
            x2, y2 = (current_x + PRE_SIZE), (current_y + PRE_SIZE)
            tmpId = self.mainPanel.create_rectangle(x1, y1, x2,y2, width=2,outline='black')
            self.pre_points.append(tmpId)
            
        self.saveImage(False)

    def mouseMove(self, event):
        self.disp.config(text='x: %d, y: %d' % (event.x, event.y))
        if self.tkimg:
            if self.hl:
                self.mainPanel.delete(self.hl)
            self.hl = self.mainPanel.create_line(0, event.y, self.tkimg.width(), event.y, width=2)
            if self.vl:
                self.mainPanel.delete(self.vl)
            self.vl = self.mainPanel.create_line(event.x, 0, event.x, self.tkimg.height(), width=2)
        if 1 == self.STATE['click']:
            if self.bboxId:
                self.mainPanel.delete(self.bboxId)
            self.bboxId = self.mainPanel.create_rectangle(self.STATE['x'], self.STATE['y'], event.x, event.y, width=2,
                                                          outline='black')

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
        print(idx)
        self.mainPanel.delete(self.bboxIdList[idx])
        self.bboxIdList.pop(idx)
        # self.bboxList.pop(idx)
        if self.listbox.size():
            self.listbox.delete(idx)
        self.json_data["class"].pop(idx)
        self.task_class()

    def clearBBox(self):
        for idx in range(len(self.bboxIdList)):
            self.mainPanel.delete(self.bboxIdList[idx])
        self.listbox.delete(0, self.listbox.size())
        self.json_data["class"] = []
        self.task_class()

    def clearClass(self):
        for idx in range(len(self.class_points)):
            self.mainPanel.delete(self.class_points[idx])
        self.listbox.delete(0, self.listbox.size())
        self.json_data["class"] = []
        self.task_class()

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
        self.task_class()
        
    def clearPre(self):
        for idx in range(len(self.pre_points)):
            self.mainPanel.delete(self.pre_points[idx])
        self.listbox.delete(0, self.listbox.size())
        self.json_data["pre"] = []
        self.task_Pre()
        
    def delPre(self):
        sel = self.listbox.curselection()
        if len(sel) != 1:
            return
        idx = int(sel[0])
        self.mainPanel.delete(self.pre_points[idx])
        self.pre_points.pop(idx)
        if self.listbox.size():
            self.listbox.delete(idx)
        self.json_data["pre"].pop(idx)
        self.task_Pre()

    def prevImage(self, event=None):
        self.saveImage(False)
        if self.cur > 1:
            self.cur -= 1
            self.new_picture()
        else:
            self.cur = self.total
            self.new_picture()

    def nextImage(self, event=None):
        self.saveImage(False)
        self.task_btn_clear()
        self.task_type = None
        if self.cur < self.total:
            self.cur += 1
            self.new_picture()
        else:
            self.cur = 1
            self.new_picture()

    def gotoImage(self):
        self.saveImage(False)
        idx = int(self.idxEntry.get())
        if 1 <= idx and idx <= self.total:
            self.cur = idx
            self.new_picture()



if __name__ == '__main__':
    root = Tk()
    root.state("zoomed")
    tool = LabelTool(root)
    root.resizable(width=True, height=True)
    root.mainloop()




