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

# colors for the bboxes
COLORS = ['green', 'yellow', 'red', 'black']
CLS_NAME = ["正常", "轻微异常", "严重异常", "转化区"]
# image sizes for the examples
SIZE = (1400, 960)
RESIZE_RATE = 0.8
NORMAL = 1
SLIGHT_ANORMAL = 2
SEVERE_ANORMAL = 3
PRE_SIZE = 7


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
        self.btnOrigColor = self.saveBtn.cget("background")

        self.disp = Label(self.ctrPanel, text='')
        self.disp.pack(side=LEFT)

        self.frame.columnconfigure(1, weight=1)
        self.frame.rowconfigure(4, weight=1)
        self.json_file_name = ""

    def loadDir(self, dbg=False):
        if not dbg:
            s = self.cbx.get()
            self.parent.focus()
            self.category = int(s)
        else:
            s = r'D:\workspace\python\labelGUI'
        # get image list
        self.imageDir = os.path.join(r'.\Images', '%03d' % (self.category))
        self.labelDir = os.path.join(r'.\Labels', '%03d' % self.category)
        if not os.path.exists(self.labelDir):
            os.mkdir(self.labelDir)
        self.label_dir = "./Labels/%03d/" % self.category
        if not os.path.exists(self.label_dir):
            os.mkdir(self.label_dir)
        self.imageList = glob.glob(os.path.join(self.imageDir, '*.jpg'))

        if len(self.imageList) == 0:
            print('No .jpg images found in the specified dir!')
            return

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
        # for (i, f) in enumerate(filelist):
        #     # if i == 3:
        #     #     break
        #     im = Image.open(f)
        #     r = min(SIZE[0] / im.size[0], SIZE[1] / im.size[1])
        #     new_size = (int(r * im.size[0]), int(r * im.size[1]))
        #     self.tmp.append(im.resize(new_size, Image.ANTIALIAS))
        #     self.egList.append(ImageTk.PhotoImage(self.tmp[-1]))
        #     # self.egLabels[i].config(image = self.egList[-1], width = SIZE[0], height = SIZE[1])

        self.new_picture()
        print('%d images loaded from %s' % (self.total, s))

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
        self.changeButtonColor()
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
        self.tkimg = ImageTk.PhotoImage(self.img)
        self.mainPanel.config(width=max(self.tkimg.width(), 400), height=max(self.tkimg.height(), 400))
        self.mainPanel.create_image(self.img_start_x, self.img_start_y, image=self.tkimg, anchor=NW)
        self.progLabel.config(text="%04d/%04d" % (self.cur, self.total))
        self.imagename = os.path.split(imagepath)[-1].split('.')[0]
        self.json_file_name = os.path.join(self.label_dir, self.imagename + '.json')
        if not os.path.exists(self.json_file_name):
            with open(self.json_file_name, 'w+') as file:
                file.write(json.dumps({"bbox": [], "class": [],"pre":[]}))
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
            self.clearPre()

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
            # self.transBtn.configure(bg='red')
        elif self.task_type == self.MODE_DEGREE:
            # self.transBtn.configure(bg=self.btnOrigColor)
            self.degreeBtn.configure(bg='red')
        elif self.task_type == self.MODE_PRE:
            self.PreBtn.configure(bg=self.btnOrigColor)   
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

                def on_click_label1():
                    self.listbox.insert(END, '({}, {}) -> ({}, {}), {}'.format(x1, y1, x2, y2, CLS_NAME[0]))
                    self.json_data["class"].append([x1, y1, x2, y2, 0])
                    self.task_class()
                    for btn in self.class_btns:
                        btn.destroy()
                    self.class_btns = []

                def on_click_label2():
                    self.listbox.insert(END, '({}, {}) -> ({}, {}), {}'.format(x1, y1, x2, y2, CLS_NAME[1]))
                    self.json_data["class"].append([x1, y1, x2, y2, 1])
                    self.task_class()
                    for btn in self.class_btns:
                        btn.destroy()
                    self.class_btns = []

                def on_click_label3():
                    self.listbox.insert(END, '({}, {}) -> ({}, {}), {}'.format(x1, y1, x2, y2, CLS_NAME[2]))
                    self.json_data["class"].append([x1, y1, x2, y2, 2])
                    self.task_class()
                    for btn in self.class_btns:
                        btn.destroy()
                    self.class_btns = []

                def on_click_label4():
                    self.listbox.insert(END, '({}, {}) -> ({}, {}), {}'.format(x1, y1, x2, y2, CLS_NAME[3]))
                    self.json_data["class"].append([x1, y1, x2, y2, 3])
                    self.task_class()
                    for btn in self.class_btns:
                        btn.destroy()
                    self.class_btns = []

                btn1 = Button(self.frame, text=CLS_NAME[0], command=on_click_label1)
                btn1.place(x=x2 + 5, y=y2, height=30)
                self.class_btns.append(btn1)
                btn2 = Button(self.frame, text=CLS_NAME[1], command=on_click_label2)
                btn2.place(x=x2 + 5, y=y2 + 30, height=30)
                self.class_btns.append(btn2)
                btn3 = Button(self.frame, text=CLS_NAME[2], command=on_click_label3)
                btn3.place(x=x2 + 5, y=y2 + 30 * 2, height=30)
                self.class_btns.append(btn3)
                btn4 = Button(self.frame, text=CLS_NAME[3], command=on_click_label4)
                btn4.place(x=x2 + 5, y=y2 + 30 * 3, height=30)
                self.class_btns.append(btn4)
                self.saveImage(False)
            #                 self.bboxIdList.append(self.bboxId)
            #                 self.bboxId = None
            #                 self.listbox.insert(END, '(%d, %d) -> (%d, %d)' %(x1, y1, x2, y2))
            #                 self.json_data["bbox"].append([x1, y1, x2, y2])
            self.STATE['click'] = 1 - self.STATE['click']
            
            
        elif self.task_type == self.MODE_PRE:
            current_x, current_y = event.x, event.y
            self.json_data["pre"].append([current_x, current_y])
            self.listbox.insert(END, '%d, %d' %(current_x, current_y))
            x1, y1 = (current_x - PRE_SIZE), (current_y - PRE_SIZE)
            x2, y2 = (current_x + PRE_SIZE), (current_y + PRE_SIZE)
            tmpId = self.mainPanel.create_rectangle(x1, y1, x2,y2, width=2,
                                                     outline='black')
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




