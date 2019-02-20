## Purpose
Data annotation
## Usage
You need python3 installed on your PC.

Choose a proper folder and type the following command in your terminal:

    git clone https://github.com/leondelee/cervical_abnormality
    cd cervical_abnormality/data_annotation
    pip install -r requirements.txt

Store your images under the folder "/Images/001". Make sure that images are of ".jpg" type.

Then run the application by:

    python main.py
   
Normally, a GUI window should show up. Type "1" in the input box, which means the pictures to be handled situate in the folder "./Images/001" and click "load". Corresponding Json files can be found in the folder "/Labels/001".

## Changing Logs
### 2019/01/22

 - fix bugs related to saving images
 - add a "保存修改" button
 - make the "标注转化区" and "标注类别" buttons' colors chageable according to current situation
 - adjust canvas's size

