from tkinter import *
from tkinter import messagebox
from PIL import ImageTk,Image
from tkinter.filedialog import askopenfilename
import cv2 as cv
import numpy as np
a=Tk()
a.title("Pneumonia_detect")
a.geometry("1200x600")


def prediction(var,e_var1,e_var2):

    name=e_var1.get()
    print(name)
    age=e_var2.get()
    print(age)
    d=var.get()
    print(d)
    
    if name=='' or age=='' or d=='Select':
        message.set("Fill the empty field!!!")
    else:

        list_box.insert(1,"Loading Image")
        list_box.insert(2,"")
        list_box.insert(3,"Image Preprocessing")
        list_box.insert(4,"")
        list_box.insert(5,"Loading Model")
        list_box.insert(6,"")
        list_box.insert(7,"Prediction")

        import numpy as np
        import cv2
        from PIL import Image
        from tensorflow.keras.models import load_model

        loaded_model = load_model("Project_Saved_Models/Pneumonia_detect_92acc.h5")

        # path_test = "D:/DENNY/Implementable_OR_not/pneumonia_chest_xray/Test/Test1/"
        width = 224
        height = 224
        data = []
        # image = cv2.imread(path_test + "/yes_1.jpeg")
        image = cv2.imread(path)
        image_from_array = Image.fromarray(image, 'RGB')
        size_image = image_from_array.resize((height, width))
        data.append(np.array(size_image))
        x_test = np.array(data)
        # print(">>>>>>>>>>>>",x_test)

        x_test=x_test/255

        my_pred = loaded_model.predict(x_test)
        my_pred = my_pred[0]
        my_pred = my_pred[0]

        if my_pred <= 0.5:
            print("RESULT: Pneumonia NOT Detected")
            a="Pneumonia NOT Detected"
            out_label.config(text=a)
        elif my_pred > 0.5:
            
            print("RESULT: Pneumonia Detected")
            import numpy as np
            import cv2
            from PIL import Image
            from tensorflow.keras.models import load_model

            loaded_model = load_model(
                "Project_Saved_Models/Pneumonia_cause_detect_95acc.h5")

            # path_test = "D:/DENNY/Implementable_OR_not/pneumonia_chest_xray/Test/Test2/"
            width = 224
            height = 224
            data = []
            # image = cv2.imread(path_test + "/b9.jpeg")
            image = cv2.imread(path)
            image_from_array = Image.fromarray(image, 'RGB')
            size_image = image_from_array.resize((height, width))
            data.append(np.array(size_image))
            x_test = np.array(data)
            # print(">>>>>>>>>>>>",x_test)

            x_test = x_test/255

            my_pred = loaded_model.predict(x_test)
            my_pred = my_pred[0]
            my_pred = my_pred[0]
            ss=my_pred*100
            ss=str(ss)
            pp=ss
            if my_pred > 0.8:
                print("RESULT: Pneumonia Detected : VIRUS")
                a="Pneumonia Detected : VIRUS"
            elif my_pred <= 0.8:
                print("RESULT: Pneumonia Detected : BACTERIA")
                a="Pneumonia Detected : BACTERIA"
            out_label.config(text=a)
            out_label1.config(text=" Percentage :" +ss)
            age=int(age)
            pp=float(pp)
            if d=='No':
                if(age<=10) and (pp>=50):
                    mm="High"
                elif(age<=10) and (20<=pp<=50):
                    mm="Medium"
                elif(age<=10) and (pp<=20):
                    mm="Low"
                elif(age>=45) and (pp>=60):
                    mm="High"
                elif(age>=45) and (20<=pp<=60):
                    mm="Medium"
                elif(age>=45) and (pp<=20):
                    mm="Low"
                elif(10<age<45) and (pp<=20):
                    mm="Low"
                elif(10<age<45) and(pp>=70):
                    mm="High"
                else:
                    mm="Medium"

            elif d=='Yes':
                if(age<=10) and (pp>=30):
                    mm="High"
                elif(age<=10) and (pp<30):
                    mm="Medium"
                elif(age>=45) and (pp>=40):
                    mm="High"
                elif(age>=45) and (pp<=40):
                    mm="Medium"
                elif(10<age<45) and(pp>=50):
                    mm="High"
                else:
                    mm="Medium"

            out_label2.config(text="Severity :" +mm)

   

def hide(var):
    if var=='Yes':
        entry3.pack(pady=125)
    if var=='No':
        entry3.pack_forget()

def Check():
    global f
    f.pack_forget()

    f=Frame(a,bg="white")
    f.pack(side="top",fill="both",expand=True)


    
    global f1
    f1=Frame(f,bg="Lavender")
    f1.place(x=0,y=0,width=560,height=610)
    f1.config()
                   
    input_label=Label(f1,text="INPUT",font="arial 16",bg="lavender")
    input_label.pack(padx=0,pady=20)

    name_label=Label(f1,text="Patient Name :",font="arial 12",bg="lavender")
    name_label.place(x=50,y=80)
    age_label=Label(f1,text="Patient Age :",font="arial 12",bg="lavender")
    age_label.place(x=50,y=120)
    d_label=Label(f1,text="Any Respiratory Disease :",font="arial 12",bg="lavender")
    d_label.place(x=50,y=160)

    global var
    var = StringVar()
    var.set("Select")
    options = ["Yes", "No",]
    op1 = OptionMenu(f1, var, *options, command=hide)
    op1.place(x=250, y=160)

    global message
    e_var1=StringVar()
    e_var2=StringVar()
    message=StringVar()
    hidden_text=StringVar()
    hidden_text.set("Disease name")

    entry1 = Entry(f1, textvariable=e_var1,bd=2 ,width=25)
    entry1.place(x=250, y=83)
    entry2 = Entry(f1, textvariable=e_var2,bd=2, width=25)
    entry2.place(x=250, y=123)
    global entry3
    entry3=Entry(f1,textvariable=hidden_text,bd=2,width=25)

    msg_label = Label(f1, text="", textvariable=message,
                      bg='lavender').place(x=250, y=220)



    upload_pic_button=Button(f1,text="Upload Picture",command=Upload,bg="pink")
    upload_pic_button.place(x=240,y=250)
    global label
    label=Label(f1,bg="Lavender")

    
    predict_button=Button(f1,text="Predict",command=lambda: prediction(var,e_var1,e_var2),bg="deepskyblue")
    predict_button.pack(side="bottom",pady=40)

    



    global f2
    f2=Frame(f,bg="aquamarine")
    f2.place(x=800,y=0,width=400,height=690)
    f2.config(pady=20)
    
    result_label=Label(f2,text="RESULT",font="arial 16",bg="aquamarine")
    result_label.pack(padx=0,pady=0)

    global out_label
    out_label=Label(f2,text="",bg="aquamarine",font="arial 16")
    out_label.pack(pady=90)
    global out_label1
    out_label1=Label(f2,text="",bg="aquamarine",font="arial 16")
    out_label1.pack()
    global out_label2
    out_label2=Label(f2,text="",bg="aquamarine",font="arial 16")
    out_label2.pack()
    
 




    f3=Frame(f,bg="Salmon")
    f3.place(x=560,y=0,width=240,height=690)
    f3.config()

    name_label=Label(f3,text="Process",font="arial 14",bg="Salmon")
    name_label.pack(pady=20)

    global list_box
    list_box=Listbox(f3,height=12,width=31)
    list_box.pack()


def Upload():
    global path
    label.config(image='')
    list_box.delete(0,END)
    out_label.config(text='')
    path=askopenfilename(title='Open a file',
                         initialdir='Test/Test1',
                         filetypes=(("JPEG","*.jpeg"),("PNG","*.png"),("JPG","*.jpg")))
    print("<<<<<<<<<<<<<",path)
    image=Image.open(path)
    global imagename
    imagename=ImageTk.PhotoImage(image.resize((224,224)))
    label.config(image=imagename)
    label.image=imagename
    # label.pack()
    label.place(x=170,y=290)
                  


def Home():
    global f
    f.pack_forget()
    
    f=Frame(a,bg="cornflower blue")
    f.pack(side="top",fill="both",expand=True)

    home_label=Label(f,text="Pneumonia Detector",font="arial 35",bg="cornflower blue")
    home_label.place(x=390,y=250)




f=Frame(a,bg="cornflower blue")
f.pack(side="top",fill="both",expand=True)

home_label=Label(f,text="Pneumonia Detector",font="arial 35",bg="cornflower blue")
home_label.place(x=390,y=250)

m=Menu(a)
m.add_command(label="Home",command=Home)
checkmenu=Menu(m)
m.add_command(label="Check",command=Check)
a.config(menu=m)




a.mainloop()
