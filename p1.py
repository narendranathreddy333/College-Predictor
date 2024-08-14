from tkinter import *
import numpy as np
import pandas as pd
# from gui_stuff import *
l1=['Above90','Above80','Above70','Above60','Above50','Above40','Below40','Entranceabove50','Entranceabove40','Entranceabove30','Entranceabove20','Entrancebelow20','AF_station_yelahanka','Adugodi','Agara','Agram','Air_Force_hospital','Amruthahalli','Anandnagar','Anekal','Anekalbazar','Arabic_College','Aranya_Bhavan','Ashoknagar','Attibele','Attur','Austin_Town','Avalahalli','Avani_Sringeri_mutt','Avenue_Road','BSK2nd_stage','Bsf_Campus_yelahanka','Bagalgunte','Bagalur','Balepete','Banashankari','BSK3rd_stage','Banaswadi','Bandikodigehalli','BDA_outer_ring','Bangalore_Bazaar','Bangalore_City','Bangalore_Corporation_building','Bangalore_Dist_offices_bldg','Bangalore_Fort','Bangalore_Sub_fgn_post','BSK1st_stage','Bannerghatta','Bapujinagar','Basavanagudi','Basavaraja_Market','Basaveshwaranagar','Bellandur','Bestamaranahalli','Bettahalsur','Bhattarahalli','Bidaraguppe','Bidrahalli','Bommanahalli','Brigade_Road','Byatarayanapura','Chamrajpet','Chandra_Lay_out','Chickpet','Chikkabettahalli','Chikkajala','Chikkalasandra','Chikkanahalli','Chunchanakuppe','Cubban_Road','Dasarahalli','Deepanjalinagar','Devanagundi','Devarjeevanahalli','Devasandra','Doddagubbi','Doddajala','Doddakallasandra','Doddanekkundi','Domlur','Dommasandra','Doorvaninagar','Electronics_City','Gaviopuram_Guttanahalli','Gayathrinagar','Girinagar','Goraguntepalya','Goripalya','Govindapura','Gunjur','Hsr_Layout','Hampinagar','Handenahalli','Harogadde','Hebbal_Kempapura','Hennagara','Highcourt','Hongasandra','Hoodi','Horamavu','Hosakerehalli','Hosur_Road','Hulimangala','Hulimavu','Hulsur_Bazaar','Hunasamaranahalli','Immedihalli','Indalavadi','Indiranagar','Industrial_Estate','Ittamadu_Layout','J_P_nagar','J_C_nagar','Jalahalli','Jalavayuvihar','Jayanagar','Jayanagar_West','Jayangar_East','Jeevanahalli','Jeevanbhimanagar','Jigani' ,'Jp_Nagar_iii_phase','Kacharakanahalli','Kadabagere','Kadugodi','Kalkunte','Kalyanagar','Kamagondanahalli','Kamakshipalya','Kannamangala','Kannur','Kanteeravanagar','Kathriguppe','Kenchanahalli','Kendriya_Sadan','Kendriya_Vihar','Kodigehalli','Konanakunte','Koramangala','Kothanur','krcircle','Krishnarajapuram','Kugur','Kumaraswamy_Layout','Kumbalgodu','Kundalahalli','Lalbagh_West','Lingarajapuram','MSR_nagar','Madhavan_Park','Madivala','Magadi_Road','Mahadevapura','Mahalakshmipuram_Layout','Mahatma_Gandhi_road','Mallathahalli','Malleswaram','Mandalay_Lines','Marathahalli_Colony','Marsur','Mathikere','Mavalli','Mayasandra','Medihalli','Medimallasandra','Milk_Colony','Msrit','Mundur','mysore_road','Muthanallur','Muthusandra','Naduvathi','Nagarbhavi','Nagasandra','Nagavara','Nandinilayout','NarasimharajaColony','Narayan_Pillai_street','Nayandahalli','Outer_ringroad','Padmanabhnagar','Palace_Guttahalli','Panathur','Pasmpamahakavi_Road','Peenya','R_T_nagar','Rajajinagar','Rajanakunte','Rajarajeshwarinagar','Rajbhavan','Ramachandrapuram','Ramagondanahalli','Ramamurthy_Nagar','Rameshnagar','Richmond_Town','Sadashivanagar','Samandur','Samethanahalli','Sampangiramnagar','Sarjapura','Seshadripuram','Shankarpura','Shanthinagar','srinivaspura','somanahalli','Soladevalahalli','Srirampuram','State_Bank_of_mysore_colony','Subhashnagar','Subramanyapura','Tarahunise','Tavarekere','Taverekere','Thammanayakanahalli','Tilaknagar','Tyagrajnagar','Ullalu_Upanagara','Vanakanahalli','Vartur','Vasanthnagar','Venkatarangapura','Venkateshapura','Vidhana_Soudha','Vidyanagara','Vidyaranyapura','Vijayanagar','Vikramnagar','Vimapura','Virgonagar','Visveswarapuram','Viswaneedam','Vittalnagar','Viveknagar','Vyalikaval_Extn','Whitefield','Wilson_Garden','Yadavanahalli','Yediyur','Yelachenahalli','Yelahanka','Yemalur','Yelhanka_south','Yeswanthpura','CSE','ECE','ISE','EEE','ME','CE','AE','BT','IM','SE','TC','AU','CH','MD','EI'
]

Collegename=['UVCE','SKSJT','BMSCE','Dr.AmbedkarIT','RVCE','MSRIT','DSCE','BIT','PESIT','IslamiaIT','MVJ','Sir MVIT','Ghousia','Oxford','AcharyaIT','JSS','H.K.B.K CE','APS']

l2=[]
for x in range(0,len(l1)):
    l2.append(0)

# TRAINING DATA df -------------------------------------------------------------------------------------
df=pd.read_csv("sample.csv")
print(df.head())



df.replace({'Collegename':{'UVCE':0,'SKSJT':1,'BMSCE':2,'Dr.AmbedkarIT':3,'RVCE':4,'MSRIT':5,'DSCE':6,'BIT':7,'PESIT':8,'IslamiaIT':9,'MVJ':10,'Sir MVIT':11,'Ghousia':12,'Oxford':13,'AcharyaIT':14,'JSS':15,'H.K.B.K CE':16,'APS':17}},inplace=True)
X= df[l1] #dataframe storing all the systoms in x

y = df[["Collegename"]] #similarly in y
np.ravel(y)


# TESTING DATA df -------------------------------------------------------------------------------------
tr=pd.read_csv("sample.csv")
print(df.head())



tr.replace({'Collegename':{'UVCE':0,'SKSJT':1,'BMSCE':2,'Dr.AmbedkarIT':3,'RVCE':4,'MSRIT':5,'DSCE':6,'BIT':7,'PESIT':8,'IslamiaIT':9,'MVJ':10,'Sir MVIT':11,'Ghousia':12,'Oxford':13,'AcharyaIT':14,'JSS':15,'H.K.B.K CE':16,'APS':17}},inplace=True)
X= df[l1] #dataframe storing all the systoms in x

y = df[["Collegename"]] #similarly in y
np.ravel(y)

X_test= tr[l1]
y_test = tr[["Collegename"]]
np.ravel(y_test)

def DecisionTree():

    from sklearn import tree

    clf3 = tree.DecisionTreeClassifier()   # empty model of the decision tree
    clf3 = clf3.fit(X,np.ravel(y)) #fitting the model or the data

    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred=clf3.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))
    # -----------------------------------------------------


    particulars = [PUmarks.get(),Entrancemarks.get(),Locality.get(),Branch.get()]

    for k in range(0,len(l1)):
        # print (k,)
        for z in particulars:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = clf3.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(Collegename)):
        if(predicted == a):
            h='yes'
            break


    if (h=='yes'):
        t1.delete("1.0", END)           #t1=tab consists of diesease that is predicted 
        t1.insert(END, Collegename[a])
    else:
        t1.delete("1.0", END)
        t1.insert(END, "Not Found")     #if not found



def randomforest():
    from sklearn.ensemble import RandomForestClassifier
    clf4 = RandomForestClassifier()
    clf4 = clf4.fit(X,np.ravel(y))

    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred=clf4.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))
    # -----------------------------------------------------

  
    particulars = [PUmarks.get(),Entrancemarks.get(),Locality.get(),Branch.get()]


    for k in range(0,len(l1)):
        for z in particulars:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = clf4.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        t2.delete("1.0", END)
        t2.insert(END, disease[a])
    else:
        t2.delete("1.0", END)
        t2.insert(END, "Not Found")


def NaiveBayes():
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb=gnb.fit(X,np.ravel(y))

    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred=gnb.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))
    # -----------------------------------------------------

    particulars = [PUmarks.get(),Entrancemarks.get(),Locality.get(),Branch.get()]
    for k in range(0,len(l1)):
        for z in particulars:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = gnb.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        t3.delete("1.0", END)
        t3.insert(END, disease[a])
    else:
        t3.delete("1.0", END)
        t3.insert(END, "Not Found")




root = Tk()
root.configure(background='blue')

# entry variables
PUmarks = StringVar()
PUmarks.set(None)
Entrancemarks = StringVar()
Entrancemarks.set(None)
Locality = StringVar()
Locality.set(None)
Branch = StringVar()
Branch.set(None)
Name = StringVar()

# Heading
w2 = Label(root, justify=LEFT, text="College Predictor using Machine Learning", fg="white", bg="blue")
w2.config(font=("Elephant", 30))
w2.grid(row=1, column=0, columnspan=2, padx=100) 
w2 = Label(root, justify=LEFT, text="", fg="white", bg="blue")
w2.config(font=("Aharoni", 30))
w2.grid(row=2, column=0, columnspan=2, padx=100)

# labels
NameLb = Label(root, text="Name of the Patient", fg="yellow", bg="black")
NameLb.grid(row=6, column=0, pady=15, sticky=W)


S1Lb = Label(root, text="PU marks", fg="yellow", bg="black")
S1Lb.grid(row=7, column=0, pady=10, sticky=W)

S2Lb = Label(root, text="Entrance marks", fg="yellow", bg="black")
S2Lb.grid(row=8, column=0, pady=10, sticky=W)

S3Lb = Label(root, text="Locality", fg="yellow", bg="black")
S3Lb.grid(row=9, column=0, pady=10, sticky=W)

S4Lb = Label(root, text="Branch", fg="yellow", bg="black")
S4Lb.grid(row=10, column=0, pady=10, sticky=W)



lrLb = Label(root, text="DecisionTree", fg="white", bg="red")
lrLb.grid(row=15, column=0, pady=10,sticky=W)

destreeLb = Label(root, text="RandomForest", fg="white", bg="red")
destreeLb.grid(row=17, column=0, pady=10, sticky=W)

ranfLb = Label(root, text="NaiveBayes", fg="white", bg="red")
ranfLb.grid(row=19, column=0, pady=10, sticky=W)

# entries
#OPTIONS = sorted(l1) #alphabetical sorting takes place

k1=['Above90','Above80','Above70','Above60','Above50','Above40','Below40']
k2=['Entranceabove50','Entranceabove40','Entranceabove30','Entranceabove20','Entrancebelow20']
k3=['AF_station_yelahanka','Adugodi','Agara','Agram','Air_Force_hospital','Amruthahalli','Anandnagar','Anekal','Anekalbazar','Arabic_College','Aranya_Bhavan','Ashoknagar','Attibele','Attur','Austin_Town','Avalahalli','Avani_Sringeri_mutt','Avenue_Road','BSK2nd_stage','Bsf_Campus_yelahanka','Bagalgunte','Bagalur','Balepete','Banashankari','BSK3rd_stage','Banaswadi','Bandikodigehalli','BDA_outer_ring','Bangalore_Bazaar','Bangalore_City','Bangalore_Corporation_building','Bangalore_Dist_offices_bldg','Bangalore_Fort','Bangalore_Sub_fgn_post','BSK1st_stage','Bannerghatta','Bapujinagar','Basavanagudi','Basavaraja_Market','Basaveshwaranagar','Bellandur','Bestamaranahalli','Bettahalsur','Bhattarahalli','Bidaraguppe','Bidrahalli','Bommanahalli','Brigade_Road','Byatarayanapura','Chamrajpet','Chandra_Lay_out','Chickpet','Chikkabettahalli','Chikkajala','Chikkalasandra','Chikkanahalli','Chunchanakuppe','Cubban_Road','Dasarahalli','Deepanjalinagar','Devanagundi','Devarjeevanahalli','Devasandra','Doddagubbi','Doddajala','Doddakallasandra','Doddanekkundi','Domlur','Dommasandra','Doorvaninagar','Electronics_City','Gaviopuram_Guttanahalli','Gayathrinagar','Girinagar','Goraguntepalya','Goripalya','Govindapura','Gunjur','Hsr_Layout','Hampinagar','Handenahalli','Harogadde','Hebbal_Kempapura','Hennagara','Highcourt','Hongasandra','Hoodi','Horamavu','Hosakerehalli','Hosur_Road','Hulimangala','Hulimavu','Hulsur_Bazaar','Hunasamaranahalli','Immedihalli','Indalavadi','Indiranagar','Industrial_Estate','Ittamadu_Layout','J_P_nagar','J_C_nagar','Jalahalli','Jalavayuvihar','Jayanagar','Jayanagar_West','Jayangar_East','Jeevanahalli','Jeevanbhimanagar','Jigani' ,'Jp_Nagar_iii_phase','Kacharakanahalli','Kadabagere','Kadugodi','Kalkunte','Kalyanagar','Kamagondanahalli','Kamakshipalya','Kannamangala','Kannur','Kanteeravanagar','Kathriguppe','Kenchanahalli','Kendriya_Sadan','Kendriya_Vihar','Kodigehalli','Konanakunte','Koramangala','Kothanur','krcircle','Krishnarajapuram','Kugur','Kumaraswamy_Layout','Kumbalgodu','Kundalahalli','Lalbagh_West','Lingarajapuram','MSR_nagar','Madhavan_Park','Madivala','Magadi_Road','Mahadevapura','Mahalakshmipuram_Layout','Mahatma_Gandhi_road','Mallathahalli','Malleswaram','Mandalay_Lines','Marathahalli_Colony','Marsur','Mathikere','Mavalli','Mayasandra','Medihalli','Medimallasandra','Milk_Colony','Msrit','Mundur','mysore_road','Muthanallur','Muthusandra','Naduvathi','Nagarbhavi','Nagasandra','Nagavara','Nandinilayout','NarasimharajaColony','Narayan_Pillai_street','Nayandahalli','Outer_ringroad','Padmanabhnagar','Palace_Guttahalli','Panathur','Pasmpamahakavi_Road','Peenya','R_T_nagar','Rajajinagar','Rajanakunte','Rajarajeshwarinagar','Rajbhavan','Ramachandrapuram','Ramagondanahalli','Ramamurthy_Nagar','Rameshnagar','Richmond_Town','Sadashivanagar','Samandur','Samethanahalli','Sampangiramnagar','Sarjapura','Seshadripuram','Shankarpura','Shanthinagar','srinivaspura','somanahalli','Soladevalahalli','Srirampuram','State_Bank_of_mysore_colony','Subhashnagar','Subramanyapura','Tarahunise','Tavarekere','Taverekere','Thammanayakanahalli','Tilaknagar','Tyagrajnagar','Ullalu_Upanagara','Vanakanahalli','Vartur','Vasanthnagar','Venkatarangapura','Venkateshapura','Vidhana_Soudha','Vidyanagara','Vidyaranyapura','Vijayanagar','Vikramnagar','Vimapura','Virgonagar','Visveswarapuram','Viswaneedam','Vittalnagar','Viveknagar','Vyalikaval_Extn','Whitefield','Wilson_Garden','Yadavanahalli','Yediyur','Yelachenahalli','Yelahanka','Yemalur','Yelhanka_south','Yeswanthpura']
k4=['CSE','ECE','ISE','EEE','ME','CE','AE','BT','IM','SE','TC','AU','CH','MD','EI']

S1En = OptionMenu(root, PUmarks,*k1)
S1En.grid(row=7, column=1)

S2En = OptionMenu(root, Entrancemarks,*k2)
S2En.grid(row=8, column=1)

S3En = OptionMenu(root, Locality,*k3)
S3En.grid(row=9, column=1)

S4En = OptionMenu(root, Branch,*k4)
S4En.grid(row=10, column=1)



dst = Button(root, text="DecisionTree", command=DecisionTree,bg="green",fg="yellow")
dst.grid(row=8, column=3,padx=10)

rnf = Button(root, text="Randomforest", command=randomforest,bg="green",fg="yellow")
rnf.grid(row=9, column=3,padx=10)

lr = Button(root, text="NaiveBayes", command=NaiveBayes,bg="green",fg="yellow")
lr.grid(row=10, column=3,padx=10)

#textfileds
t1 = Text(root, height=1, width=40,bg="orange",fg="black")
t1.grid(row=15, column=1, padx=10)

t2 = Text(root, height=1, width=40,bg="orange",fg="black")
t2.grid(row=17, column=1 , padx=10)

t3 = Text(root, height=1, width=40,bg="orange",fg="black")
t3.grid(row=19, column=1 , padx=10)

root.mainloop()
