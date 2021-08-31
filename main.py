'''
from tkinter import *
box=Tk()
box.geometry("500x500")
https://www.youtube.com/watch?v=bt4-FwVe1Fk&list=RDHhWum37Mg8o&index=22
username_label = Label(box,text="username").place(x=60,y=60)
password_label = Label(box,text="password").place(x=60,y=100)
username_text= Entry(box,width=20).place(x=140,y=60)
password_text= Entry(box,width=20).place(x=140,y=100)
submit_button=Button(box,text="submit").place(x=110,y=150)
box.mainloop()
'''
#
# # pip install pafy
# #pip install youtube_dl
# import pafy
#
# from tkinter import *
#
# def getMetaData(video):
#     print("Video Details are ---")
#     print("video title : ",video.title)  #print title
#     # print view count
#     # print(f"Total views : {video.viewcount}| video lenght : {video.lengh} secounds")
#     print("channel name : ",video.author) #print author
#
# def download_As_video(video):
#     getMetaData(video) #get video details
#     best= video.getbest()
#     print(f"Video Resolution:{best.resolution}\n video extension:{best.extension}")
#     best.download()#download the video
#     print("Video is downloaded...")
# #
# # # if __name__ == "__main__":
# # url = input("Enter video url : ")
# # #create instance
# # video = pafy.new(url)
# # download_As_video(video)
# def fn():
#     url=textbox.get()
#     video = pafy.new(url)
#     download_As_video(video)
# #     print(textbox.get())
#
# frame=Tk()
#
# frame.geometry("300x300")
# textbox = Entry(frame,width="20",textvariable="textbox")
# textbox.place(x=30,y=30)
#
# btn = Button(frame,text="submit",command=fn).place(x=50,y=70)
# frame.mainloop()\

# pip install pandas
# pip install sklearn
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

def random_foerst():
    candidates = {'pc': [780,750,690,710,680,730,690,720,740,690,610,690,710,680,770,610,580,650,540,590,620,600,550,550,570,670,660,580,650,660,640,620,660,660,680,650,670,580,590,690],
                  'load': [4,3.9,3.3,3.7,3.9,3.7,2.3,3.3,3.3,1.7,2.7,3.7,3.7,3.3,3.3,3,2.7,3.7,2.7,2.3,3.3,2,2.3,2.7,3,3.3,3.7,2.3,3.7,3.3,3,2.7,4,3.3,3.3,2.3,2.7,3.3,1.7,3.7],
                  'ipc': [3,4,3,5,4,6,1,4,5,1,3,5,6,4,3,1,4,6,2,3,2,1,4,1,2,6,4,2,6,5,1,2,4,6,5,1,2,1,4,5],
                  'vehicle':[6,4,3,1,4,6,3,4,3,5,4,6,1,4,5,1,3,5,6,4,3,1,4,6,2,3,2,1,4,1,2,6,4,2,6,5,1,2,4,6],
                  'speed':[70,55,40,91,30,71,40,80,30,20,20,81,91,40,70,35,23,70,40,30,30,20,30,20,45,100,80,40,71,61,40,30,51,81,71,20,30,20,40,60],
                  'accident': [1,1,0,1,0,1,0,1,1,0,0,1,1,0,1,0,0,1,0,0,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,0,0,1]
                  }

    df = pd.DataFrame(candidates,columns= ['pc', 'load','ipc','vehicle','speed','accident'])
    print(df)
    x = df.iloc[:, 0:5].values
    y = df.iloc[:, 5].values
    print(x)
    print(y)

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.40)
    sc = StandardScaler()
    X_train = sc.fit_transform(xtrain)
    X_test = sc.transform(xtest)
    regressor = RandomForestRegressor(n_estimators=200, random_state=0)
    regressor.fit(X_train, ytrain)
    y_pred = regressor.predict(X_test)

    print('Mean Absolute Error:', metrics.mean_absolute_error(ytest, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(ytest, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(ytest, y_pred)))

random_foerst()