# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 00:09:36 2018
@author: USER
"""
"""
Created on Tue Jul 31 20:23:36 2018

@author: USER
"""

from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QLineEdit , QPushButton ,QMessageBox 
import sys
import numpy as np 
import pandas as pd
import warnings
from scipy.stats import pearsonr 


dataFile='/Users/USER/Desktop/Thesis/Recommendation_code_kevon_15_7_18/AllCheckInfoDF_exceptTime.csv'
data=pd.read_csv(dataFile)

fileName = "/Users/USER/Desktop/Thesis/Recommendation_code_kevon_15_7_18/avgDistanceTravelDF.csv"
avgDistanceTravelDF = pd.read_csv(fileName)

centerOfVenuesDF = pd.read_csv("/Users/USER/Desktop/Thesis/Recommendation_code_kevon_15_7_18/centerOfVenuesDF.csv")

data=data.loc[data.CheckInCount>3]
userPlacedCheckInMatrix=pd.pivot_table(data, values='CheckInCount',
                                    index=['userId'], columns=['venueId'])

def similarityPearson(user1,user2):
    warnings.filterwarnings('error')
    user1=np.array(user1)-np.nanmean(user1)  
    user2=np.array(user2)-np.nanmean(user2)
    commonItemIds=[i for i in range(len(user1)) if user1[i]>0 and user2[i]>0]
    if len(commonItemIds)==0:
        return 0
    else:
        user1=np.array([user1[i] for i in commonItemIds])
        user2=np.array([user2[i] for i in commonItemIds])
        try:
            return pearsonr(user1,user2)[0]
        except RuntimeWarning:
            return 0
   
predictedItemCheckinGlobal=pd.DataFrame()

def nearestNeighbourCheckins(activeUser,K):
    similarityMatrix=pd.DataFrame(index=userPlacedCheckInMatrix.index,
                                  columns=['Similarity'])
    for i in userPlacedCheckInMatrix.index:
        similarityMatrix.loc[i]=similarityPearson(userPlacedCheckInMatrix.loc[activeUser],
                                          userPlacedCheckInMatrix.loc[i])
    similarityMatrix=pd.DataFrame.sort_values(similarityMatrix,
                                              ['Similarity'],ascending=[0])
    nearestNeighbours=similarityMatrix[:K]
    neighbourItemRatings=userPlacedCheckInMatrix.loc[nearestNeighbours.index]
    SumOfNearestNeighbourRatings=nearestNeighbours.loc[neighbourItemRatings.index,'Similarity'].sum()
    predictItemCheckin=pd.DataFrame(index=userPlacedCheckInMatrix.columns, columns=['CheckInCount'])
    
    for i in userPlacedCheckInMatrix.columns:
        predictedCheckin=np.nanmean(userPlacedCheckInMatrix.loc[activeUser])
        predict = 0
        for j in neighbourItemRatings.index:
            if userPlacedCheckInMatrix.loc[j,i]>0:
                 predict += (userPlacedCheckInMatrix.loc[j,i]
                                    -np.nanmean(userPlacedCheckInMatrix.loc[j]))*nearestNeighbours.loc[j,'Similarity']
   
        if  SumOfNearestNeighbourRatings>0:
            predict=predict/SumOfNearestNeighbourRatings
        predictedCheckin+=predict
        predictItemCheckin.loc[i,'CheckInCount']=predictedCheckin
    global predictedItemCheckinGlobal
    predictedItemCheckinGlobal=predictItemCheckin
    return predictItemCheckin


def topNRecommendations(activeUser,N):
    predictPlaceCheckins=nearestNeighbourCheckins(activeUser,10)
    placeAlreadyChecked=list(userPlacedCheckInMatrix.loc[activeUser]
                              .loc[userPlacedCheckInMatrix.loc[activeUser]>0].index)
    predictPlaceCheckins=predictPlaceCheckins.drop(placeAlreadyChecked)
    topRecommendations=pd.DataFrame.sort_values(predictPlaceCheckins,
                                                ['CheckInCount'],ascending=[0])[:N]
    topRecommendationTitles=(data.loc[data.venueId.isin(topRecommendations.index)])
    topRecommendationTitles=topRecommendationTitles.drop_duplicates(['venueId'], keep='first')
    return list(topRecommendationTitles.venueCategory)     


def favoritePlaces(activeUser,N):
    topPlaceCheckedIn=pd.DataFrame.sort_values(
        data[data.userId==activeUser],['CheckInCount'],ascending=[0])[:N]
    return list(topPlaceCheckedIn.venueCategory)


import math
def rootMeanSquareError(activeUser,K):
    global predictedItemCheckinGlobal
    rmse = 0
    ratedItemIds = [i for i in  userPlacedCheckInMatrix.columns 
                        if userPlacedCheckInMatrix.loc[activeUser,i]>0]
    N = len(ratedItemIds) 
    for i in ratedItemIds:
        rmse+= math.pow((predictedItemCheckinGlobal.loc[i,'CheckInCount']-userPlacedCheckInMatrix.loc[activeUser,i]),2)

    rmse = rmse/N
    rmse = math.sqrt(rmse)

    return rmse

def usersVenueDistancePreference(activeUser):
    activeUserDF = data.loc[data.userId==activeUser,['distanceFromCenter','CheckInCount']]
    avgDistance = float(avgDistanceTravelDF.loc[avgDistanceTravelDF.userId == activeUser,'avgDistanceTravel'])
    closeVenue = 0 #no. of checkins venues which are less than and equal to average distance
    farVenue = 0  #no. of checkins venues which are more than average distance
    for index,row in activeUserDF.iterrows():
        if row['distanceFromCenter']<= avgDistance:
            closeVenue = closeVenue + row['CheckInCount']
        else:
            farVenue = farVenue + row['CheckInCount']

    venuePreference = {'avgDistancePerCheckIn':avgDistance} #the average distance per checkIn
    if closeVenue>farVenue: #likes closer places
         venuePreference['likesPlace']='close'
    elif closeVenue<farVenue: #likes farther places
         venuePreference['likesPlace']='far'
    elif closeVenue==farVenue: #likes both type of places equally
         venuePreference['likesPlace']='both'
    else:
        print('Some Error Happened,Check!!')
        
    return venuePreference


def topNRecommendationsFilterByCheckIn(activeUser,N):
    
    N = N+10 #incerese the limit by 10 and send 10 more venue 
    predictPlaceCheckins=nearestNeighbourCheckins(activeUser,10)
    placeAlreadyChecked=list(userPlacedCheckInMatrix.loc[activeUser]
                              .loc[userPlacedCheckInMatrix.loc[activeUser]>0].index)
    predictPlaceCheckins=predictPlaceCheckins.drop(placeAlreadyChecked)

    topRecommendations=pd.DataFrame.sort_values(predictPlaceCheckins,
                                            ['CheckInCount'],ascending=[0])[:N]
    return topRecommendations

def topNRecommendationsFilterByDistanceAndCheckIn(activeUser,N):
    
    topNRecommendations = topNRecommendationsFilterByCheckIn(activeUser,N).reset_index(drop=False)
    predictedItemDF = venueDistanceFromActiveUser(activeUser,topNRecommendations)
    userPreference = usersVenueDistancePreference(activeUser)  #returns a dictionary
    topRecommendationTitles=(data.loc[data.venueId.isin(predictedItemDF.venueId)])
    
    if (userPreference['likesPlace'] =='close') |  (userPreference['likesPlace'] =='both' ):
        topRecommendationTitles = pd.DataFrame.sort_values(topRecommendationTitles,
                                                ['distanceFromCenter'],ascending=[True])[:N]
    elif (userPreference['likesPlace'] =='far'):
        topRecommendationTitles = pd.DataFrame.sort_values(topRecommendationTitles,
                                                ['distanceFromCenter'],ascending=[False])[:N]
    else:
        print('Something Went Wrong! Check Code')

    topRecommendationTitles=topRecommendationTitles.drop_duplicates(['venueId'], keep='first')
    return topRecommendationTitles


def distance(centerLat,centerLong,targetVenueLat,targetVenueLong):
    
    lat1 = centerLat
    lon1 = centerLong  
    lat2 = targetVenueLat
    lon2 = targetVenueLong
    radius = 6371 # km
    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c
    return d


def venueDistanceFromActiveUser(activeUser,predictedItemDF):
    centerLat = centerOfVenuesDF.loc[centerOfVenuesDF.userId == activeUser,'latitude'].item()
    centerLong = centerOfVenuesDF.loc[centerOfVenuesDF.userId == activeUser,'longitude'].item()
    venueDistanceDF = pd.DataFrame()

    for index,row in predictedItemDF.iterrows():
        targetVenueLat = data.loc[data.venueId==row['venueId'],'latitude'][:1].item()
        targetVenueLong = data.loc[data.venueId==row['venueId'],'longitude'][:1].item()
        distanceFromCenter = distance(centerLat,centerLong,targetVenueLat,targetVenueLong)
        copyData=[[row['venueId'],row['CheckInCount'],distanceFromCenter]]
        df = pd.DataFrame(copyData,columns=['venueId','CheckInCount','distanceFromCenter'])
        venueDistanceDF=venueDistanceDF.append(df)
    return venueDistanceDF

class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        self.title = "Location-based Recommendation System"        
        self.top = 100        
        self.left = 100        
        self.width = 680        
        self.height = 500

        self.InitWindow()


    def InitWindow(self):
        self.setWindowTitle(self.title)
        self.setWindowIcon(QtGui.QIcon("location.png"))
        self.setGeometry(self.top, self.left, self.width, self.height)
        p = self.palette()
        p.setColor(self.backgroundRole(), QtGui.QColor(100,100,150))
        self.setPalette(p)
        self.linedit=QLineEdit(self)
        self.linedit.setPlaceholderText("Enter User Id")
        self.linedit.setStyleSheet("font-size: 13px;")
        self.linedit.move(200,200)
        self.linedit.resize(280,40)
        self.button=QPushButton("Recommend",self)
        self.button.move(200,250)
        self.button.resize(280,31)
        self.button.setStyleSheet("background-color: #A1887f;font-size: 13px;color:white;font:bold;")
        self.setStyleSheet("QMessageBox {background-color: #646496;font-size: 14px;}")
        self.button.clicked.connect(self.onClick)
        self.button=QPushButton("Exit",self)
        self.button.clicked.connect(self.onClick2)
        self.button.move(200,290)
        self.button.resize(280,31)
        self.button.setStyleSheet("background-color: #A1887f;font-size: 13px;color:white;font:bold;")
        self.button.colorCount
        self.show()
        
    def onClick(self):
        textValue = self.linedit.text()
        try:
            d= int(textValue)
            if d>0 and d<1084:
              recommend=topNRecommendationsFilterByDistanceAndCheckIn(d,3)
              showMessage(self,recommend)
            else:
               showErrorMessage(self,"Sorry , This user id doesn't exist")
        except ValueError:
              showErrorMessage(self,"Please provide valid user id")
    
    def onClick2(self):
       self.close()
    

def showMessage(self,recommend):
    catagoryType1=""
    catagoryType2=""
    catagoryType3=""
    latitude1=""
    latitude2=""
    latitude3=""
    longitude1=""
    longitude2=""
    longitude3=""
    
    i=0
    for index, row in recommend.iterrows():
        if i==0:
            catagoryType1=row["venueCategory"]
            latitude1=row["latitude"]
            longitude1=row["longitude"]
            i=i+1
        elif i==1:
            catagoryType2=row["venueCategory"]
            latitude2=row["latitude"]
            longitude2=row["longitude"]
            i=i+1
        elif i==2:
            catagoryType3=row["venueCategory"]
            latitude3=row["latitude"]
            longitude3=row["longitude"]
            i=i+1
            
    msg=QMessageBox()
    msg.about(self, 'Recommendation',
            """<font color='white'><p><b><br/>Recommended Place For You</b></p>
            <br/>
            <p><b>Catagory :</b> """+catagoryType1+"""</p>
            <p><b>Latitude: </b>"""+str(latitude1)+
            """<p><b>Longitude: </b>"""+str(longitude1)+"<br/>"+
            """<p><b>Catagory :</b> """+catagoryType2+"""</p>
            <p><b>Latitude: </b>"""+str(longitude2)+
            """<p><b>Longitude: </b>"""+str(latitude2)+"<br/>"
            """<p><b>Catagory :</b> """+catagoryType3+"""</p>
            <p><b>Latitude: </b>"""+str(latitude3)+
            """<p><b>Longitude: </b>"""+str(longitude3)+"<br/>"
            
            """</p>
            <font color='#646496'><p><b>Email: </b>farhantanvir65@gmail@gmail.com</p>
            <p><b>Copyright:</b>  &copy; 2014 farhandroid Ltd.
            All rights reserved.
            <p>This application can be used to recommend new place                     </p><br/>"""
    
            )
    
def showErrorMessage(self, message):
    msg=QMessageBox()
    msg.about(self, 'Error',
            """<font color='white'><p><b><br/>"""+message+"""</b></p>
            <br/>
            </p>
            <font color='#646496'><p><b>Email: </b>farhantanvir65@gmail@gmail.com</p>
            <p>""")
    
app = QtCore.QCoreApplication.instance()
if app is None:
    app = QApplication(sys.argv)
    
window = Window()#
sys.exit(app.exec())
