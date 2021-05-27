# -*- coding: utf-8 -*-
"""
Created on Mon May 10 10:30:39 2021

@author: Louis
"""
import numpy as np
import cv2
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from time import perf_counter
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage import color

def computeFilterBank(sigma,theta,lamba,gamma,kSize):
	
	k = kSize
	phi = 0
	
	nb_filter = len(sigma)*len(theta)*len(lamba)*len(gamma)
	print("Number of filters in the bank : ",nb_filter)
	filter_bank = np.ones((k,k,nb_filter))
	
	it = 0
	
	for s in sigma:
		for t in theta:
			for l in lamba:
				for g in gamma:
					filter_bank[:,:,it] = cv2.getGaborKernel((k,k),s,t,l,g,phi)
					it+=1
	
	print("Filter bank initialized")
				
	return filter_bank, nb_filter




def computeGaborCoef(X,filtBank,nbFilt):
	
	coef_X = []


	for index in range(X.shape[0]):

		img = cv2.cvtColor(X[index],cv2.COLOR_BGR2GRAY)
		img = np.float32(img[31:64,31:64])
		
		
		coef = []
		
		for f in range(nbFilt):
			coef.append(convolve2d(filtBank[:,:,f],img,mode='valid')[0][0])				
						
		coef_X.append(coef)
		
	print("Gabor coeficients computed")
		
	coef_X = np.reshape(coef_X,(len(coef_X),len(coef_X[0])))
		
	return coef_X


def computeGaborEnergy(X,filtBank,nbFilt,blur=False,accent=False):
	
	
	coef_X = []

	tps1 = perf_counter()
	tp = perf_counter()
	
	for index in range(X.shape[0]):
		
		
		
		img = cv2.cvtColor(X[index],cv2.COLOR_BGR2GRAY)
		img = np.float32(img[25:71,25:71])
		
		if blur:
			img = cv2.GaussianBlur(img,(5,5),0.5)
		if accent:
			kernel = np.array([[0,-0.5,0],[-0.5,3,-0.5],[0,-0.5,0]])
			img = cv2.filter2D(img,-1,kernel)
		
		
		coef = []
		
		for f in range(nbFilt):
			convImg = convolve2d(filtBank[:,:,f],img,mode='full')
			#convImg = img[7:39,7:39]
			energy = np.sum(np.power(convImg,2))/(convImg.shape[0]*convImg.shape[0])
			coef.append(energy)				
						
		coef_X.append(coef)
		
		if np.mod(index,1000)==0:
			print("process time",index,"images =",perf_counter()-tp)
		
		
	print("Gabor coeficients computed")
	print("Process time =",perf_counter()-tps1)
		
	coef_X = np.reshape(coef_X,(len(coef_X),len(coef_X[0])))
		
	return coef_X



def explainer(X,filtBanks,nbFilt,model,stdScale,pca_,predLabel,GT):
	
	
	coef_X = np.zeros((8,16))

	tps1 = perf_counter()
	tp = perf_counter()
	
	for nbf,fb in enumerate(filtBanks):
	
		filtBank = fb
		
		for index in range(X.shape[0]):
			
			img = cv2.cvtColor(X[index],cv2.COLOR_BGR2GRAY)
			img = np.float32(img[25:71,25:71])
			
			img0 = img[:int(img.shape[0]/2),:int(img.shape[1]/2)] #top left
			img1 = img[int(img.shape[0]/2):img.shape[0],:int(img.shape[1]/2)] #top right
			img2 = img[:int(img.shape[0]/2),int(img.shape[1]/2):img.shape[1]] #bottom left
			img3 = img[int(img.shape[0]/2):img.shape[0],int(img.shape[1]/2):img.shape[1]] #bottom right
			
			img4 = img[:int(img.shape[0]/2),:img.shape[1]] #top horizontal
			img5 = img[int(img.shape[0]/2):img.shape[1],:img.shape[1]] #bottom horizontal
			img6 = img[:img.shape[0],:int(img.shape[1]/2)] #left vertical
			img7 = img[:img.shape[0],int(img.shape[1]/2):img.shape[1]] #right vertical
			
			sub_img = [img0,img1,img2,img3,img4,img5,img6,img7]
			
			for k,image in enumerate(sub_img):
				
				
				coef = []
				
				for f in range(nbFilt):
					convImg = convolve2d(filtBank[:,:,f],image,mode='full')
					energy = np.sum(np.power(convImg,2))/(convImg.shape[0]*convImg.shape[0])
					coef.append(energy)				
								
				coef_X[k,nbf*8:nbf*8+8] = coef
		
	print("Gabor coeficients computed")
	print("Process time =",perf_counter()-tps1)
		
	X_test = np.reshape(coef_X,(len(coef_X),len(coef_X[0])))
	
	X_test[0,:] = pca_.transform(stdScale.transform([X_test[0,:]]))
	X_test[1,:] = pca_.transform(stdScale.transform([X_test[1,:]]))
	X_test[2,:] = pca_.transform(stdScale.transform([X_test[2,:]]))
	X_test[3,:] = pca_.transform(stdScale.transform([X_test[3,:]]))
	X_test[4,:] = pca_.transform(stdScale.transform([X_test[4,:]]))
	X_test[5,:] = pca_.transform(stdScale.transform([X_test[5,:]]))
	X_test[6,:] = pca_.transform(stdScale.transform([X_test[6,:]]))
	X_test[7,:] = pca_.transform(stdScale.transform([X_test[7,:]]))
	
	y_pred = model.predict(X_test)

	print(y_pred)
	print("GT label :",GT)
	print("Predicted :",predLabel)
	
	X = X[0]
	heatmap = 2*np.ones(X.shape[:2])

	heatmap[25:25+23,25:25+23] = y_pred[0]
	heatmap[25:25+23,25+23:heatmap.shape[1]-25] = y_pred[1]
	heatmap[25+23:heatmap.shape[0]-25,25:25+23] =y_pred[2]
	heatmap[25+23:heatmap.shape[0]-25,25+23:heatmap.shape[1]-25] = y_pred[3]
	
	heatmapHorz = 2*np.ones(X.shape[:2])
	heatmapHorz[25:25+23,25:heatmapHorz.shape[1]-25] = y_pred[4]
	heatmapHorz[25+23:heatmapHorz.shape[0]-25,25:heatmapHorz.shape[1]-25] = y_pred[5]
	
	heatmapVert = 2*np.ones(X.shape[:2])
	heatmapVert[25:heatmapVert.shape[0]-25,25:25+23] = y_pred[6]
	heatmapVert[25:heatmapVert.shape[0]-25,25+23:heatmapVert.shape[1]-25] = y_pred[7]
	
	
	c=['red','green']
	result_image = color.label2rgb(heatmap,X,colors=c,image_alpha=1,bg_label=2)
	result_imageVert = color.label2rgb(heatmapVert,X,colors=c,image_alpha=1,bg_label=2)
	result_imageHorz = color.label2rgb(heatmapHorz,X,colors=c,image_alpha=1,bg_label=2)
	
	plt.subplot(221)
	plt.imshow(X)
	plt.title('original')
	plt.subplot(222)
	plt.imshow(result_image)
	plt.title('Square')
	plt.subplot(223)
	plt.imshow(result_imageVert)
	plt.title('Vertical')
	plt.subplot(224)
	plt.imshow(result_imageHorz)
	plt.title('Horizontal')
	plt.tight_layout()

	return coef_X
	
	
def trainRF(X_train,X_test,y_train,y_test,usePCA=False,nbPCA=16):
	
	stdScale = StandardScaler()
	stdScale.fit(X_train)
	X_train = stdScale.transform(X_train)
	X_test = stdScale.transform(X_test)
	
	if usePCA:
		pca = PCA(n_components=nbPCA)
		pca.fit(X_train)
		X_train = pca.transform(X_train)
		X_test = pca.transform(X_test)
	
	acc = []
	classifier = []
	depth = [1,2,5,10,25,50,75,100,150,250]
	for k in depth:	
	  RF = RandomForestClassifier(criterion='entropy',n_estimators=k,random_state=42)
	  RF.fit(X_train,y_train)
	  classifier.append(RF)
	  y_pred = RF.predict(X_test)
	
	  #print("Accuracy =",accuracy_score(y_test1,y_pred1)*100,"% (",k,"trees in the forest)")
	  acc.append(accuracy_score(y_test,y_pred)*100)
	
	plt.figure(figsize=(8,5))
	plt.plot(depth,acc)
	plt.xlabel("Number of estimators")
	plt.ylabel("Accuracy (%)")
	plt.grid()
	plt.show()
	
	RFbest = classifier[np.argmax(acc)]
	y_pred = RFbest.predict(X_test)
	print("AUC =",roc_auc_score(y_test,RFbest.predict_proba(X_test)[:,1]))
	tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
	print("Best classifier (for 1000 images) depth = ",depth[np.argmax(acc)],
		   ", accuracy =",np.max(acc),"%")
	print("TP =",tp)
	print("TN =",tn)
	print("FP =",fp)
	print("FN =",fn)
	
	if usePCA:
		return RFbest,stdScale,pca
	else :
		return RFbest,stdScale

def classifyRF(X_test,stdScale,model,usePCA=False,pca_=None):

	X_test = stdScale.transform(X_test)
	
	if usePCA:
		X_test = pca_.transform(X_test)
	
	return model.predict(X_test)