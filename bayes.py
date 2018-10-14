import numpy as np
def loadDataSet(): #试验样本
	postingList = [['my','dog','has','flea',\
					'problems','help','please'],
					['maybe','not','take','him',\
					'to','dog','park','stupid'],
					['my','dalmation','is','so','cute',\
					'I','love','him'],
					['stop','posting','stupid','worthless','garbage'],
					['mr','lick','ate','my','steak','how',\
					'to','stop','him'],
					['quit','buying','worthless','dog','food','stupid']]
	classVec = [0,1,0,1,0,1] #1代表侮辱性文字，0代表正常言论
	return postingList,classVec

def createVocabList(dataSet):
	vocabSet = set([])  #创建1个空集，用于加载不重复的词
	for document in dataSet: 
		vocabSet = vocabSet | set(document) #创建两个集合的并集
	return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):
	returnVec = [0] * len(vocabList) #初始设置为0
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] = 1 #1表示在词汇表中存在
		else:
			print("the word: %s is not in my vocabulary!" % word)
	return returnVec

#朴素贝叶斯分类器训练函数
def trainNB0(trainMatrix,trainCategory):
	numTrainDocs = len(trainMatrix)
	numWords = len(trainMatrix[0])
	pAbusive = sum(trainCategory) / float(numTrainDocs)
	p0Num = np.ones(numWords); #p0属于正常文档的概率
	p1Num = np.ones(numWords); #p1属于侮辱性文档的概率
	p0Denom = 2.0;
	p1Denom = 2.0;
	for i in range(numTrainDocs): #遍历文档中的词
		if trainCategory[i] == 1: #出现侮辱性词p1加1，总词数也+1
			p1Num += trainMatrix[i];
			p1Denom += sum(trainMatrix[i])
		else:                    #出现非侮辱性词p0加1，总词数也+1
			p0Num += trainMatrix[i];
			p0Denom += sum(trainMatrix[i])
	p1Vect = np.log(p1Num / p1Denom)
	p0Vect = np.log(p0Num / p0Denom)
	return p0Vect,p1Vect,pAbusive

#朴素贝叶斯分类函数
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
	p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
	p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)

	if p1 > p0:
		return 1
	else:
		return 0

def testingNB():
	listOPosts,listClasses = loadDataSet()
	myVocabList = createVocabList(listOPosts)
	trainMat = []
	for postinDoc in listOPosts:
		trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
	p0v,p1v,pAb = trainNB0(trainMat,listClasses)

	testEntry = ['love','my','dalmation']
	thisDoc = np.array(setOfWords2Vec(myVocabList,testEntry))
	print(testEntry,'classified as:',classifyNB(thisDoc,p0v,p1v,pAb))
	testEntry = ['stupid','垃圾']
	thisDoc = np.array(setOfWords2Vec(myVocabList,testEntry))
	print(testEntry,'classified as:',classifyNB(thisDoc,p0v,p1v,pAb))
	
def bagOfWords2VecMN(vocabList,inputSet):
	returnVec = [0] * len(vocabList) #初始设置为0
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] += 1 #增加词向量
		else:
			print("Note：the word: %s is not in my vocabulary!" % word)
	return returnVec

def textParse(bigString): #接受一个大字符串并解析为字符串列表
	import re
	listOfTokens = re.split('\W*',bigString)
	return [tok.lower() for tok in listOfTokens if len(tok) > 2] #去掉少于2个字符的字符串并将所有字符串转为小写

def spamTest(): #对贝叶斯垃圾邮件进行自动化处理
	docList = [] #
	classList = []
	fullText = []
	for i in range(1,26):
		wordList = textParse(open(r'E:/python/machinelearning/机器学习实战课文相关下载/machinelearninginaction/Ch04/email/spam/%d.txt' % i).read())
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(1)
		wordList = textParse(open('E:/python/machinelearning/机器学习实战课文相关下载/machinelearninginaction/Ch04/email/ham/%d.txt' % i,'rt',encoding = 'ISO-8859-1').read())
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(0)
	vocabList = createVocabList(docList)
	trainingSet = list(range(50)) #trainSet为整数列表，其中的值为0-49
	testSet = []
	for i in range(10): #随机选择10个邮件文件作为测试集
		randIndex = int(np.random.uniform(0,len(trainingSet)))#随机选出的数字
		testSet.append(trainingSet[randIndex]) #选出的数字所对应的文档添加到测试集
		del(trainingSet[randIndex])            #同时从训练集中删除
	trainMat = []
	trainClasses = []
	for docIndex in trainingSet:#遍历训练集的所有文档
		trainMat.append(setOfWords2Vec(vocabList,docList[docIndex])) #每封邮件基于词汇表使用setOfWord2Vec构建词向量
		trainClasses.append(classList[docIndex])
	p0V,p1V,pSpam = trainNB0(np.array(trainMat),np.array(trainClasses)) #计算分类所需的概率
	errorCount = 0

	for docIndex in testSet: #遍历测试集
		wordVector = setOfWords2Vec(vocabList,docList[docIndex])
		if classifyNB(np.array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:#对电子邮件进行分类，如果分类错误，errorCount+1
			errorCount += 1
	print("the error rate is: {}%".format(100 * float(errorCount) / len(testSet)))
	return float(errorCount) / len(testSet) #总错误百分比

#RSS源分类器及高频词去除函数
#计算出现的频率
def calcMostFreq(vocabList,fullText):
	import operator
	freqDict = {}
	for token in vocabList:
		freqDict[token] = fullText.count(token)
	sortedFreq = sorted(freqDict.items(),key = operator.itemgetter(1),reverse = True)
	return sortedFreq[:30] #返回前30个高频词

#
def localWords(feed1,feed0):
	import feedparser
	docList = []
	classList = []
	fullText = []
	minLen = min(len(feed1['entries']),len(feed0['entries']))#取feed0和feed1中的‘entries’列表最短

	print("minLen:%d"% minLen)

	for i in range(minLen):
		wordList = textParse(feed1['entries'][i]['summary']) #每次访问一条RSS源
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(1)
		wordList = textParse(feed0['entries'][i]['summary']) #每次访问一条RSS源
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(0)
	vocabList = createVocabList(docList)
	top30Words = calcMostFreq(vocabList,fullText)
	print("top30Words: %d" % len(top30Words))
	print(top30Words)
	for pairW in top30Words: #去掉高频词
		if pairW[0] in vocabList:
			vocabList.remove(pairW[0])
	trainingSet = list(range(2*minLen))
	testSet = []
	for i in range(5): #随机选择5个作为测试集
		randIndex = int(np.random.uniform(0,len(trainingSet))) #随机选出的数字
		testSet.append(trainingSet[randIndex])  #选出的数字所对应的文档添加到测试集
		del(trainingSet[randIndex])             #同时从训练集中删除

	trainMat = []
	trainClasses = []
	for docIndex in trainingSet: #遍历训练集
		trainMat.append(bagOfWords2VecMN(vocabList,docList[docIndex]))
		trainClasses.append(classList[docIndex])

	p0V,p1V,pSpam = trainNB0(np.array(trainMat),np.array(trainClasses)) #计算分类所需的概率
	errorCount = 0

	for docIndex in testSet: #遍历测试集
		wordVector = bagOfWords2VecMN(vocabList,docList[docIndex])
		if classifyNB(np.array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:#分类，如果分类错误，errorCount+1
			errorCount += 1
	print("the error rate is: {}%".format(100 * float(errorCount) / len(testSet)))
	return vocabList,p0V,p1V 

#最具表征性的词汇显示函数
def getTopWords(ny,sf):
	import operator
	vocabList,p0V,p1V = localWords(ny,sf)
	topNY = []; topSF = [] #列表存储元组
	for i in range(len(p0V)):
		if p0V[i] > -6.0: topSF.append((vocabList[i],p0V[i])) #-6.0为阈值
		if p1V[i] > -6.0: topNY.append((vocabList[i],p1V[i]))
	sortedSF = sorted(topSF,key = lambda pair:pair[1],reverse = True)
	print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
	for item in sortedSF:
		print(item[0])
	sortedNY = sorted(topNY,key = lambda pair:pair[1],reverse = True)
	print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
	for item in sortedNY:
		print(item[0])