import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
import plotly.graphobjectsasgo
fromplotly.subplotsimportmakesubplots
fromsklearn.utilsimportclassweight
fromsklearn.modelselectionimporttraintestsplit
importtensorflowastf
fromtensorflow.keras.preprocessing.imageimport
ImageDataGenerator
fromkeras.modelsimportSequential
fromkeras.layersimportConv2D, M axP ool2D, Flatten, Dense,
Dropout, BatchNormalization, Input
fromtensorflow.keras.callbacksimportCallback,
EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
fromtensorflow.keras.optimizersimportAdam
fromtensorflow.keras.applications.inceptionv3importInceptionV 3
fromkeras.applications.vgg16importV GG16
importos
importcv2
cobra = r”/kaggle/input/snake − bites/train/cobra”
coral = r”/kaggle/input/snake − bites/train/coral”
37
kingcobra = r”/kaggle/input/snake − bites/train/kingcobra”
krait = r”/kaggle/input/snake − bites/train/krait”
seasnake = r”/kaggle/input/snake − bites/train/seasnake”
viper = r”/kaggle/input/snake − bites/train/viper”
imgc
lasses = [”cobra”, ”coral”,
”kingcobra”, ”krait”, ”seasnake”, ”viper”]
pathlist = [cobra, coral, kingcobra, krait, seasnake, viper]
imgpath = []
classlabels = []
fori, dirlistinenumerate(pathlist) :
nameimg = os.listdir(dirlist)
fornamef ileinnameimg :
img = os.path.join(dirlist, namef ile)
imgpath.append(img)
classlabels.append(imgc
lasses[i])
df = pd.DataFrame(”imgpath” : imgpath, ”label” : classlabels)
df.head()
for category, group in df.groupby(”label”):
fig, ax = plt.subplots(1,3, figsize = (8,8))
ax = ax.ravel()
for i, (,r)inenumerate(group.sample(3).iterrows()) :
img = cv2.imread(r.imgpath)
ax[i].imshow(img)
ax[i].axis(”off”)
ax[i].settitle(r.label)
plt.show()
fromplotly.offlineimportinitnotebookmode, iplot
initnotebookmode(connected = T rue)
countData = df[”label”].valuecounts().resetindex()
countData.head(10)
f ig = px.histogram(dataf rame = countData, x =′
label′
, y =′
count′
)
38
f ig.show()
sizes = []
resolutions = []
colordistributions = []
for imgpathindf[”imgpath”] :
loadimage
img = cv2.imread(imgpath)
img = cv2.cvtColor(img, cv2.COLORBGR2RGB)
Getimagesize
size = os.path.getsize(imgpath)
sizes.append(size)
Extracttheresolutionof images
resolution = img.shape[: 2]
resolutions.append(resolution)
Extractcolordistribution
meancolordistributions = np.bincount(img.flatten(), minlength = 256)
colordistributions.append(meancolordistributions)sizes = np.array(sizes)
resolutions = np.array(resolutions)
colordistributions = np.array(colordistributions)
plt.hist(sizes)
plt.title(”DistributionofImageSize”)
plt.xlabel(”bytes”)
plt.ylabel(”NumberofImages”)
plt.show()
sizesMB = []
forimgpathindf[”imgpath”] :
loadimage
img = cv2.imread(imgpath)
Getimagesize
size = os.path.getsize(imgpath)
sizesMB.append(size/1000000)
fig = px.histogram(x=sizesMB, nbins = 50, title = ”DistributionofImageSizes”)
f ig.updatelayout(xaxistitle = ”F ileSize(MB)”,
39
yaxistitle = ”NumberofImages”,
showlegend = F alse,
bargap = 0.1,
bargroupgap = 0.1)
f ig.updatetraces(marker = dict(color = ”green”))
f ig.show()
plt.scatter(resolutions[:, 0], resolutions[:, 1])
plt.title(”DistributionofImageResolutions”)
plt.ylabel(”Height(P ixel)”)
plt.xlabel(”W idth(P ixel)”)
plt.show()
f ig = px.scatter(x = resolutions[:, 0], y = resolutions[:, 1], title = ”DistributionofImagf ig.updatelayout(xaxistitle = ”W idth(P ixel)”,
yaxistitle = ”Height(P ixel)”,
showlegend = F alse,
hovermode = ”closest”)
f ig.updatetraces(marker = dict(color = ”red”))
f ig.show()
importplotly.graphobjectsasgomeancolordistributions = np.mean(colordistributions, ax0)
f ig = go.F igure(go.Bar(x = np.arange(256), y = meancolordistributions, name =
”MeanColorDistributions”))
f ig.updatelayout(title = ”MeanColorDistribution”, xaxistitle = ”ColorV alues”, yaxis”NumberofP ixel”)
f ig.show()
trainratio = 0.70
testratio = 0.15
valratio = 0.15
dftrain, dftestval = traintestsplit(df, trainsize = trainratio, randomstate =
42)
dftest, dfval = traintestsplit(dftestval, trainsize = testratio/(testratio+valratio), random42)
print(f”Train shape = dftrain.shape”)
40
print(f”T estshape = dftest.shape”)
print(f”V alidationshape = dfval.shape”)
defpreprocessingdenoise(img) :
denoiseimg = cv2.medianBlur(img, 1)
denoiseimg = cv2.cvtColor(denoiseimg, cv2.COLORBGR2RGB)
returndenoiseimg
IMGW IDT H = 224
IMGHEIGHT = 224
imagesize = (IMGW IDT H, IMGHEIGHT)
batchsize = 32
TRAINDAT AGEN = ImageDataGenerator(rescale = 1./255., preprocessingfunctiopreprocessingdenoise, rotationrange = 30, widthshif trange = 0.1, heightshif trange =
0.2, shearrange = 0.1, zoomrange = 0.2, horizontalf lip = T rue)
TESTDAT AGEN = ImageDataGenerator(rescale = 1./255.)
traingenerator = T RAINDAT AGEN.flowf romdataframe(dftrain, xcol =
”imgpath”, ycol = ”label”, targetsize = imagesize, batchsize = batchsize, colormode =
”rgb”, classmode = ”categorical”, shuffle = T rue)
valgenerator = T ESTDAT AGEN.flowf romdataframe(dfval, xcol = ”imgpath”, ycol ”label”, targetsize = imagesize, batchsize = batchsize, colormode = ”rgb”, classmode =
”categorical”, shuffle = T rue)
testgenerator = T ESTDAT AGEN.flowf romdataframe(dftest, xcol = ”imgpath”, ycol”label”, targetsize = imagesize, batchsize = batchsize, colormode = ”rgb”, classmode =
”categorical”, shuffle = T rue)
classes = list(traingenerator.classindices.keys())classes
classweights = classweight.computec
lassweight(classweight =′
balanced′
, classes =
np.unique(traingenerator.classes), y = traingenerator.classes)
trainc
lassweights = dict(enumerate(classweights))
41
for idx, weight, in trainc
lassweights.items() :
classname = classes[idx]
print(f”classname : weight”)
model1 = Sequential([Conv2D(f ilters = 64, kernelsize = (3, 3), strides =
(1, 1), inputshape = (224, 224, 3), activation = ”relu”, padding = ”same”),
M axP ool2D(poolsize = (2, 2)),
BatchNormalization(),
Conv2D(filters = 64, kernelsize = (3, 3), strides = (1, 1), padding = ”valid”),
M axP ool2D(poolsize = (2, 2)),
BatchNormalization(),
Conv2D(filters = 128, kernelsize = (3, 3), strides = (1, 1), padding = ”valid”),
M axP ool2D(poolsize = (2, 2)),
BatchNormalization(),
Conv2D(filters = 128, kernelsize = (3, 3), strides = (1, 1), padding = ”valid”),
M axP ool2D(poolsize = (2, 2)),
BatchNormalization(),
Conv2D(filters = 256, kernelsize = (3, 3), strides = (1, 1), padding = ”valid”),
M axP ool2D(poolsize = (2, 2)),
BatchNormalization(),
Conv2D(filters = 256, kernelsize = (3, 3), strides = (1, 1), padding = ”valid”),
M axP ool2D(poolsize = (2, 2)),
BatchNormalization(),
Flatten(),
Dense(4096, activation = ”relu”),
Dropout(0.5),
Dense(256),
Dropout(0.25),
Dense(6, activation = ”softmax”) ])
42
model1.summary()
model1.compile( loss = ”categoricalcrossentropy”, optimizer = Adam(learningrate =
0.0005),
metrics = [”accuracy”],)
epochs = 50
history = model1.fit(traingenerator, stepsperepoch = len(traingenerator), batchsize =
32, validationdata = valgenerator, validationsteps = len(valgenerator), classweight =
trainc
lassweights, callbacks = [EarlyStopping(monitor = ”valloss”, watchthevallossme5, restorebestweights = T rue),
ReduceLROnPlateau(monitor =′
valloss′
, f actor = 0.2, patience = 4, mode =′
min′
)],
epochs = epochs)
defhistoryplot(epochs, history) : f ig1 = makesubplots()
f ig1.addtrace(go.Scatter(x = np.arange(1, epochs+1), y = history.history[”accuracy”],”T rainingAccuracy”))
f ig1.addtrace(go.Scatter(x = np.arange(1, epochs+1), y = history.history[”valaccuracy”V alidationAccuracy”))
f ig1.updatelayout(title = ”T rainingandV alidationAccuracy”, xaxistitle = ”Epoch”, ya”Accuracy”)
f ig1.show()
f ig2 = makesubplots()
f ig2.addtrace(go.Scatter(x = np.arange(1, epochs+1), y = history.history[”loss”], nam”T rainingLoss”))f ig2.addtrace(go.Scatter(x = np.arange(1, epochs + 1), y =
history.history[”valloss”], name = ”V alidationLoss”))
f ig2.updatelayout(title = ”T rainingandV alidationLoss”, xaxistitle = ”Epoch”, yaxisti”Loss”)f ig2.show()
historyplot(epochs, history)
defevaluatemodel(model, testgenerator) :
Calculatetestlossandaccuracyresults = model.evaluate(testgenerator, verbose =
0)
print(f”T estLoss = results[0]”)
print(f”T estAccuracy = results[1]”)
evaluatemodel(model1, testgenerator)
model1.save(”CNN(Custom).keras”)
43
basemodel = InceptionV 3(inputshape = (IMGW IDT H, IMGHEIGHT, 3), includetop =F alse, weights = ”imagenet”)
forlayerinbasemodel.layers : layer.trainable = F alse
model2 = Sequential()
model2.add(Input(shape = (IMGW IDT H, IMGHEIGHT, 3)))
model2.add(basemodel)
model2.add(Flatten())
model2.add(Dense(1024, activation = ”relu”))
model2.add(Dropout(0.4))
model2.add(Dense(6, activation = ”sof tmax”))
epochs = 50
model2.compile(optimizer = Adam(0.0005),
loss = ”categoricalcrossentropy”,
metrics = [”accuracy”])
history = model2.f it(traingenerator, stepsperepoch = len(traingenerator), batchsize =
64, validationdata = valgenerator, validationsteps = len(valgenerator), classweight =
trainc
lassweights, callbacks = [EarlyStopping(monitor = ”valloss”, watchthevallossme5, restorebestweights = T rue),
ReduceLROnPlateau(monitor =′
valloss′
, f actor = 0.2, patience = 2, mode =′
min′
)], epochs = epochs)
print(model2.summary())
historyplot(epochs, history)
evaluatemodel(model2, testgenerator)
basemodel.trainable = T rue
for l in basemodel.layers[: 10] : print(l.name, l.trainable)
model2.compile(optimizer = Adam(0.00001), loss = ”categoricalcrossentropy”, metrics =[”accuracy”])
print(model2.summary())
epochs = 50
history = model2.fit(traingenerator, stepsperepoch = len(traingenerator), batchsize =
64, validationdata = valgenerator, validationsteps = len(valgenerator), classweight =
44
trainc
lassweights, callbacks = [EarlyStopping(monitor = ”valloss”, watchthevallossme5, restorebestweights = T rue),
ReduceLROnPlateau(monitor =′
valloss′
, f actor = 0.2, patience = 2, mode =′
min′
)], epochs = epochs)
historyplot(epochs, history)
evaluatemodel(model2, testgenerator)
model2.save(”InceptionV 3.keras”)
basemodelvgg16 = V GG16(inputshape = (IMGW IDT H, IMGHEIGHT, 3), includetop F alse, weights = ”imagenet”)
forlayerinbasemodelvgg16.layers : layer.trainable = F alse
model3 = Sequential()
model3.add(Input(shape = (IMGW IDT H, IMGHEIGHT, 3)))model3.add(basemodelvgmodel3.add(Flatten())
model3.add(Dense(1024, activation = ”relu”))
model3.add(Dropout(0.4))
model3.add(Dense(3, activation = ”sof tmax”))
model3.compile(optimizer = Adam(0.0005),
loss = ”categoricalcrossentropy”,
metrics = [”accuracy”])
epochs = 50
history = model3.fit(traingenerator,
stepsperepoch = len(traingenerator), batchsize = 64, validationdata = valgenerator, valilen(valgenerator), classweight = trainc
lassweights, callbacks = [EarlyStopping(monito”valloss”, watchthevallossmetricpatience = 5, restorebestweights = T rue),
ReduceLROnPlateau(monitor =′
valloss′
, f actor = 0.2, patience = 4, mode =′
min′
)], epochs = epochs)
print(model3.summary())
newmodel = tf.keras.models.loadmodel(
′V GG16.keras′
)
