# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:49:42 2020

@author: hemahemu
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from datetime import timedelta
#import plotly.graph_objs as go
#from plotly.offline import init_notebook_mode,iplot
#import warnings
#warnings.filterwarnings('ignore')
df1=pd.read_csv('G:/Machine learning projects/Customer segmentation-finalyearproject/year2009-10.csv',encoding='unicode_escape')
print(df1.head())
print(df1.shape)
print(df1.dtypes)
print(df1.describe())
df2=pd.read_csv('G:/Machine learning projects/Customer segmentation-finalyearproject/year2010-11.csv',encoding='unicode_escape')
print(df2.head())
print(df2.shape)
print(df2.dtypes)
print(df2.describe())
#concat
df=pd.concat([df1,df2])
print(df.head())
print(df.shape)
print('Number of duplicated records: ', df.duplicated(keep='first').sum())
#identifying null values
print(df.isnull().sum())
#Identify the number of NAs in each feature and select only those having NAs

totalna= df.isnull().sum()[df.isnull().sum() != 0]
print(totalna)

# Calculate the percentage of NA in each feature
percentageofna = totalna/df.shape[0]
print(percentageofna)
# Summarize our findings in a dataframe

missing = pd.concat([totalna, percentageofna], axis=1, keys=['Total NA Values', 'Percentage(%)']).sort_values('Total NA Values', ascending=False)
print(missing)

#droping null values
# Drop transactions with missing Customer ID
df.dropna(axis=0, subset=['Customer ID'], inplace= True)
print(df.isnull().sum())
print(df.shape)
print(df.info())
#changing the datatype 
df[['StockCode']] = df['StockCode'].astype(str)
df[['Customer ID']] = df['Customer ID'].astype(int).astype(str)
df[['Invoice']] = df['Invoice'].astype(str)
df['InvoiceDate'] = pd.to_datetime(df.InvoiceDate, format='%m/%d/%Y %H:%M')
print(df.info())

indx1 = df.Invoice[df.Invoice.str.contains('C') == True].index
print('Number of orders cancelled: {}/{} ({:.2f}%) '.format(len(indx1),df.shape[0],len(indx1)/df.shape[0]*100))
# Drop cancelled transactions
df.drop(index= indx1, inplace= True)
print(df.shape)
print(df.describe())
print(df.info())

#feature engineering
# Revenue per transaction which is the product of sale price and quantity
df['totalsales'] = df['Price'] * df['Quantity']
df[df.totalsales == 0].shape
# Drop transactions with zero Revenue
indx2 = df.loc[df.totalsales == 0].index
df_new = df.drop(index= indx2)

print(df_new.shape)
print(df_new.tail())
print(df_new.describe().T)
# Summary statistics of categorical variables
df_new.select_dtypes(include='object').describe().T
df_new.to_csv("Combined_cleaned_data_2009-12.csv")


print(df_new['Country'].value_counts())
print(df_new.Country.nunique())
print(df_new.Country.unique()) 

# Grouping by Country
country_df = df_new.groupby(['Country']).agg({
        'Customer ID': 'count',
        'Invoice': 'nunique',
        'totalsales': 'sum'})
# Rename the columns 

country_df.rename(columns={'Customer ID': 'No_of_customers',
                         'Invoice': 'No_of_transactions',
                         'totalsales': 'Sales'}, inplace=True)
country_df.sort_values('Sales',ascending=False,inplace=True)
country_df.head()



country_df['Avg_sales_per_customer'] = country_df['Sales'] / country_df['No_of_customers']
country_df.head()
country_df['Avg_sales_per_customer'].plot(kind='bar',figsize=(10,5),title='Average amount paid by the customer from a particular country')
plt.ylabel('Average amount')
plt.xlabel('Country')

final_df = df_new[['Invoice','totalsales','InvoiceDate','Country']].copy()
final_df['Month'] =final_df['InvoiceDate'].dt.month 
final_df['Year'] =final_df['InvoiceDate'].dt.year
final_df['Day'] =final_df['InvoiceDate'].dt.day

date_df = final_df.groupby(['Year','Month']).agg({'Invoice': 'nunique',
        'totalsales': 'sum'})
print(date_df)

month_df = final_df.groupby(['Month']).agg({'totalsales': 'sum'})
print(month_df)
month_df['totalsales'].plot(kind='bar',figsize=(10,5),title='Revenue collection by month')
plt.ylabel('total sales')
plt.xlabel('Month')
plt.show()
day_df = final_df.groupby(['Day']).agg({'totalsales': 'sum'})
print(day_df)
day_df['totalsales'].plot(kind='bar',figsize=(10,5),title='Revenue collection by day')
plt.ylabel('total sales')
plt.xlabel('day')
plt.show()
year_df = final_df.groupby(['Year']).agg({'totalsales': 'sum'})
print(year_df)
year_df['totalsales'].plot(kind='bar',figsize=(10,5),title='Revenue collection by year')
plt.ylabel('total sales')
plt.xlabel('year')
plt.show()

#NATURAL LANGUAGE PROCESSING
import nltk

is_noun = lambda pos: pos[:2] == 'NN'# NN is noun singular mass
#import nltk
#from nltk.stem import SnowballStemmer
#nltk.download('punkt')  #sentence_tokenizer
#nltk.download('averaged_perceptron_tagger') #pre trained english
def keywords_inventory(df_new, column = 'Description'):
    stemmer = nltk.stem.SnowballStemmer("english") #Unit tests for Snowball stemmer 
    #Create a new instance of a language specific subclass
    #Stem a word. Decide not to stem stopwords. 
    #The 'english' stemmer is better than the original 'porter' stemmer.
    keywords_roots  = dict()  # collect the words / root
    keywords_select = dict()  # association: root <-> keyword
    category_keys   = []
    count_keywords  = dict()
    #icount = 0
    for s in df_new[column]:
        if pd.isnull(s): continue
        lines = s.lower()
        tokenized = nltk.word_tokenize(lines)
        nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]  
        #pos is a part of speeching tagging

        for t in nouns:
            t = t.lower() ; root= stemmer.stem(t)
            if root in keywords_roots:                
                keywords_roots[root].add(t)
                count_keywords[root] += 1                
            else:
                keywords_roots[root] = {t}
                count_keywords[root] = 1

    for s in keywords_roots.keys():
        if len(keywords_roots[s]) > 1:  
            min_length = 1000
            for k in keywords_roots[s]:
                if len(k) < min_length:
                    key = k ; min_length = len(k)            
            category_keys.append(key)
            keywords_select[s] = key
        else:
            category_keys.append(list(keywords_roots[s])[0])
            keywords_select[s] = list(keywords_roots[s])[0]

    print("No of keywords in variable '{}': {}".format(column,len(category_keys)))
    return category_keys, keywords_roots, keywords_select, count_keywords


df_products = pd.DataFrame(df_new['Description'].unique()).rename(columns = {0:'Description'})
keywords, keywords_roots, keywords_select, count_keywords = keywords_inventory(df_products)
list_products = []
for k,v in count_keywords.items():
    list_products.append([keywords_select[k],v])
list_products.sort(key = lambda x:x[1], reverse = True)

liste = sorted(list_products, key = lambda x:x[1], reverse = True)
#_______________________________
plt.rc('font', weight='normal')
fig, ax = plt.subplots(figsize=(7, 25))
y_axis = [i[1] for i in liste[:125]]
x_axis = [k for k,i in enumerate(liste[:125])]
x_label = [i[0] for i in liste[:125]]
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 13)
plt.yticks(x_axis, x_label)
plt.xlabel("Nb. of occurences", fontsize = 18, labelpad = 10)
ax.barh(x_axis, y_axis, align = 'center')
ax = plt.gca()
ax.invert_yaxis()

plt.title("Words occurence",bbox={'facecolor':'k', 'pad':5}, color='w',fontsize = 25)
plt.show()

list_products = []
for k,v in count_keywords.items():
    word = keywords_select[k]
    if word in ['pink', 'blue', 'tag', 'green', 'orange']: continue
    if len(word) < 3 or v < 13: continue
    if ('+' in word) or ('/' in word): continue
    list_products.append([word, v])
#______________________________________________________    
list_products.sort(key = lambda x:x[1], reverse = True)
print('words retained:', len(list_products))

liste_products = df_new['Description'].unique()
X = pd.DataFrame()
for key, occurence in list_products:
    X.loc[:, key] = list(map(lambda x:int(key.upper() in x),liste_products))
threshold = [0, 1, 2, 3, 5, 10]
label_col = []
for i in range(len(threshold)):
    if i == len(threshold)-1:
        col = '.>{}'.format(threshold[i])
    else:
        col = '{}<.<{}'.format(threshold[i],threshold[i+1])
    label_col.append(col)
    X.loc[:, col] = 0

for i, prod in enumerate(liste_products):
    prix = df_new[df_new['Description'] == prod]['Price'].mean()
    j = 0
    while prix > threshold[j]:
        j+=1
        if j == len(threshold): break
    X.loc[i, label_col[j-1]] = 1
print("{:<8} {:<20} \n".format('range', 'nb. products') + 20*'-')
for i in range(len(threshold)):
    if i == len(threshold)-1:
        col = '.>{}'.format(threshold[i])
    else:
        col = '{}<.<{}'.format(threshold[i],threshold[i+1])    
    print("{:<10}  {:<20}".format(col, X.loc[:, col].sum()))

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans 

matrix = X.values
for n_clusters in range(3,10):
    kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=30)
    kmeans.fit(matrix)
    clusters = kmeans.predict(matrix)
    silhouette_avg = silhouette_score(matrix, clusters)
    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
n_clusters = 6
silhouette_avg = -1
#while silhouette_avg < 0.145:
kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=30)
kmeans.fit(matrix)
clusters = kmeans.predict(matrix)
silhouette_avg = silhouette_score(matrix, clusters)
    
    #km = kmodes.KModes(n_clusters = n_clusters, init='Huang', n_init=2, verbose=0)
    #clusters = km.fit_predict(matrix)
    #silhouette_avg = silhouette_score(matrix, clusters)
#print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
pd.Series(clusters).value_counts()

from wordcloud import WordCloud, STOPWORDS

liste = pd.DataFrame(liste_products)
liste_words = [word for (word, occurence) in list_products]

occurence = [dict() for _ in range(n_clusters)]

for i in range(n_clusters):
    liste_cluster = liste.loc[clusters == i]
    for word in liste_words:
        if word in ['art', 'set', 'heart', 'pink', 'blue', 'tag']: continue
        occurence[i][word] = sum(liste_cluster.loc[:, 0].str.contains(word.upper()))
#________________________________________________________________________

def random_color_func(word=None, font_size=None, position=None,
                      orientation=None, font_path=None, random_state=None):
    h = int(360.0 * tone / 255.0)
    s = int(100.0 * 255.0 / 255.0)
    l = int(100.0 * float(random_state.randint(70, 120)) / 255.0)
    return "hsl({}, {}%, {}%)".format(h, s, l)
#________________________________________________________________________
def make_wordcloud(liste, increment):
    ax1 = fig.add_subplot(4,2,increment)
    words = dict()
    trunc_occurences = liste[0:150]
    for s in trunc_occurences:
        words[s[0]] = s[1]
    #________________________________________________________
    wordcloud = WordCloud(width=1000,height=400, background_color='lightgrey', 
                          max_words=1628,relative_scaling=1,
                          color_func = random_color_func,
                          normalize_plurals=False)
    wordcloud.generate_from_frequencies(words)
    ax1.imshow(wordcloud, interpolation="bilinear")
    ax1.axis('off')
    plt.title('cluster nº{}'.format(increment-1))
#________________________________________________________________________
fig = plt.figure(1, figsize=(14,14))
color = [0, 160, 130, 95, 280, 40, 330, 110, 25]
for i in range(n_clusters):
    list_cluster_occurences = occurence[i]

    tone = color[i] # define the color of the words
    liste = []
    for key, value in list_cluster_occurences.items():
        liste.append([key, value])
    liste.sort(key = lambda x:x[1], reverse = True)
    make_wordcloud(liste, i+1)





new_df = df_new.reset_index()
print(new_df.head())
new_df.drop(columns=['index'],axis=1,inplace=True)
print(new_df.tail())

# Compute the maximum date to know the latest transaction date

max_date = max(final_df['InvoiceDate'])
print(max_date)
# Compute the difference between max date and transaction date

new_df['Difference'] = max_date - new_df['InvoiceDate']
print(new_df['Difference'])

new_df['Difference'] = new_df['Difference'].dt.days
print(new_df.head())


r = new_df.groupby('Customer ID')['Difference'].min()
print(r)
r = r.reset_index()
r.rename(columns= {'Difference':'Recency'}, inplace=True)
print(r.shape)
print(r.head())


f_m = new_df.groupby(['Customer ID']).agg({'Invoice': 'nunique','totalsales': 'sum'})
f_m.rename(columns={'Invoice': 'Frequency',
                         'totalsales': 'MonetaryValue'}, inplace=True)
f_m = f_m.reset_index()
print(f_m.shape)
print(f_m.head())

# Merging the two dataframes

rfm = pd.merge(r,f_m, on='Customer ID', how='inner')
print(rfm.head())
print(rfm.shape)
print(rfm.describe().T)
# Plot RFM distributions
plt.figure(figsize=(8,10))

# Plot distribution of R
plt.subplot(3, 1, 1);

sns.distplot(rfm['Recency'])

# Plot distribution of F
plt.subplot(3, 1, 2); 

sns.distplot(rfm['Frequency'])

# Plot distribution of M
plt.subplot(3, 1, 3);

sns.distplot(rfm['MonetaryValue'])

# Show the plot
plt.show()
# Rescaling the attributes using Standardisation

rfm_std = rfm[['Recency','Frequency','MonetaryValue']]

from sklearn import preprocessing

scaler = preprocessing.StandardScaler().fit(rfm_std)

rfm_scaled = scaler.transform(rfm_std)
print(rfm_scaled.shape)

rfm_scaled = pd.DataFrame(rfm_scaled)
rfm_scaled.columns = ['Recency','Frequency','MonetaryValue']
print(rfm_scaled.head())
print(rfm_scaled.describe().T)



from sklearn.cluster import KMeans 
# within cluster sum of squares - WCSS (inertia)
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init = 'k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(rfm_scaled)
    wcss.append(kmeans.inertia_)
plt.plot(wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


from sklearn.metrics import silhouette_score

# Silhouette analysis
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

for num_clusters in range_n_clusters:
    
    # intialise kmeans
    kmeans = KMeans(n_clusters=num_clusters, max_iter=100)
    kmeans.fit(rfm_scaled)
    
    cluster_labels = kmeans.labels_
    
    # silhouette score
    silhouette_avg = silhouette_score(rfm_scaled, cluster_labels)
    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))
sil = []
kmax = 10

# dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
for k in range(2, kmax+1):
  kmeans = KMeans(n_clusters = k).fit(rfm_scaled)
  labels = kmeans.labels_
  sil.append(silhouette_score(rfm_scaled, labels, metric = 'euclidean'))

plt.plot(sil)
plt.title('The Silhouette Method')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()
# Final model with k=3
kmeans = KMeans(n_clusters=3,init = 'k-means++', max_iter=300, n_init=10, random_state=0)
kmeans.fit(rfm_scaled)
# assign the label
rfm['Cluster_Id'] = kmeans.labels_
print(rfm.head())
rfm_cluster = rfm.groupby('Cluster_Id').agg({'Recency': ['max','mean', 'min'], 'Frequency': ['max','mean','min'],'MonetaryValue': ['max','mean','min'],'Customer ID' : 'count'})

print(rfm_cluster)

rfm_cluster.to_csv('G:/Machine learning projects/Customer segmentation-finalyearproject/rfm_cluster_summary.csv')
# Box plot to visualize Cluster Id vs Recency

sns.boxplot(x='Cluster_Id', y='Recency', data=rfm)
# Box plot to visualize Cluster Id vs Frequency

sns.boxplot(x='Cluster_Id', y='Frequency', data=rfm)
# Box plot to visualize Cluster Id vs MonetaryValue

sns.boxplot(x='Cluster_Id', y='MonetaryValue', data=rfm)

from mpl_toolkits import mplot3d

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8,6))
ax = plt.axes(projection="3d")

plt.show()
fig = plt.figure(figsize=(8,6))
ax = plt.axes(projection="3d")


x_points = rfm_scaled['Recency'].to_numpy(dtype='float')
y_points = rfm_scaled['Frequency'].to_numpy(dtype='float')
z_points = rfm_scaled['MonetaryValue'].to_numpy(dtype='float')
y = rfm['Cluster_Id'].to_numpy(dtype='float')
scatter = ax.scatter3D(x_points, y_points, z_points, c=y)

ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('MonetaryValue')
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="upper left", title="Cluster_Id")
ax.add_artist(legend1)

plt.show()
# Define rfm_level function
def rfm_level(df):
    if df['Cluster_Id'] == 1:
        return 'Likely to churn'
    elif df['Cluster_Id'] == 0:
        return 'Loyal or Potential'
    else:
        return 'Can\'t Loose Them'
# Create a new column Customer_type 
rfm['Customer_type'] = rfm.apply(rfm_level, axis=1)

print(rfm.head(10))
print(rfm.shape)
rfm.Customer_type.value_counts()
rfm.to_csv('G:/Machine learning projects/Customer segmentation-finalyearproject/RFM_groupby_customerId.csv')



#model building



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
predictdf= pd.read_csv('G:/Machine learning projects/Customer segmentation-finalyearproject/RFM_groupby_customerId.csv')
print(predictdf.head())
predictdf.drop(columns=['Unnamed: 0'],axis=1,inplace=True)
print(predictdf.shape)
print(predictdf.Customer_type.value_counts())


'''def cust_churn(x):
    if(x == 'Likely to churn'):
        return 1
    else :
        return 0'''
    
    
def cust_churn(x):
    if(x == 'Loyal or Potential'):
        return 0
    else :
        return 1
    
predictdf['Cust_churn'] = predictdf['Customer_type'].apply(cust_churn)
predictdf.head()
predictdf['Cust_churn'].value_counts()/predictdf.shape[0]

#X = df[['Recency','Frequency','MonetaryValue']]
#Y = df[['Cust_churn']]

from sklearn import preprocessing 

#label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 

#Encode labels in column 'customertype'. 

X1 = predictdf[['Recency','Frequency','MonetaryValue']]
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X1)
X_std = scaler.transform(X1)
X = pd.DataFrame(X_std) 
dftype= label_encoder.fit_transform(predictdf['Cust_churn']) 

Y=pd.DataFrame(dftype)

Y1=predictdf['Customer_type']

print(X.shape)
print(Y.shape)
X.rename(columns={0:'Recency',1:'Frequency',
                         2: 'Monetary_value'}, inplace=True)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=0)




from sklearn.naive_bayes import GaussianNB
model_Gaus = GaussianNB()
from sklearn.model_selection import cross_val_score
cross_val_score(GaussianNB(),X,Y,cv=4)
from sklearn.model_selection import GridSearchCV
clf_NB = GridSearchCV(GaussianNB(),{},cv=4, return_train_score=False,)
clf_NB.fit(X,Y)
clf_NB.cv_results_

nvresults= pd.DataFrame(clf_NB.cv_results_)
nvresults[['mean_test_score']]
NB_pred=clf_NB.predict(X_test)
#target_names = ['class 0-loyal or potential customers', 'class 1-cant-loose them','class 2-likely to churn']
#print(classification_report(y_test,NB_pred, target_names=target_names))
print(classification_report(y_test,NB_pred))

print("Naive Bayes model accuracy : ",clf_NB.score(X_test,y_test))
cm_nb = confusion_matrix(y_test,NB_pred)
plt.figure(figsize = (3,2))
sns.heatmap(cm_nb,annot=True,fmt='d',linewidths=.5)
plt.xlabel('Predicted')
plt.ylabel('Truth')



from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(random_state=0)
lr.fit(X,Y)

lr_pred=lr.predict(X_test)


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,lr_pred)
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
conf_matrix=confusion_matrix(y_test,lr_pred)
accuracy=accuracy_score(y_test,lr_pred)

print(accuracy)
print(classification_report(y_test,lr_pred))
print(conf_matrix)

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()
classifier.fit(X,Y)
## Prediction
random_pred=classifier.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
conf_matrix=confusion_matrix(y_test,random_pred)
accuracy=accuracy_score(y_test,random_pred)
print(conf_matrix,accuracy)
print(classification_report(y_test,random_pred))


from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier(criterion='entropy')
dtree.fit(X,Y)

dcpredict=dtree.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
conf_matrix=confusion_matrix(y_test,dcpredict)
accuracy=accuracy_score(y_test,dcpredict)

print(conf_matrix,accuracy)

print(classification_report(y_test,dcpredict))





'''import pickle
from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()

#Fitting model with trainig data
regressor.fit(X,Y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[0.125645,-0.407159 ,-0.132402 ]]))'''

#
import pickle
from sklearn.ensemble import RandomForestClassifier
ran_regressor = RandomForestClassifier()

#Fitting model with trainig data
ran_regressor.fit(X,Y)

# Saving model to disk
pickle.dump(ran_regressor, open('model1.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model1.pkl','rb'))
print(model.predict([[0.125645,-0.407159 ,-0.132402 ]]))


