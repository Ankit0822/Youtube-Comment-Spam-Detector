from flask import Flask, render_template, url_for, request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

app=Flask(__name__)
file=open('model.pkl','rb')
clf=pickle.load(file)
file.close()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    df=pd.read_csv("data.csv")
    df_data=df[["CONTENT","CLASS"]]
    #Features and Labels
    df_x=df_data['CONTENT']
    df_y=df_data.CLASS
    # Extract Feature with CountVectorizer
    corpus=df_x
    cv=CountVectorizer()
    X=cv.fit_transform(corpus) #Fit tha data
    X_train,X_test,Y_train,Y_test=train_test_split(X,df_y,test_size=0.33,random_state=42)
    #Naive Bayes Classifier
    clf=MultinomialNB()
    clf.fit(X_train,Y_train)
    clf.score(X_test,Y_test)
    #Alternative Usage of Saved Models
    #n_model=open("model.pkl","rb")
    #clf=joblib.load(n_model)
    file=open('model.pkl','wb')
    pickle.dump(clf,file)
    file.close()

    if request.method=='POST':
        comment=request.form['comment']
        data=[comment]
        vect=cv.transform(data).toarray()
        my_prediction=clf.predict(vect)


    return render_template('result.html', prediction=my_prediction)





if __name__== '__main__':
    app.run(debug=True)