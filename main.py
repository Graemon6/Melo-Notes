from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)


data = pd.read_csv("songpref1.csv")


X = data[['Rating','song_views']]
y = data['song_title'] 

model = DecisionTreeClassifier()
model.fit(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    rating = int(request.form['rating'])
    song_view = int(request.form['song_view'])
    
    predicted_song = model.predict([[rating, song_view]])[0]

    song_image = data[data['song_title'] == predicted_song]['images'].values[0]


    return render_template('index.html', 
        predicted_song=predicted_song,
        song_image=song_image
        )

if __name__ == '__main__':
    app.run(debug=True,port=3000)

# http://127.0.0.1:3000
