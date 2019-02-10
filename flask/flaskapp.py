# import libraries
from flask import Flask
from flask import request, render_template
import json
import pandas as pd
app = Flask(__name__)
import numpy as np
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

# load in the data
df = pd.read_csv('features_file_for_flask.csv')

# clean the data by dropping the first unuseful column
df = df.iloc[:, 1:]

# Create dicionaries for an Arabic word and English translation

# create a dictionaary with web family name to convert it to dataframe family_name
family_name_dic = {'Alotaibi': 'العتيبي', 'Alonazi': 'العنزي', 'Almutairi':'المطيري',
                  'Algahtani': 'القحطاني', 'Alharbi': 'الحربي', 'Aldosari': 'الدوسري',
                  'Alsubaie': 'السبيعي', 'Alshahrani': 'الشهراني', 'Alshamri': 'الشمري',
                  'Alosaimi': 'العصيمي', 'Alshahri': 'الشهري', 'Alzahrani': 'الزهراني',
                  'Albishi': 'البيشي', 'Alomari': 'العمري', 'Albogami': 'البقمي',
                  'Alajmi': 'العجمي', 'Alrashidi': 'الرشيدي', 'Alghamdi': 'الغامدي',
                  'Alsulami': 'السلمي', 'Alamil': 'العميل', 'Algobaishi': 'الغبيشي',
                  'Alroaili': 'الرويلي', 'Albaloi': 'البلوي'}


# create a dictionaary with web city name to convert it to dataframe city_
city_dic = {'Riyadh': 'الرياض', 'Abha': 'ابها', 'Qassim': 'القصيم', 'Dammam': 'الدمام',
           'Tabuk': 'تبوك', 'Taif': 'الطائف', 'Bisha': 'بيشه', 'Wadi ad-Dawasir': 'وادي الدواسر',
           'Hail': 'حائل', 'Medina': 'المدينه', 'Jeddah': 'جدة'}


def train_data(df):
    '''
    this function would accept a dataframe to train, so that we can use the model
    in a new dataset
    '''
    # set X and y
    X = df.drop('country', axis=1)
    y = df['country']
    
    # balance the features
    # we'll balance our targets since it's imbalance
    X, y = SMOTE().fit_resample(X, y)
    
    # instantiate the model
    model = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                  decision_function_shape='ovr', degree=15, gamma='auto_deprecated',
                  kernel='linear', max_iter=-1, probability=True, random_state=None,
                  shrinking=True, tol=0.001, verbose=False)

    # fit the model
    model.fit(X, y);
    score = model.score(X, y)
    
    # return the trained model
    return model

# run the function
train_data(df)


one_row = df.iloc[:1, :].drop('country', axis=1)

def predict(row):
    '''
    this function would accept data as a user input
    convert them into a dataframe of one rwo
    then pass the dataframe into the model to predict the y which is the country
    '''
    # assign the trained model that we got from the moethod above 'train_model' to a variable
    # so that we can use it to do predictions
    trained_model = train_data(df)
    # assign predicted probability to a variable named pred_prob
    pred_prob = trained_model.predict_proba(one_row)
    # assign the probabilites predictions into a dataframe
    proba_df = pd.DataFrame(data=pred_prob, columns= trained_model.classes_)
    
    # return the predicted probabilites
    return proba_df
predict(one_row)


# create an empty row
df_model = df.drop('country', axis=1)
empty_row = pd.DataFrame(columns=df_model.columns)

# instantiate an empty list
l = []
# loop through the length of our empty dataframe
for i in list(empty_row.columns):
    # append a value of 0 to the list
    l.append(0)
    
# plug the first row with 0s 'the list'
empty_row.loc[1,:] = l



def feed_df(empty_row, family_name, city, avg_age, appearance, experience, good_with_kids, 
            good_with_elderly, religion, status):
    '''
    feed_df is a function that takes some inputs as a predefined dataframe columns
    and then it parse them into a dataframe and then it returns a one row dataframe.
    
    empty_row: the dataframe that you want to feed with data.
    
    family_name: family name of the client.
    
    city: they city that the client is from.
    
    ave_age: the average age of the worker as the client specified.
    
    appearance: a True, False values represent that client preference of the worker.
    
    experience: a True, False values represent that client preference of the worker.
    
    good_with_kids: a True, False values represent that client preference of the worker.
    
    good_with_elderly: a True, False values represent that client preference of the worker.
    
    religion: a string that represents that client preference of the worker religion. 
    
    status: a string that represents that client preference of the worker status. 
    '''
    
    if family_name == 'family name is not specified':
        empty_row['family_name_family_not_specified'] = True
    elif family_name in family_name_dic.keys():
        srt = 'family_name_{}'.format(family_name)
        empty_row[srt] = True

    if city == 'city is not specified':
        empty_row['city_not_specified'] = True
    elif city in city_dic.keys():
        srt = 'city_{}'.format(city)
        empty_row[srt] = True

    if avg_age == 'average age is not specified':
        empty_row['average_age_age_not_specified'] = True
    else:
        srt = 'average_age_{}'.format(avg_age)
        empty_row[srt] = True

    if appearance == True:
        empty_row['appearance'] = True
    else:
        empty_row['appearance'] = False

    if experience == 'experience_not_specified':
        empty_row['experience_not_specified'] = True
    elif experience == 'have_wokred':
        empty_row['have_wokred'] = True
    elif experience == 'no_experience':
        empty_row['no_experience'] = True

    if good_with_kids == True:
        empty_row['good_with_kids'] = True
    else:
        empty_row['good_with_kids'] = False

    if good_with_elderly == True:
        empty_row['good_with_elderly'] = True
    else:
        empty_row['good_with_elderly'] = False

    if religion == 'religion_not_specified':
        empty_row['religion_not_specified'] = True
    elif religion == 'muslim':
        empty_row['muslim'] = True
    elif religion == 'not_muslim':
        empty_row['not_muslim'] = True

    if status == 'status_not_specified':
        empty_row['status_not_specified'] = True
    elif status == 'marrid':
        empty_row['marrid'] = True
    elif status == 'not_marrid':
        empty_row['not_marrid'] = True
    
    return empty_row
@app.route('/')
def hello():
    return render_template('submit_page.html')


@app.route('/predict', methods=['POST'])
def predict_page():
    fname = request.form['family_name']
    city = request.form['city']
    average_age = request.form['average_age']
    appearance = request.form['appearance']
    experience = request.form['experience']
    good_with_kids = request.form['good_with_kids']
    good_with_elderly = request.form['good_with_elderly']
    religion = request.form['religion']
    status = request.form['status']
    
    row = feed_df(empty_row ,fname, city, average_age, appearance, experience, good_with_kids, good_with_elderly,
            religion, status)
    
#     result  = str(list(zip(predict(row).columns.tolist(), predict(row).values.tolist()[0])))
    
    # dump json instead for easier webservice consumption\
    print(predict(row))
    result = pd.Series(predict(row).loc[0,:]).to_dict()

    return render_template('web_page.html', **result)






if __name__ == '__main__':
  app.run(debug=True)

