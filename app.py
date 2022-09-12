from pyexpat.errors import XML_ERROR_NOT_STANDALONE
from flask import Flask, redirect, url_for, request, render_template, jsonify
from flask import *
import pandas as pd
import numpy as np
import re
import json
from sklearn.model_selection import train_test_split 
from sklearn import cluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.svm import SVC
import pickle

app = Flask(__name__)


personal_type = pd.read_excel('personality_answers.xlsx',sheet_name='personal_type')
satisfaction_answers = pd.read_excel('personality_answers.xlsx',sheet_name='satisfaction')
personality_answers = pd.read_excel('personality_answers.xlsx',sheet_name='personality')

#reading question and answer
def json_answer(string): 
    
    return  json.loads(string)
satisfaction_answers['json_answer']= satisfaction_answers['answers'].apply(json_answer)


def convert_salary(string):
    
    
    values={
            'Less than 2000$': np.random.randint(1000, 2000), '2000-4000$':np.random.randint(2000, 4000),'4000-7000$': np.random.randint(4000, 7000),'More than 7000$':np.random.randint(7000, 8000)}
    return  values.get(string, np.random.randint(1000, 2000))
def convert_experience(string):
    
    
    values={'Less than3':np.random.randint(1, 3),
            '3-6':np.random.randint(3, 6),
            '6-10':np.random.randint(6, 10),
            'More than 10':np.random.randint(10, 11)}
    return  values.get(string, np.random.randint(1, 3))

def questionnaire_satisfaction(dataset_row,first_col,last_col):
    sum_value=0
    
    
    
    for question in dataset_row[first_col : last_col].index:
    
        
        answer=dataset_row[question]
        
        
        new_questionnairest=questionnaire(question,answer)
        
        sum_value=new_questionnairest.Gallup_Employee_Engagement(sum_value)
#         print('out',sum_value)
    result=''
   
    if (sum_value>=48 and sum_value<=60): 
        result='satisfied'
    elif (sum_value>=38 and sum_value<=47): 
        result='less_satisfied'
    elif (sum_value<38): 
        result='unsatisfied'  
  
    return result

def person_type(result_dict):
    person_type=''
    if result_dict['E']>=result_dict['I']:
        person_type+='E'
    if result_dict['E']<result_dict['I']:
        person_type+='I'
    if result_dict['S']<result_dict['N']:
        person_type+='N'    
    if result_dict['S']>=result_dict['N']:
        person_type+='S'
    if result_dict['T']<result_dict['F']:
        person_type+='F'
    if result_dict['F']<=result_dict['T']:
        person_type+='T'    
    if result_dict['J']<result_dict['P']:
        person_type+='P'
    if result_dict['P']<=result_dict['J']:
        person_type+='J'  
      
    return  person_type    
    
def questionnaire_maker(dataset_row,first_col,last_col):
    
    result_dict={'E':0,'I':0,'S':0,'N':0,'T':0,'F':0,'J':0,'P':0}
    

    #for question in dataset_row.loc[:,first : last].columns:
    for question in dataset_row[first_col : last_col].index:
        
        
        answer=dataset_row[question]
        
        new_questionnaire=questionnaire(question,answer)
        
        new_questionnaire.Myers_Briggs_analyze(result_dict)
        
    
    return person_type(result_dict)

class questionnaire:
    
    def __init__(self,question,answer):
        self.question=question
        self.answer=answer
#         self.sum_value=sum_value
        
#     def change(self,n):
#         print('n is',n)

#         self.sum_value+=n  
#         print('sfter change',self.sum_value)
        
    def Myers_Briggs_analyze(self,result_dict):
         
        
        select_row=personality_answers.loc[personality_answers['question']==self.question]
        
        for key,value in select_row['json_answer'][select_row.index[0]].items():
            
            
            
            if key== self.answer:
                

                for key1,value1 in select_row['json_answer'][select_row.index[0]].items():
                    
                    
                    if key1==value:
                        

                        result_dict[value1]+=int(key1)
                        
    
    def Gallup_Employee_Engagement(self,sum_value):
        
        select_row=satisfaction_answers.loc[satisfaction_answers['question']==self.question]
        
        for key,value in select_row['json_answer'][select_row.index[0]].items():
            
            

             if key== (self.answer.lstrip()).rstrip():
                

                    sum_value+=int(value)


        return sum_value        
    
               
                
#     (filter_name[0].lstrip()).rstrip()            
    
def remove_none_english_character(string):
    return re.sub(r"\(.*\)", "", string)


def make_dataframe(input_file):

    import json

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    Json_value=[]
    Json_value.append(data)
    Json_value_df = pd.DataFrame.from_dict(Json_value, orient='columns')
    return Json_value_df

def retuen_personality_type(Json_file):
   
    #dataframe = make_dataframe(Json_file)

    dataframe=Json_file

    
    column_names=list(dataframe.columns.values.tolist())
    column_names
    column_names_new={}
    column_names_list=[]
    
    for name in column_names:
        filter_name=name.split("\n")
        column_names_new[name]= (filter_name[0].lstrip()).rstrip()
        column_names_list.append((filter_name[0].lstrip()).rstrip())


    data_frame_new=dataframe.rename(columns = column_names_new, inplace = False)
    data_frame_new=data_frame_new.applymap(remove_none_english_character, na_action='ignore')
    first= 'At a party do you:'  
    last='Are you satisfied mostly:'
    personality_answers['json_answer']= personality_answers['answers'].apply(json_answer)
    data_frame_new['person_type']=data_frame_new.apply(lambda x:questionnaire_maker(x,first ,last),axis=1)
    final_dataframe=data_frame_new.copy()
    property_array=[]
    
    
    for row in personal_type.iterrows():
      
      for property in row[1]['properties'].split(','):
            property_array.append(property)
    def set_columns(row,y):
        select_property=personal_type.loc[personal_type['code']==row]
        for property in select_property['properties']:
            for index, element in enumerate(property.split(',')):
                if element==y:

                    return 1
        return 0 

    for pr in set(property_array):
        
        final_dataframe[pr]=final_dataframe['person_type'].apply(lambda x:set_columns(x,pr)).astype(int)
  
    first_st= 'I know what is expected of me at work.' 

    last_st='This last year, I have had opportunities at work to learn and grow.'
    
    final_dataframe['satisfied']=final_dataframe.apply(lambda x:questionnaire_satisfaction(x,first_st ,last_st),axis=1)
    
    final_dataframe1=final_dataframe.fillna('-')

    final_dataframe1['salary']=final_dataframe1['What is your salary rate?'].apply(convert_salary)
    final_dataframe1['experience_year']=final_dataframe1['How many years of work experience in do you have?'].apply(convert_experience)
    final_dataframe1.rename(columns={'1- How is your feeling about your current job?':'Feeling','What is your education degree?':'Education_Degree','Is your organization structure close to which group?':'Organization_Structure'}, inplace=True)
    final_dataframe1.loc[final_dataframe1['Organization_Structure']=='-','Organization_Structure']='Opposite of the first option '
    final_dataframe1.loc[final_dataframe1['Education_Degree']=='-','Education_Degree']='Master'
    final_dataframe1.loc[final_dataframe1['Organization_Structure']=='Opposite of the first option ','Organization_Structure']='Fix'
    final_dataframe1.loc[final_dataframe1['Organization_Structure']!='Fix','Organization_Structure']='Flexible'

    return final_dataframe1
def print_characteristic(personality_type):
    result=personal_type.loc[personal_type['code']==personality_type[0]]['properties']
    return '\n\n your personality type is: '+ str(personality_type.values) + '\n\n and your characteristics is:' + str(result.values)

    


def create_prediction_data(final_dataframe1):
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.feature_extraction.text import CountVectorizer
    from googletrans import Translator, constants
    from pprint import pprint
    translator = Translator()
    sample_data = pd.read_excel('sampleData.xlsx')

    final_dataframe1['Translate_Feeling']=final_dataframe1['Feeling'].apply(lambda x:translator.translate(x).text)
    sample_data['Translate_Feeling']=sample_data['Feeling'].apply(lambda x:translator.translate(x).text)
    
    vectorizer = CountVectorizer(stop_words='english')
 
    
    #final_dataframe2= final_dataframe1.drop(['Translate_Feeling','satisfied'], axis=1)
    
    X=final_dataframe1['Translate_Feeling']
    y=final_dataframe1['satisfied']
    sample_data=sample_data.append(final_dataframe1)
    sample_data = sample_data.reset_index(drop=True)

    X_test= vectorizer.fit_transform(sample_data['Translate_Feeling'])

    y_test = sample_data['satisfied']
    
    clf = SVC(kernel='linear', C=1).fit(X_test ,y_test)
     
    filename='Support_Vector'
    with open( filename, "wb") as file:
        pickle.dump(clf, file) 
        
   
    
    with open( 'Support_Vector', "rb") as file:
        pickle_model_sv = pickle.load(file)
        
    model_sv=pickle_model_sv     
    
    pred_sv=model_sv.predict(X_test)
    sample_data['pred_sv']=pred_sv
    
    result=sample_data.loc[sample_data['Translate_Feeling']==X[0]]['pred_sv']
    
    return 'based on your answer you are'+str(y.values) +'with your job\n\n and based on this project predicrion you are'+ str(result.values) +'  with your job.' 

      




@app.route('/')
def read_satisfaction_page ():
    return render_template('Satisfaction_MainPage.html')

@app.route('/display', methods=["GET", "POST"])
def display():
    if request.method == 'POST':
        
        result = request.form.to_dict(flat=False)
        
        Json_value_df = pd.DataFrame.from_dict(result, orient='columns')
 
        final_df=retuen_personality_type(Json_value_df)
        
        result_pesonality=print_characteristic(final_df['person_type'])
        resultx=create_prediction_data(final_df)
        
        
        
        
        return  render_template('display.html' ,text= resultx + result_pesonality)

    
    # if request.method == 'POST':
    #     return jsonify(request.form)
        
 
 # @app.route("/", methods=['GET', 'POST'])
# def get_input():
#     if request.method == 'POST':
#         json_param=request.form['Submit']
          
#         return redirect (url_for('run_pred',values=json_param))
# ########   
       

          
@app.route("/display") 
def return_values(values):
    
    #final_df=retuen_personality_type(values)
    return values
            
       

     


 
     

if __name__ == '__main__':
    app.run (host='127.0.0.1',port=5000,debug=True,threaded=True)