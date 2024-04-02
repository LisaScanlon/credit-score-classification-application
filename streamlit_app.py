import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import joblib  # Import joblib to load the saved model

# Load the saved model
model = joblib.load('best_bagging_model.pkl')

# Page title
st.set_page_config(page_title='Credit Score Classification', page_icon='📊')
st.title('📊 Credit Score Classification')

with st.expander('About this app'):
  st.markdown('**What can this app do?**')
  st.info('This app allows you to learn more about your credit! After entering some basic information, this app will classify your credit as either Good, Standard, or Poor.')
  st.markdown('**How to use the app?**')
  st.warning('To engage with the app, 1. Click on each of the categories to learn more about what they mean, and how to calculate them. 2. Enter appropriate values for each of the categories. 3. Click SUBMIT to reveal your credit score classification!')

with st.expander('Learn More About Our Categories'):
  st.markdown('**Age**') 
  st.info('Your current age.')
  st.markdown('**Monthly In Hand Salary**') 
  st.info('The monthly base salary you receive, afteer taxes have been taken out.')
  st.markdown('**Interest Rate**') 
  st.info('The average interest rate on your credit cards.')
  st.markdown('**Delay from Due Date**') 
  st.info('The average number of days you are delayed on making payments. If you always pay on time, enter 0.')
  st.markdown('**Number of Delayed Payments**') 
  st.info('The average number of delayed payments you make in a given year.')
  st.markdown('**Changed Credit Limit**') 
  st.info('The percentage change in your credit card limit in a given year.')
  st.markdown('**Credit Mix**') 
  st.info('Your best estimate of the classification mix of your credit. Enter either Good, Standard, or Bad.')
  st.markdown('**Outstanding Debt**') 
  st.info('The remaining debt you have to pay in USD.')
  st.markdown('**Credit Utilization Ratio**') 
  st.info('The average utilization ratio of your credit cards. Calculated by _______.')
  st.markdown('**Credit History Age**') 
  st.info('How long you have been establishing credit (typically how long you have owned a credit card). Enter months as a decimal. For instance, if you have had credit for 2 years and 3 months, enter 2.25.')
  st.markdown('**Total EMI Per Month**') 
  st.info('Your monthly EMI payments in USD. This is calculated by _________.')
  st.markdown('**Amount Invested Monthly**') 
  st.info('The average monthly amount invested towards paying off your debts each month.')


st.subheader('Enter appropriate values for each of the following:')

# Load data
#df = pd.read_csv('data/movies_genres_summary.csv')
#df.year = df.year.astype('int')

# Input widgets
## Genres selection
#genres_list = df.genre.unique()
#genres_selection = st.multiselect('Select genres', genres_list, ['Action', 'Adventure', 'Biography', 'Comedy', 'Drama', 'Horror'])
#genres_selection = st.multiselect('Select genres', genres_list, ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
#       'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
 #      'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
  #     'Num_Credit_Inquiries', 'Credit_Mix', 'Outstanding_Debt',
  #     'Credit_Utilization_Ratio', 'Credit_History_Age', 'Total_EMI_per_month',
  #     'Amount_invested_monthly', 'Payment_Behaviour'])

# Genres selection
genres_list = [
'Age', 'Monthly_Inhand_Salary', 'Interest_Rate', 'Delay_from_due_date',
       'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Credit_Mix',
       'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Credit_History_Age',
       'Total_EMI_per_month', 'Amount_invested_monthly'
]

# Dictionary to store user inputs
user_inputs = {}

# Display input fields for each category
for genre in genres_list:
    user_input = st.text_input(f'Enter value for {genre}')
    user_inputs[genre] = user_input



# Submit button to initiate prediction
if st.button('SUBMIT'):
    # Convert user inputs to the appropriate data types
    user_inputs['Age'] = int(user_inputs['Age'])
    user_inputs['Monthly_Inhand_Salary'] = float(user_inputs['Monthly_Inhand_Salary'])
    user_inputs['Interest_Rate'] = int(user_inputs['Interest_Rate'])
    user_inputs['Delay_from_due_date'] = int(user_inputs['Delay_from_due_date'])
    user_inputs['Num_of_Delayed_Payment'] = int(user_inputs['Num_of_Delayed_Payment'])
    user_inputs['Changed_Credit_Limit'] = float(user_inputs['Changed_Credit_Limit'])
    user_inputs['Outstanding_Debt'] = float(user_inputs['Outstanding_Debt'])
    user_inputs['Credit_Utilization_Ratio'] = float(user_inputs['Credit_Utilization_Ratio'])
    user_inputs['Credit_History_Age'] = float(user_inputs['Credit_History_Age'])
    user_inputs['Total_EMI_per_month'] = float(user_inputs['Total_EMI_per_month'])
    user_inputs['Amount_invested_monthly'] = float(user_inputs['Amount_invested_monthly'])

    # Convert data types if necessary
    input_data = pd.DataFrame([user_inputs])

    print(input_data)

    # Convert DataFrame to NumPy array
    input_array = input_data.values

    # Make prediction
    prediction = model.predict(input_array)

    # Convert prediction to human-readable form
    prediction_mapping = {0: 'Good', 1: 'Standard', 2: 'Poor'}
    predicted_class = prediction_mapping[prediction[0]]

    # Display prediction
    st.subheader('Prediction:')
    st.write(f'The predicted credit score classification is: {predicted_class}')


# genres_selection = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
#       'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
#       'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
#       'Num_Credit_Inquiries', 'Credit_Mix', 'Outstanding_Debt',
#       'Credit_Utilization_Ratio', 'Credit_History_Age', 'Total_EMI_per_month',
#      'Amount_invested_monthly', 'Payment_Behaviour']


## Year selection
#year_list = df.year.unique()
#year_selection = st.slider('Select year duration', 1986, 2006, (2000, 2016))
#year_selection_list = list(np.arange(year_selection[0], year_selection[1]+1))

#df_selection = df[df.genre.isin(genres_selection) & df['year'].isin(year_selection_list)]
#reshaped_df = df_selection.pivot_table(index='year', columns='genre', values='gross', aggfunc='sum', fill_value=0)
#reshaped_df = reshaped_df.sort_values(by='year', ascending=False)


# Display DataFrame

#df_editor = st.data_editor(reshaped_df, height=212, use_container_width=True,
#                            column_config={"year": st.column_config.TextColumn("Year")},
#                            num_rows="dynamic")
#df_chart = pd.melt(df_editor.reset_index(), id_vars='year', var_name='genre', value_name='gross')

# Display chart
#chart = alt.Chart(df_chart).mark_line().encode(
#            x=alt.X('year:N', title='Year'),
#            y=alt.Y('gross:Q', title='Gross earnings ($)'),
#            color='genre:N'
#            ).properties(height=320)
#st.altair_chart(chart, use_container_width=True)'''


#Index(['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
#       'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
#       'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
#       'Num_Credit_Inquiries', 'Credit_Mix', 'Outstanding_Debt',
#       'Credit_Utilization_Ratio', 'Credit_History_Age', 'Total_EMI_per_month',
#       'Amount_invested_monthly', 'Payment_Behaviour'],
#      dtype='object')
