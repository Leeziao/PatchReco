import streamlit as st
from predict import predictItem

st.title(':small_blue_diamond: Security Patch Identification :small_blue_diamond:')
p = st.empty()
# p.metric('Result', 'Not Predicted')

button = st.button('Go Predict', use_container_width=True)

if button:
	print('Do Predict')
	patch = st.session_state['patch']
	with st.spinner('Predicting'):
		try:
			predictResult = predictItem('msg', patch)
		except Exception as e:
			print(e)
			predictResult = None
	print(predictResult)
	if predictResult is None: predictResult = 'Wrong Input Format'
	else: predictResult = 'Security Patch' if predictResult else 'Non Security Patch'
	st.metric('Result', predictResult)

text = st.text_area('Enter Your Patch Here', 'Some text', key='patch', height=600)
