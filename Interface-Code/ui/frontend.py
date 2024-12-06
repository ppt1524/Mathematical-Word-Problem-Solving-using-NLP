import streamlit as st
import requests
import json

def main():
    st.title("Mathematical word problem solver")
    
    Word_Problem = st.text_input("Word Problem")
    
    input_data={
        "Word_Problem":Word_Problem
    }

    if st.button("Predict"):
        answer=requests.post(url="http://127.0.0.1:8000/predict",data=json.dumps(input_data))
        answer=answer.json()
        ans=answer['answer']
        st.success(f'The answer is: {ans}')
    

if __name__=='__main__':
    main()

