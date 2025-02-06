#bard API
#https://www.youtube.com/watch?v=mGa68pSne8M

#token
#inspect > Application > Cookies > __Secure-1PSID
_BARD_API_KEY =

#from bardapi import Bard
#import os
#from dotenv import load_dotenv

#load_dotenv()
#token = os.getenv("BARD_API_KEY")
#token = BARD_API_KEY

#bard = Bard(token = token)

#result = bard.get_answer("What is EPF?")
#print(result)

import streamlit as st
from dotenv.main import load_dotenv
import os
load_dotenv()
st.title("BardAPI for Python")

#_BARD_API_KEY=os.environ['_BARD_API_KEY']
#import os
from bardapi import Bard

with st.container () :

    with st.spinner ( 'Wait till Bard gets the answers...' ) :
        input_text = "What is the largest state in the USA?"

        try :
            bard = Bard ( token = _BARD_API_KEY, timeout = 20 )
            response = (bard.get_answer ( input_text ) [ 'content' ])
            st.write ( response )
        except :
            st.error ( 'Please check if you have set the Cookie ' )