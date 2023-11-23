import re

def clean_mimic3(text):
    text = text.lower().replace("\n"," ").replace("\r"," ")
    text = re.sub('dr\.','doctor',text)
    text = re.sub('m\.d\.','doctor',text)
    text = re.sub('admission date:','',text)
    text = re.sub('discharge date:','',text)
    text = re.sub('--|__|==','',text)
    return re.sub(r'  +', ' ', text)