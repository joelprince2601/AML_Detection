@echo off
echo Setting up AML Detection System...

REM Create and activate virtual environment
python -m venv venv
call venv\Scripts\activate

REM Install requirements
pip install -r requirements.txt
pip install -e .

REM Run the Streamlit app
streamlit run aml_detection/app.py

pause 