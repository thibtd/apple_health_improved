make run: 
	uvicorn app.main:app --reload

make streamlit:
	streamlit run health_dashboard.py --server.maxUploadSize 5000