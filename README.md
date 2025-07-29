# AI-Powered-Earnings-Signal-Extractor
This AI-powered application analyzes NVIDIA's earnings call transcripts from the last four quarters to extract key business insights and sentiment signals. The app provides comprehensive analysis of management sentiment, Q&amp;A tone, strategic focuses, and quarter-over-quarter trends.

📊 NVIDIA Earnings Call Signal Extraction
🚀 Live Demo: (https://ai-powered-earnings-signal-extractor-4osvch2kmzukge2es4deze.streamlit.app/)

🧠 **Overview**
This AI-powered web application analyzes NVIDIA's earnings call transcripts from the last four quarters to surface four types of investment-relevant signals:

📈 Management Sentiment (prepared remarks)

🎤 Q&A Sentiment (analyst questions + executive responses)

🔁 Quarter-over-Quarter Tone Change

🧩 Strategic Focuses (business themes: e.g., AI growth, data center expansion)

The app uses natural language processing to extract and visualize insights for use by analysts, quants, and anyone tracking market-moving information.

✨ Features
✅ Management Sentiment Analysis – Extracts sentiment from executive prepared remarks
✅ Q&A Sentiment Tracking – Classifies tone during analyst Q&A sections
✅ Quarter-over-Quarter Trend Detection – Highlights tone shifts across transcripts
✅ Strategic Focus Identification – Extracts 3–5 key themes emphasized in each quarter
✅ Interactive Visualizations – Plotly charts for sentiment and trend exploration
✅ Clean, Responsive Dashboard – Built with Streamlit for a fast and intuitive UX


🧰 **Tech Stack**
Layer	Tools Used
Frontend	Streamlit
Backend	Python (3.10+)
Data	Pandas, NumPy
Visualization	Plotly
NLP / AI	TextBlob, NLTK, keyword frequency (LLM optional)
Deployment	Streamlit Cloud


**Local Testing**
If you'd like to run it locally:

**In your terminal:**
# Navigate to your project folder in Terminal
cd /path/username/nvidia-earnings-analysis or cd nvidia-earnings-project

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
It’ll open your browser



📂** **Project Structure****

nvidia-earnings-analysis/
app.py                 # Main Streamlit app
requirements.txt       # Required Python packages
README.md              # This file
data/                  # Raw transcript .txt files (Q1–Q4)


**Data Sources & Methodology**
T**ranscripts**

🔍 Source: Manually retrieved from Motley Fool using site search

🗓️ Coverage: NVIDIA earnings calls from Q1–Q4 FY2024

🧾 Format: Clean .txt files split by management section and Q&A

**AI / NLP Approach**

**Signal Type**	                          **Method Used**
Management Sentiment	              TextBlob sentiment analysis
Q&A Sentiment	                      TextBlob + parsing analyst/executive sections
QoQ Tone Comparison	                Delta in average polarity across quarters
Strategic Focus Themes	            Keyword frequency + NLTK phrase chunking



**Key Insights Extracted**
Sentiment Scores for both prepared and Q&A sections, per quarter

Tone Change Trends over time, visually and numerically

Strategic Themes extracted quarterly (e.g., AI, gaming, LLMs)

Interactive UI for analysts, quants, and business users



🧪 **Usage Flow**
📊 Overview Tab – Key metrics, executive summary

📈 Sentiment Analysis – Quarterly sentiment breakdowns

🧩 Strategic Focuses – Themes emphasized per quarter

🔁 Quarter Comparison – Tone shifts + side-by-side charts

📃 Raw Data – Sentiment scores + raw transcript references



🔧 **Development Note**s
Built and tested on macOS using VS Code and Streamlit Cloud

TextBlob chosen for simplicity and clarity in sentiment classification

Manual fallback for missing transcript data if API fails

Streamlit’s caching used for faster UI response



🚀 **Future Enhancements**
API-based live earnings transcript ingestion (e.g., via AlphaSense, Refinitiv)

OpenAI/Gemini for improved tone classification and semantic theme extraction

Additional companies beyond NVIDIA (S&P 500 support)

Downloadable CSV/Excel export of insights

Strategy suggestion layer (e.g., flag bullish tone increases for asset pricing)


🤝 **Contributing**
Fork the repo

Create a feature branch

Submit a pull request!
