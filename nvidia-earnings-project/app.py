import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import numpy as np
from datetime import datetime
import os
import re
from collections import Counter
from textblob import TextBlob
import nltk

@st.cache_resource
def download_nltk_data():
    """Download required NLTK data with proper error handling"""
    import ssl
    
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    # List of required NLTK data
    nltk_downloads = [
        'punkt',
        'punkt_tab',
        'stopwords',
        'averaged_perceptron_tagger',
        'maxent_ne_chunker',
        'words',
        'vader_lexicon',
        'brown'
    ]
    
    for item in nltk_downloads:
        try:
            nltk.data.find(f'tokenizers/{item}')
        except LookupError:
            try:
                nltk.data.find(f'corpora/{item}')
            except LookupError:
                try:
                    nltk.data.find(f'taggers/{item}')
                except LookupError:
                    try:
                        nltk.data.find(f'chunkers/{item}')
                    except LookupError:
                        try:
                            print(f"Downloading {item}...")
                            nltk.download(item, quiet=True)
                        except Exception as e:
                            print(f"Failed to download {item}: {e}")
                            pass
    
    return True

# Initializing NLTK
download_nltk_data()


from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag

# Page configuration
st.set_page_config(
    page_title="NVIDIA Earnings Call Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #76b900;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .sentiment-positive { color: #28a745; font-weight: bold; }
    .sentiment-negative { color: #dc3545; font-weight: bold; }
    .sentiment-neutral { color: #6c757d; font-weight: bold; }
    .transcript-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)

class TranscriptProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.ai_keywords = [
            'ai', 'artificial intelligence', 'machine learning', 'deep learning',
            'neural network', 'generative ai', 'large language model', 'llm',
            'chatgpt', 'transformer', 'inference', 'training', 'gpu acceleration'
        ]
        self.datacenter_keywords = [
            'data center', 'datacenter', 'cloud', 'hyperscale', 'server',
            'infrastructure', 'h100', 'a100', 'hopper', 'ampere', 'compute'
        ]
        self.gaming_keywords = [
            'gaming', 'rtx', 'geforce', 'graphics', 'gpu', 'ray tracing',
            'dlss', 'gaming laptop', 'desktop', 'console'
        ]
        
    def split_transcript(self, text):
        """Split transcript into management remarks and Q&A sections"""
        # Find the Q&A section start
        qa_patterns = [
            r'Questions?\s*&?\s*Answers?:',
            r'Q&A',
            r'Question-and-Answer Session',
            r'We\'ll take our first question',
            r'We\'re now going to open the call for questions'
        ]
        
        qa_start = len(text)
        for pattern in qa_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                qa_start = min(qa_start, match.start())
        
        if qa_start < len(text):
            management_section = text[:qa_start]
            qa_section = text[qa_start:]
        else:
            # If no Q&A section found, assume first 60% is management
            split_point = int(len(text) * 0.6)
            management_section = text[:split_point]
            qa_section = text[split_point:]
            
        return management_section, qa_section
    
    def analyze_sentiment(self, text):
        """Analyze sentiment using TextBlob"""
        if not text.strip():
            return {'sentiment': 'neutral', 'average_score': 0.0, 'sentences': []}
        
        sentences = sent_tokenize(text)
        sentence_sentiments = []
        
        for sentence in sentences:
            if len(sentence.strip()) > 10:  # Skip very short sentences
                blob = TextBlob(sentence)
                polarity = blob.sentiment.polarity
                sentence_sentiments.append({
                    'text': sentence,
                    'score': polarity
                })
        
        if not sentence_sentiments:
            return {'sentiment': 'neutral', 'average_score': 0.0, 'sentences': []}
        
        avg_score = np.mean([s['score'] for s in sentence_sentiments])
        
        # Classify overall sentiment
        if avg_score > 0.1:
            sentiment = 'positive'
        elif avg_score < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
            
        return {
            'sentiment': sentiment,
            'average_score': avg_score,
            'sentences': sentence_sentiments
        }
    
    def extract_strategic_focuses(self, text):
        """Extract strategic focuses using keyword analysis and NLP"""
        focuses = []
        
        # AI & Machine Learning
        ai_mentions = self._count_keyword_mentions(text, self.ai_keywords)
        if ai_mentions['count'] > 0:
            focuses.append({
                'theme': 'AI & Machine Learning',
                'score': ai_mentions['count'],
                'details': ', '.join([f"{k} ({v})" for k, v in ai_mentions['breakdown'].items() if v > 0])
            })
        
        # Data Centering
        dc_mentions = self._count_keyword_mentions(text, self.datacenter_keywords)
        if dc_mentions['count'] > 0:
            focuses.append({
                'theme': 'Data Center',
                'score': dc_mentions['count'],
                'details': ', '.join([f"{k} ({v})" for k, v in dc_mentions['breakdown'].items() if v > 0])
            })
        
        
        gaming_mentions = self._count_keyword_mentions(text, self.gaming_keywords)
        if gaming_mentions['count'] > 0:
            focuses.append({
                'theme': 'Gaming',
                'score': gaming_mentions['count'],
                'details': ', '.join([f"{k} ({v})" for k, v in gaming_mentions['breakdown'].items() if v > 0])
            })
        
        # Extracting other themes using NLP
        other_themes = self._extract_business_themes(text)
        focuses.extend(other_themes)
        
        # Sorting by score and return top 5
        focuses.sort(key=lambda x: x['score'], reverse=True)
        return focuses[:5]
    
    def _count_keyword_mentions(self, text, keywords):
        """Count mentions of keywords in text"""
        text_lower = text.lower()
        breakdown = {}
        total_count = 0
        
        for keyword in keywords:
            count = len(re.findall(r'\b' + re.escape(keyword.lower()) + r'\b', text_lower))
            if count > 0:
                breakdown[keyword] = count
                total_count += count
        
        return {'count': total_count, 'breakdown': breakdown}
    
    def _extract_business_themes(self, text):
        """Extract additional business themes using NLP"""
        themes = []
        
        
        theme_keywords = {
            'Automotive': ['automotive', 'car', 'vehicle', 'autonomous', 'self-driving', 'orin'],
            'Software & Platform': ['software', 'platform', 'cuda', 'omniverse', 'enterprise'],
            'Networking': ['networking', 'infiniband', 'ethernet', 'mellanox', 'bluefield'],
            'Professional Visualization': ['professional', 'workstation', 'visualization', 'rendering']
        }
        
        for theme_name, keywords in theme_keywords.items():
            mentions = self._count_keyword_mentions(text, keywords)
            if mentions['count'] > 2:  # Threshold for inclusion
                themes.append({
                    'theme': theme_name,
                    'score': mentions['count'],
                    'details': ', '.join([f"{k} ({v})" for k, v in mentions['breakdown'].items() if v > 0])
                })
        
        return themes

def load_transcript_file(filepath):
    """Load transcript from file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error loading {filepath}: {str(e)}")
        return None

@st.cache_data
def process_transcripts():
    """Process all transcript files"""
    processor = TranscriptProcessor()
    quarters = ['Q1_2024', 'Q2_2024', 'Q3_2024', 'Q4_2024']
    transcripts = []
    
    for quarter in quarters:
        # Trying multiple possible paths for local vs cloud incase it doesnt work for a particular path
        possible_paths = [
            f'data/{quarter}.txt',  # For local (when running from nvidia-earnings-project/)
            f'nvidia-earnings-project/data/{quarter}.txt'  # For Streamlit Cloud
        ]
        
        text = None
        for filepath in possible_paths:
            if os.path.exists(filepath):
                text = load_transcript_file(filepath)
                break
        
        if text is None:
            
            st.warning(f"Could not load data files for {quarter}. Using sample data.")
            # Add sample data structure
            transcripts.append({
                'quarter': quarter,
                'raw_text': 'Sample data - transcript not available',
                'sentiment_scores': {
                    'management': {'sentiment': 'positive', 'average_score': 0.65},
                    'qa': {'sentiment': 'positive', 'average_score': 0.45},
                    'overall': {'sentiment': 'positive', 'average_score': 0.55}
                },
                'strategic_focuses': {
                    'strategic_focuses': [
                        {'theme': 'AI & Machine Learning', 'score': 15, 'details': 'Sample data'},
                        {'theme': 'Data Center', 'score': 12, 'details': 'Sample data'}
                    ]
                }
            })
            continue
        
        
        # Splitting into sections
        management_text, qa_text = processor.split_transcript(text)
        
        # Analyzing sentiment
        mgmt_sentiment = processor.analyze_sentiment(management_text)
        qa_sentiment = processor.analyze_sentiment(qa_text)
        
        # Calculating overall sentiment
        overall_score = (mgmt_sentiment['average_score'] + qa_sentiment['average_score']) / 2
        if overall_score > 0.1:
            overall_sentiment = 'positive'
        elif overall_score < -0.1:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'
        
        # Extracting strategic focuses
        strategic_focuses = processor.extract_strategic_focuses(text)
        
        transcripts.append({
            'quarter': quarter,
            'raw_text': text,
            'management_text': management_text,
            'qa_text': qa_text,
            'sentiment_scores': {
                'management': mgmt_sentiment,
                'qa': qa_sentiment,
                'overall': {'sentiment': overall_sentiment, 'average_score': overall_score}
            },
            'strategic_focuses': {
                'strategic_focuses': strategic_focuses
            }
        })
    
    # Calculating tone changes
    tone_changes = []
    for i in range(len(transcripts) - 1):
        current = transcripts[i]
        next_quarter = transcripts[i + 1]
        
        current_score = current['sentiment_scores']['overall']['average_score']
        next_score = next_quarter['sentiment_scores']['overall']['average_score']
        
        change = next_score - current_score
        
        if abs(change) > 0.1:
            significance = 'High'
        elif abs(change) > 0.05:
            significance = 'Medium'
        else:
            significance = 'Low'
        
        if change > 0:
            trend = 'More Positive'
        elif change < 0:
            trend = 'More Negative'
        else:
            trend = 'No Change'
        
        tone_changes.append({
            'from_quarter': current['quarter'],
            'to_quarter': next_quarter['quarter'],
            'sentiment_change': change,
            'trend': trend,
            'significance': significance
        })
    
    # Calculating summary stats
    mgmt_scores = [t['sentiment_scores']['management']['average_score'] for t in transcripts]
    qa_scores = [t['sentiment_scores']['qa']['average_score'] for t in transcripts]
    overall_scores = [t['sentiment_scores']['overall']['average_score'] for t in transcripts]
    
    # Aggregating strategic focuses
    all_focuses = {}
    for transcript in transcripts:
        for focus in transcript['strategic_focuses']['strategic_focuses']:
            theme = focus['theme']
            if theme in all_focuses:
                all_focuses[theme] += focus['score']
            else:
                all_focuses[theme] = focus['score']
    
    summary_stats = {
        'total_quarters': len(transcripts),
        'avg_management_sentiment': np.mean(mgmt_scores),
        'avg_qa_sentiment': np.mean(qa_scores),
        'avg_overall_sentiment': np.mean(overall_scores),
        'most_common_focuses': dict(sorted(all_focuses.items(), key=lambda x: x[1], reverse=True))
    }
    
    return {
        'transcripts': transcripts,
        'tone_changes': tone_changes,
        'summary_stats': summary_stats
    }

def display_sentiment_badge(sentiment, score):
    """Display sentiment with colored badge"""
    if sentiment == 'positive':
        return f'<span class="sentiment-positive">● {sentiment.upper()}</span> ({score:.3f})'
    elif sentiment == 'negative':
        return f'<span class="sentiment-negative">● {sentiment.upper()}</span> ({score:.3f})'
    else:
        return f'<span class="sentiment-neutral">● {sentiment.upper()}</span> ({score:.3f})'

def main():
    # Header
    st.markdown('<h1 class="main-header">🚀 NVIDIA Earnings Call Signal Extraction</h1>', unsafe_allow_html=True)
    
    # Loading and processing data
    with st.spinner('Processing NVIDIA earnings transcripts... This may take a moment.'):
        data = process_transcripts()
    
    transcripts = data['transcripts']
    tone_changes = data['tone_changes']
    summary_stats = data['summary_stats']
    
    
    st.sidebar.header("📊 Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis View:",
        ["📈 Overview", "🎭 Sentiment Analysis", "🎯 Strategic Focuses", "⚖️ Quarter Comparison", "📝 Raw Transcripts", "📊 Detailed Data"]
    )
    
    if page == "📈 Overview":
        st.header("📈 Executive Summary")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Quarters Analyzed", summary_stats['total_quarters'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Avg Management Sentiment", f"{summary_stats['avg_management_sentiment']:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Avg Q&A Sentiment", f"{summary_stats['avg_qa_sentiment']:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            if summary_stats['most_common_focuses']:
                top_focus = list(summary_stats['most_common_focuses'].keys())[0]
                st.metric("Top Strategic Focus", top_focus)
            else:
                st.metric("Top Strategic Focus", "N/A")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Sentiment trend chart
        st.subheader("📊 Sentiment Trends Over Time")
        
        quarters = [t['quarter'] for t in reversed(transcripts)]
        mgmt_scores = [t['sentiment_scores']['management']['average_score'] for t in reversed(transcripts)]
        qa_scores = [t['sentiment_scores']['qa']['average_score'] for t in reversed(transcripts)]
        overall_scores = [t['sentiment_scores']['overall']['average_score'] for t in reversed(transcripts)]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=quarters, y=mgmt_scores,
            mode='lines+markers',
            name='Management Sentiment',
            line=dict(color='#2E8B57', width=3),
            marker=dict(size=10)
        ))
        
        fig.add_trace(go.Scatter(
            x=quarters, y=qa_scores,
            mode='lines+markers',
            name='Q&A Sentiment',
            line=dict(color='#FF6347', width=3),
            marker=dict(size=10)
        ))
        
        fig.add_trace(go.Scatter(
            x=quarters, y=overall_scores,
            mode='lines+markers',
            name='Overall Sentiment',
            line=dict(color='#4169E1', width=3),
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            title="Sentiment Score Trends Across Quarters",
            xaxis_title="Quarter",
            yaxis_title="Sentiment Score",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Quarter-over-quarter changes
        if tone_changes:
            st.subheader("📈 Quarter-over-Quarter Tone Changes")
            
            for change in tone_changes:
                col1, col2, col3 = st.columns([3, 3, 4])
                
                with col1:
                    st.write(f"**{change['from_quarter']} → {change['to_quarter']}**")
                
                with col2:
                    if "Positive" in change['trend']:
                        st.write(f"🟢 {change['trend']}")
                    elif "Negative" in change['trend']:
                        st.write(f"🔴 {change['trend']}")
                    else:
                        st.write(f"⚪ {change['trend']}")
                
                with col3:
                    st.write(f"Change: **{change['sentiment_change']:+.3f}** ({change['significance']} significance)")
    
    elif page == "🎭 Sentiment Analysis":
        st.header("🎭 Detailed Sentiment Analysis")
        
        # Quarter selection
        selected_quarter = st.selectbox(
            "Select Quarter for Detailed Analysis:",
            [t['quarter'] for t in transcripts]
        )
        
        selected_data = next(t for t in transcripts if t['quarter'] == selected_quarter)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📝 Management Remarks")
            mgmt_sentiment = selected_data['sentiment_scores']['management']
            
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 10px; border-left: 4px solid #2E8B57;">
                <h4>Overall Sentiment</h4>
                {display_sentiment_badge(mgmt_sentiment['sentiment'], mgmt_sentiment['average_score'])}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("❓ Q&A Session")
            qa_sentiment = selected_data['sentiment_scores']['qa']
            
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 10px; border-left: 4px solid #FF6347;">
                <h4>Overall Sentiment</h4>
                {display_sentiment_badge(qa_sentiment['sentiment'], qa_sentiment['average_score'])}
            </div>
            """, unsafe_allow_html=True)
        
        # Comparison chart
        st.subheader("📊 Management vs Q&A Sentiment Comparison")
        
        comparison_data = {
            'Section': ['Management Remarks', 'Q&A Session'],
            'Sentiment Score': [
                selected_data['sentiment_scores']['management']['average_score'],
                selected_data['sentiment_scores']['qa']['average_score']
            ]
        }
        
        fig = px.bar(
            comparison_data,
            x='Section',
            y='Sentiment Score',
            title=f"Sentiment Comparison - {selected_quarter}",
            color='Sentiment Score',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🎯 Strategic Focuses":
        st.header("🎯 Strategic Focus Analysis")
        
        # Overall distribution
        st.subheader("📈 Strategic Theme Distribution")
        
        focus_data = summary_stats['most_common_focuses']
        
        if focus_data:
            fig = px.bar(
                x=list(focus_data.values()),
                y=list(focus_data.keys()),
                orientation='h',
                title="Most Common Strategic Themes Across All Quarters",
                color=list(focus_data.values()),
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Quarter breakdown
        st.subheader("📅 Strategic Focuses by Quarter")
        
        for transcript in transcripts:
            with st.expander(f"📊 {transcript['quarter']} Strategic Focuses"):
                focuses = transcript['strategic_focuses']['strategic_focuses']
                
                if focuses:
                    for i, focus in enumerate(focuses, 1):
                        st.markdown(f"""
                        **{i}. {focus['theme']}** (Score: {focus['score']})  
                        *Details: {focus['details']}*
                        """)
                else:
                    st.write("No strategic focuses identified for this quarter.")
    
    elif page == "⚖️ Quarter Comparison":
        st.header("⚖️ Quarter-by-Quarter Comparison")
        
        # Creating comparison table
        comparison_data = []
        for transcript in transcripts:
            mgmt_sent = transcript['sentiment_scores']['management']
            qa_sent = transcript['sentiment_scores']['qa']
            
            focuses = transcript['strategic_focuses']['strategic_focuses']
            top_focuses = [f['theme'] for f in focuses[:3]]
            
            comparison_data.append({
                'Quarter': transcript['quarter'],
                'Management Sentiment': f"{mgmt_sent['sentiment']} ({mgmt_sent['average_score']:.3f})",
                'Q&A Sentiment': f"{qa_sent['sentiment']} ({qa_sent['average_score']:.3f})",
                'Top Focus': top_focuses[0] if len(top_focuses) > 0 else 'N/A',
                'Second Focus': top_focuses[1] if len(top_focuses) > 1 else 'N/A',
                'Third Focus': top_focuses[2] if len(top_focuses) > 2 else 'N/A'
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True)
        
        # Tone changes visualization
        if tone_changes:
            st.subheader("📈 Sentiment Change Visualization")
            
            change_data = []
            for change in tone_changes:
                change_data.append({
                    'Transition': f"{change['from_quarter']} → {change['to_quarter']}",
                    'Change': change['sentiment_change'],
                    'Trend': change['trend']
                })
            
            df_changes = pd.DataFrame(change_data)
            
            fig = px.bar(
                df_changes,
                x='Transition',
                y='Change',
                color='Trend',
                title="Quarter-over-Quarter Sentiment Changes"
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "📝 Raw Transcripts":
        st.header("📝 Raw Transcript Viewer")
        
        # Quarter selection
        selected_quarter = st.selectbox(
            "Select Quarter to View:",
            [t['quarter'] for t in transcripts],
            key="transcript_selector"
        )
        
        selected_data = next(t for t in transcripts if t['quarter'] == selected_quarter)
        
        # Showing sections
        tab1, tab2, tab3 = st.tabs(["Full Transcript", "Management Remarks", "Q&A Session"])
        
        with tab1:
            st.subheader(f"Full Transcript - {selected_quarter}")
            if 'raw_text' in selected_data:
                st.text_area("", selected_data['raw_text'], height=400, key="full_transcript")
            else:
                st.write("Raw transcript not available.")
        
        with tab2:
            st.subheader(f"Management Remarks - {selected_quarter}")
            if 'management_text' in selected_data:
                st.text_area("", selected_data['management_text'], height=400, key="mgmt_transcript")
            else:
                st.write("Management section not available.")
        
        with tab3:
            st.subheader(f"Q&A Session - {selected_quarter}")
            if 'qa_text' in selected_data:
                st.text_area("", selected_data['qa_text'], height=400, key="qa_transcript")
            else:
                st.write("Q&A section not available.")
    
    elif page == "📊 Detailed Data":
        st.header("📊 Detailed Analysis Data")
        
        tab1, tab2, tab3 = st.tabs(["Summary Statistics", "Raw Sentiment Data", "Strategic Focus Data"])
        
        with tab1:
            st.subheader("📈 Summary Statistics")
            stats_df = pd.DataFrame({
                'Metric': ['Total Quarters', 'Avg Management Sentiment', 'Avg Q&A Sentiment', 'Avg Overall Sentiment'],
                'Value': [
                    summary_stats['total_quarters'],
                    f"{summary_stats['avg_management_sentiment']:.3f}",
                    f"{summary_stats['avg_qa_sentiment']:.3f}",
                    f"{summary_stats['avg_overall_sentiment']:.3f}"
                ]
            })
            st.dataframe(stats_df, use_container_width=True)
        
        with tab2:
            st.subheader("📊 Raw Sentiment Scores")
            sentiment_data = []
            for transcript in transcripts:
                sentiment_data.append({
                    'Quarter': transcript['quarter'],
                    'Management_Score': transcript['sentiment_scores']['management']['average_score'],
                    'QA_Score': transcript['sentiment_scores']['qa']['average_score'],
                    'Overall_Score': transcript['sentiment_scores']['overall']['average_score']
                })
            
            sentiment_df = pd.DataFrame(sentiment_data)
            st.dataframe(sentiment_df, use_container_width=True)
        
        with tab3:
            st.subheader("🎯 Strategic Focus Details")
            focus_details = []
            for transcript in transcripts:
                for focus in transcript['strategic_focuses']['strategic_focuses']:
                    focus_details.append({
                        'Quarter': transcript['quarter'],
                        'Theme': focus['theme'],
                        'Score': focus['score'],
                        'Details': focus['details']
                    })
            
            if focus_details:
                focus_df = pd.DataFrame(focus_details)
                st.dataframe(focus_df, use_container_width=True)
            else:
                st.write("No strategic focus data available.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        🚀 Built with Streamlit | NVIDIA Earnings Call Analysis | 
        📊 Data: Real NVIDIA earnings transcripts Q1-Q4 2024 |
        🤖 AI-Powered Analysis using TextBlob & NLTK
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
