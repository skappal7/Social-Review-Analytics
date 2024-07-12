import streamlit as st
import pandas as pd
import time
from google_play_scraper import Sort, reviews as gp_reviews, app as gp_app
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.util import ngrams
import altair as alt
from collections import Counter
from textblob import TextBlob
import nltk
import json

from word_tree_component import word_tree

nltk.download('stopwords')

# Custom CSS for branding and logo placement
st.markdown(
    """
    <style>
    .main-header img {
        display: block;
        margin: 0 auto;
    }
    .sidebar .sidebar-content {
        text-align: center;
    }
    .sidebar .sidebar-content img {
        border-radius: 50%;
    }
    .css-18e3th9 {
        font-family: 'Poppins', sans-serif;
    }
    .css-1avcm0n {
        font-family: 'Poppins', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add logos
st.image('logo.png', width=100)  # Ensure logo.png is in the same directory as your script
st.sidebar.image('logo.png', width=100)  # Ensure logo.png is in the same directory as your script

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'reviews_data' not in st.session_state:
    st.session_state['reviews_data'] = None
if 'labeled_data' not in st.session_state:
    st.session_state['labeled_data'] = None

# Helper function to download data as CSV
def download_csv(data, filename):
    csv = data.to_csv(index=False)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name=filename,
        mime='text/csv',
    )

# App 1: Google Play Store Review Scraper
def app1():
    st.title('Google Play Store Review Scraper')

    st.write("To find the app ID, go to the Google Play Store, search for the app, and copy the part of the URL after `id=` (e.g., for `https://play.google.com/store/apps/details?id=com.example.app`, the app ID is `com.example.app`).")
    st.write("[Click here for a guide on how to find the Google Play Store app ID](https://www.sociablekit.com/how-to-find-google-play-app-id/)")

    app_id = st.text_input('Enter the Google Play App ID:')
    num_reviews = st.slider('Select number of reviews to scrape', min_value=1, max_value=1000, step=1, value=100)
    sort_order = st.selectbox('Select the sort order of the reviews', ['Newest', 'Rating'])
    sort_order_map = {'Newest': Sort.NEWEST, 'Rating': Sort.RATING}
    sort_order_selected = sort_order_map[sort_order]

    min_rating, max_rating = None, None
    if sort_order_selected == Sort.RATING:
        min_rating, max_rating = st.slider('Select the rating range', min_value=1, max_value=5, value=(1, 5))

    if st.button('Scrape Reviews'):
        reviews = scrape_google_play(app_id, num_reviews, sort_order_selected, min_rating, max_rating)
        if reviews:
            app_details = fetch_google_play_app_details(app_id)
            st.write(f"App Title: {app_details['title']}")
            st.write(f"Installs: {app_details['installs']}")
            st.write(f"Average Rating: {app_details['score']}")
            st.write(f"Total Ratings: {app_details['ratings']}")
            st.write(f"Total Reviews: {app_details['reviews']}")
            st.write(f"Description: {app_details['description']}")
            st.write(f"Scraped {len(reviews)} reviews for App ID {app_id}")
            reviews_df = pd.DataFrame(reviews, columns=['Review'])
            st.dataframe(reviews_df)
            st.session_state['reviews_data'] = reviews_df  # Store reviews in session state for next step
            download_csv(reviews_df, 'google_play_reviews.csv')
        else:
            st.write("No reviews found or unable to scrape.")
    
    st.write("Note: Scraping reviews from certain websites may violate their terms of service. Use responsibly and ensure compliance with the website's policies.")

# App 2: Review Labeling and Categorization App
def app2():
    st.title('Review Labeling and Categorization App')

    if st.session_state['reviews_data'] is not None:
        reviews_df = st.session_state['reviews_data']

        # Convert the 'Review' column to string data type
        reviews_df['Review'] = reviews_df['Review'].astype(str)

        # Apply the classify_review function to the 'Review' column
        reviews_df['Label'] = reviews_df['Review'].apply(classify_review)

        # Apply the categorize_review function to the 'Review' column
        reviews_df['Category'] = reviews_df['Review'].apply(categorize_review)

        # Display the labeled and categorized reviews
        st.write(reviews_df)

        # Store labeled data for next step
        st.session_state['labeled_data'] = reviews_df

        # Allow the user to download the labeled and categorized reviews
        download_csv(reviews_df, 'labeled_categorized_reviews.csv')

        # Visualization: Label Distribution in Tabulated Form
        label_counts = reviews_df['Label'].value_counts().reset_index()
        label_counts.columns = ['Label', 'Count']
        label_counts['Percentage'] = (label_counts['Count'] / label_counts['Count'].sum() * 100).round(1).astype(str) + '%'
        label_counts['Bar'] = label_counts['Count'].apply(lambda x: f"<div style='width: {x/label_counts['Count'].max() * 100}px; background-color: #07B1FC;'>&nbsp;</div>")

        st.write("### Label Distribution")
        st.markdown(label_counts[['Label', 'Count', 'Percentage', 'Bar']].to_html(index=False, escape=False), unsafe_allow_html=True)

        # Visualization: Category Distribution in Tabulated Form
        category_counts = reviews_df['Category'].value_counts().reset_index()
        category_counts.columns = ['Category', 'Count']
        category_counts['Percentage'] = (category_counts['Count'] / category_counts['Count'].sum() * 100).round(1).astype(str) + '%'
        category_counts['Bar'] = category_counts['Count'].apply(lambda x: f"<div style='width: {x/category_counts['Count'].max() * 100}px; background-color: #07B1FC;'>&nbsp;</div>")

        st.write("### Category Distribution")
        st.markdown(category_counts[['Category', 'Count', 'Percentage', 'Bar']].to_html(index=False, escape=False), unsafe_allow_html=True)

    else:
        st.write("No review data available. Please scrape reviews first.")

# App 3: Text and Sentiment Preliminary Analysis
def app3():
    st.title('Text and Sentiment Preliminary Analysis')

    # Load stopwords
    stop_words = set(stopwords.words('english'))

    # Sidebar for input parameters
    st.sidebar.header("Input Parameters")
    exclude_words = st.sidebar.text_input("Words to Exclude (comma separated)", "")
    min_freq = st.sidebar.number_input("Minimum Frequency", value=2, min_value=1)
    max_words = st.sidebar.number_input("Maximum Words", value=200, min_value=1)

    # Function to clean text
    def clean_text(text):
        if isinstance(text, str):
            text = text.lower()
            text = ''.join([c for c in text if c not in ('!', '.', ':', ',', '?')])
            return text
        else:
            return ""

    # Function to analyze text data
    def analyze_text(data, column):
        data[column] = data[column].apply(clean_text)
        words = ' '.join(data[column]).split()
        words = [word for word in words if word not in stop_words]
        if exclude_words:
            exclude = exclude_words.split(',')
            words = [word for word in words if word not in exclude]
        return words

    # Function to perform sentiment analysis
    def sentiment_analysis(data, column):
        data['sentiment'] = data[column].apply(lambda x: TextBlob(x).sentiment.polarity if x else 0)
        data['sentiment_type'] = data['sentiment'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))
        return data

    # Plot word cloud
    def plot_wordcloud(words):
        wordcloud = WordCloud(width=800, height=400, max_words=max_words, background_color='white').generate(' '.join(words))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

    # Plot sentiment analysis
    def plot_sentiment(data):
        sentiment_counts = data['sentiment_type'].value_counts().reset_index()
        sentiment_counts.columns = ['sentiment', 'count']
        chart = alt.Chart(sentiment_counts).mark_bar().encode(
            x='sentiment',
            y='count',
            color='sentiment'
        ).properties(
            title="Sentiment Analysis"
        )
        st.altair_chart(chart, use_container_width=True)

    # Plot n-grams
    def plot_ngrams(words, n):
        n_grams = ngrams(words, n)
        n_grams_freq = FreqDist(n_grams).most_common(50)
        n_grams_df = pd.DataFrame(n_grams_freq, columns=['ngram', 'count'])
        n_grams_df['ngram'] = n_grams_df['ngram'].apply(lambda x: ' '.join(x))
        chart = alt.Chart(n_grams_df).mark_bar().encode(
            x=alt.X('ngram', sort='-y'),
            y='count',
            tooltip=['ngram', 'count']
        ).properties(
            title=f"Top 50 {'Bigrams' if n == 2 else 'Trigrams'}"
        )
        st.altair_chart(chart, use_container_width=True)

    # Plot top positive and negative words
    def plot_top_words(data, sentiment):
        words = data[data['sentiment_type'] == sentiment]['Review'].str.cat(sep=' ').split()
        words = [word for word in words if word not in stop_words]
        words_freq = Counter(words).most_common(20)
        words_df = pd.DataFrame(words_freq, columns=['word', 'count'])
        chart = alt.Chart(words_df).mark_bar().encode(
            x=alt.X('word', sort='-y'),
            y='count',
            color=alt.value('green' if sentiment == 'Positive' else 'red')
        ).properties(
            title=f"Top 20 {sentiment} Words"
        )
        st.altair_chart(chart, use_container_width=True)

    # Main panel for displaying analysis
    if st.session_state['labeled_data'] is not None:
        data = st.session_state['labeled_data']
        words = analyze_text(data, 'Review')
        sentiment_data = sentiment_analysis(data, 'Review')

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Word Cloud", "Text Analytics", "Sentiment Analysis", "N-grams", "Top Words"])

        with tab1:
            st.header("Word Cloud")
            plot_wordcloud(words)

        with tab2:
            st.header("Text Analytics")
            text_freq = pd.DataFrame(FreqDist(words).most_common(), columns=['word', 'count'])
            st.dataframe(text_freq)

        with tab3:
            st.header("Sentiment Analysis")
            plot_sentiment(sentiment_data)
            st.dataframe(sentiment_data[['Review', 'sentiment', 'sentiment_type']])

        with tab4:
            st.header("N-grams")
            plot_ngrams(words, 2)
            plot_ngrams(words, 3)

        with tab5:
            st.header("Top Words")
            plot_top_words(sentiment_data, 'Positive')
            plot_top_words(sentiment_data, 'Negative')

        # Download button
        csv = sentiment_data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results", data=csv, file_name="results.csv", mime="text/csv")
    else:
        st.info("No labeled data available. Please label reviews first.")

# App 4: Word Tree Analysis with D3.js
def app4():
    st.title('Word Tree Analysis')

    if st.session_state['labeled_data'] is not None:
        data = st.session_state['labeled_data']

        # Prepare hierarchical data for D3.js
        def prepare_word_tree_data(data):
            tree = {"name": "root", "children": []}
            for _, row in data.iterrows():
                review = row['Review']
                words = review.split()
                current_level = tree
                for word in words:
                    match = next((child for child in current_level["children"] if child["name"] == word), None)
                    if match:
                        current_level = match
                    else:
                        new_node = {"name": word, "children": []}
                        current_level["children"].append(new_node)
                        current_level = new_node
                current_level["label"] = row['Label']
                current_level["category"] = row['Category']
                current_level["sentiment"] = row['sentiment']
            return tree

        tree_data = prepare_word_tree_data(data)

        # Render the word tree component
        word_tree(tree_data)

    else:
        st.write("No labeled data available. Please label reviews first.")

# Cover page with login
def cover_page():
    st.title("Welcome to RevAI Fusion 360 💽")
    st.subheader("Please login to continue")

    # Login form
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "humach" and password == "password":
            st.success("Login successful!")
            st.session_state['logged_in'] = True
        else:
            st.error("Invalid username or password")

# Main function to run the app
def main():
    if not st.session_state['logged_in']:
        cover_page()
    else:
        st.sidebar.title('Navigation')
        app_selection = st.sidebar.radio('Go to', ['Review Scraper', 'Review Labeler', 'Text2Insights', 'Word Tree Analysis'])

        if app_selection == 'Review Scraper':
            app1()
        elif app_selection == 'Review Labeler':
            app2()
        elif app_selection == 'Text2Insights':
            app3()
        elif app_selection == 'Word Tree Analysis':
            app4()

# Helper Functions for App1
def scrape_google_play(app_id, num_reviews=100, sort_order=Sort.NEWEST, min_rating=None, max_rating=None):
    all_reviews = []
    next_token = None
    while len(all_reviews) < num_reviews:
        current_reviews, token = gp_reviews(
            app_id,
            lang='en',
            country='us',
            sort=sort_order,
            count=min(num_reviews - len(all_reviews), 100),
            filter_score_with=None if min_rating is None and max_rating is None else list(range(min_rating, max_rating + 1)),
            continuation_token=next_token
        )
        all_reviews.extend(current_reviews)
        next_token = token
        if not next_token:
            break
        time.sleep(2)  # Add delay to avoid rate limiting
    reviews_text = [review['content'] for review in all_reviews]
    return reviews_text

def fetch_google_play_app_details(app_id):
    app_details = gp_app(app_id)
    return {
        'title': app_details['title'],
        'installs': app_details['installs'],
        'score': app_details['score'],
        'ratings': app_details['ratings'],
        'reviews': app_details['reviews'],
        'description': app_details['description']
    }

# Helper Functions for App2
def classify_review(review):
    process_keywords = [
        'smooth', 'seamless', 'process', 'efficient', 'straightforward', 'hassle-free', 'user-friendly',
        'convenient', 'fast', 'prompt', 'timely', 'organized', 'structured', 'streamlined',
        'consistent', 'transparent', 'informative', 'clear', 'secure', 'reliable', 'accurate',
        'personalized', 'tailored', 'flexible', 'adaptable', 'responsive', 'accessible',
        'compliant', 'ethical', 'trustworthy', 'confidential', 'supportive', 'value-added',
        'cost-effective', 'comprehensive', 'end-to-end', 'customer-focused', 'needs-based',
        'intuitive', 'logical', 'well-documented', 'standardized', 'repeatable', 'scalable',
        'agile', 'data-driven', 'insightful', 'proactive', 'preventive', 'continuous improvement',
        'innovative', 'transformative', 'sustainable', 'resilient', 'risk-aware', 'compliant',
        'auditable', 'traceable', 'measurable', 'transparent', 'accountable', 'collaborative',
        'workflow', 'procedure', 'protocol', 'methodology', 'approach', 'technique', 'strategy',
        'framework', 'lifecycle', 'transition', 'migration', 'implementation', 'execution', 'delivery',
        'simple', 'easy', 'slow', 'disorganized', 'inefficient', 'confusing', 'outdated', 
        'cumbersome', 'rigid', 'error-prone', 'bureaucratic', 'redundant', 'inconsistent', 'unresponsive'
    ]
    technology_keywords = [
        'intuitive', 'invalid', 'pin', 'user-friendly', 'modern', 'advanced', 'innovative', 'cutting-edge',
        'reliable', 'efficient', 'fast', 'responsive', 'secure', 'robust', 'stable',
        'integrated', 'seamless', 'accessible', 'mobile-friendly', 'omnichannel', 'personalized',
        'customizable', 'interactive', 'intelligent', 'automated', 'self-service', 'convenient',
        'informative', 'transparent', 'real-time', 'data-driven', 'analytical', 'insightful',
        'scalable', 'flexible', 'adaptable', 'future-proof', 'compatibility', 'interoperability',
        'cloud-based', 'virtualized', 'resilient', 'fault-tolerant', 'privacy-compliant',
        'contextual', 'predictive', 'cognitive', 'conversational', 'natural language', 'voice-enabled',
        'multi-modal', 'immersive', 'augmented', 'virtual', 'blockchain-powered', 'distributed',
        'decentralized', 'sustainable', 'energy-efficient', 'eco-friendly', 'ethical', 'transparent',
        'accountable', 'inclusive', 'accessible', 'register', 'buggy', 'glitchy', 'glitch',
        'app', 'application', 'mobile app', 'android app', 'ios app', 'website', 'web app', 'portal',
        'platform', 'interface', 'dashboard', 'chatbot', 'virtual assistant', 'voice assistant',
        'bug', 'download', 'attachment', 'load', 'crashing', 'install', 'reinstall', 'code', 'error',
        'rin', 'hack', 'hacking', 'scam', 'login', 'log', 'notification', 'track', 'location', 'offline',
        'pdf', 'image', 'upload', 'photo', 'click', 'slow', 'unresponsive', 'insecure', 'complicated',
        'hard-to-use', 'malfunction', 'fail', 'downtime'
    ]
    people_keywords = [
        'friendly', 'knowledgeable', 'helpful', 'patient', 'attentive', 'empathetic', 'responsive',
        'professional', 'courteous', 'communicative', 'efficient', 'dedicated', 'well-trained',
        'reliable', 'accessible', 'multilingual', 'personable', 'understanding', 'supportive',
        'experienced', 'reassuring', 'proactive', 'approachable', 'caring', 'compassionate',
        'trustworthy', 'accommodating', 'service-minded', 'respectful', 'polite', 'empowering',
        'motivating', 'encouraging', 'engaging', 'clear communication', 'problem-solving',
        'situational awareness', 'emotional intelligence', 'customer-centric', 'culturally aware',
        'adaptive', 'resilient', 'collaborative', 'team-oriented', 'passionate', 'committed',
        'accountable', 'ethical', 'transparent', 'authentic', 'they', 'he', 'man', 'lady', 'she',
        'rude', 'incompetent', 'unhelpful', 'impatient', 'inattentive', 'insensitive', 'unresponsive',
        'unprofessional', 'discourteous', 'uncommunicative', 'inefficient', 'careless', 'untrained',
        'unreliable', 'inaccessible', 'uninformed', 'impersonal', 'unsupportive', 'inexperienced',
        'dismissive', 'passive', 'unapproachable', 'uncaring', 'untrustworthy', 'inflexible',
        'condescending', 'demotivating', 'discouraging', 'disengaging', 'unclear communication',
        'problem-ignoring', 'situationally unaware', 'emotionally unintelligent', 'self-centered',
        'culturally insensitive', 'rigid', 'fragile', 'uncooperative', 'individualistic', 'apathetic',
        'uncommitted', 'unaccountable', 'unethical', 'opaque', 'inauthentic', 'customer service', 'agent'
    ]

    # Convert review to string and then to lowercase for case-insensitive matching
    review = str(review).lower()

    # Check for process-related keywords
    for keyword in process_keywords:
        if keyword in review:
            return "Process"

    # Check for technology-related keywords
    for keyword in technology_keywords:
        if keyword in review:
            return "Technology"

    # Check for people-related keywords
    for keyword in people_keywords:
        if keyword in review:
            return "People"

    # If no keywords found, return "Unknown"
    return "Unknown"

def categorize_review(review):
    billing_keywords = ['invoice', 'payment', 'bill', 'charge', 'refund', 'credit', 'debit', 'balance', 'overdue', 'fee', 'statement', 'account', 'transaction', 'receipt', 'pay', 'finance', 'cost', 'expense', 'price', 'amount', 'due', 'overcharge', 'undercharge', 'billing cycle']
    technical_support_keywords = ['tech support', 'technical', 'troubleshoot', 'error', 'issue', 'bug', 'glitch', 'malfunction', 'repair', 'fix', 'installation', 'setup', 'connectivity', 'network', 'software', 'hardware', 'reboot', 'reset', 'upgrade', 'update', 'compatibility', 'diagnostics', 'assistance', 'support', 'help desk']
    account_management_keywords = ['account', 'profile', 'login', 'password', 'username', 'registration', 'sign up', 'sign in', 'subscription', 'renewal', 'cancel', 'deactivate', 'activate', 'update', 'modify', 'personal information', 'user ID', 'credentials', 'security', 'verification', 'access', 'account settings', 'account recovery']
    product_information_keywords = ['product', 'feature', 'specification', 'details', 'model', 'version', 'variant', 'description', 'availability', 'stock', 'price', 'cost', 'warranty', 'guarantee', 'manual', 'guide', 'brochure', 'catalog', 'options', 'selection', 'usage', 'demo', 'sample']
    service_inquiry_keywords = ['service', 'inquiry', 'information', 'details', 'availability', 'schedule', 'appointment', 'booking', 'reservation', 'timing', 'location', 'facility', 'feature', 'benefit', 'offer', 'package', 'plan', 'subscription', 'contract', 'agreement', 'terms', 'service status', 'service update']
    complaints_feedback_keywords = ['complaint', 'issue', 'problem', 'dissatisfaction', 'feedback', 'suggestion', 'review', 'criticism', 'concern', 'trouble', 'negative experience', 'poor service', 'bad', 'unhappy', 'unsatisfied', 'resolved', 'unresolved', 'escalate', 'escalation', 'grievance', 'refund', 'compensation', 'dispute']
    sales_renewals_keywords = ['sales', 'purchase', 'buy', 'order', 'renew', 'renewal', 'contract', 'agreement', 'deal', 'discount', 'offer', 'promo', 'pricing', 'cost', 'quote', 'billing', 'payment', 'subscription', 'trial', 'demo', 'upgrade', 'conversion', 'checkout', 'transaction']
    shipping_delivery_keywords = ['shipping', 'delivery', 'dispatch', 'shipment', 'package', 'courier', 'tracking', 'track', 'status', 'estimated delivery', 'delay', 'lost', 'damaged', 'return', 'replacement', 'logistics', 'freight', 'parcel', 'order', 'receive', 'warehouse', 'logistics', 'carrier']
    returns_exchanges_keywords = ['return', 'exchange', 'replacement', 'refund', 'credit', 'policy', 'terms', 'conditions', 'process', 'procedure', 'defective', 'damaged', 'wrong item', 'incorrect', 'sent', 'receive', 'receipt', 'product', 'package', 'return label', 'authorization', 'approval', 'inspection']
    general_inquiry_keywords = ['general', 'inquiry', 'question', 'ask', 'information', 'details', 'assistance', 'help', 'support', 'contact', 'reach out', 'need', 'want', 'clarify', 'understand', 'query', 'explore', 'guidance', 'advice', 'basic', 'common', 'general service', 'support']

    # Convert review to string and then to lowercase for case-insensitive matching
    review = str(review).lower()

    # Check for billing-related keywords
    for keyword in billing_keywords:
        if keyword in review:
            return "Billing and Payments"

    # Check for technical support-related keywords
    for keyword in technical_support_keywords:
        if keyword in review:
            return "Technical Support"

    # Check for account management-related keywords
    for keyword in account_management_keywords:
        if keyword in review:
            return "Account Management"

    # Check for product information-related keywords
    for keyword in product_information_keywords:
        if keyword in review:
            return "Product Information"

    # Check for service inquiry-related keywords
    for keyword in service_inquiry_keywords:
        if keyword in review:
            return "Service Inquiry"

    # Check for complaints and feedback-related keywords
    for keyword in complaints_feedback_keywords:
        if keyword in review:
            return "Complaints and Feedback"

    # Check for sales and renewals-related keywords
    for keyword in sales_renewals_keywords:
        if keyword in review:
            return "Sales and Renewals"

    # Check for shipping and delivery-related keywords
    for keyword in shipping_delivery_keywords:
        if keyword in review:
            return "Shipping and Delivery"

    # Check for returns and exchanges-related keywords
    for keyword in returns_exchanges_keywords:
        if keyword in review:
            return "Returns and Exchanges"

    # Check for general inquiry-related keywords
    for keyword in general_inquiry_keywords:
        if keyword in review:
            return "General Inquiry"

    # If no keywords found, return "Unknown"
    return "Unknown"

if __name__ == '__main__':
    main()
