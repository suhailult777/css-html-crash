import tweepy
import re
from typing import List, Dict, Union
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class TweetPreprocessor:
    @staticmethod
    def clean_tweet(tweet: str) -> str:
        """
        Preprocess tweet text by removing URLs, mentions, hashtags, and special characters.
        """
        # Remove URLs
        tweet = re.sub(r'http\S+|www\S+', '', tweet)
        # Remove mentions
        tweet = re.sub(r'@\w+', '', tweet)
        # Remove hashtags
        tweet = re.sub(r'#\w+', '', tweet)
        # Remove special characters and numbers
        tweet = re.sub(r'[^\w\s]', '', tweet)
        # Remove extra whitespace
        tweet = ' '.join(tweet.split())
        return tweet.lower()

class TwitterClient:
    def __init__(self, bearer_token: str):
        """Initialize Twitter client with authentication."""
        self.client = tweepy.Client(
            bearer_token=bearer_token,
            wait_on_rate_limit=True
        )
        
    def fetch_tweets(self, query: str, max_results: int = 100) -> List[Dict]:
        """
        Fetch tweets with error handling and rate limiting.
        """
        tweets = []
        try:
            response = self.client.search_recent_tweets(
                query=query,
                max_results=max_results,
                tweet_fields=['created_at', 'lang', 'context_annotations']
            )
            if response.data:
                for tweet in response.data:
                    tweets.append({
                        'id': tweet.id,
                        'text': tweet.text,
                        'created_at': tweet.created_at,
                        'lang': tweet.lang
                    })
        except tweepy.TweepyException as e:
            print(f"Error fetching tweets: {str(e)}")
        return tweets

class SentimentAnalyzer:
    def __init__(self):
        """Initialize sentiment analyzers."""
        self.preprocessor = TweetPreprocessor()
        self.vader = SentimentIntensityAnalyzer()
        
    def analyze_text(self, text: str, include_confidence: bool = True) -> Dict[str, Union[float, str]]:
        """
        Analyze text sentiment using both TextBlob and VADER.
        Returns sentiment scores and confidence metrics.
        """
        # Preprocess the text
        cleaned_text = self.preprocessor.clean_tweet(text)
        
        # TextBlob analysis
        blob = TextBlob(cleaned_text)
        textblob_sentiment = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        # VADER analysis
        vader_scores = self.vader.polarity_scores(cleaned_text)
        
        # Calculate weighted average and confidence
        weighted_score = (textblob_sentiment + vader_scores['compound']) / 2
        confidence_score = self._calculate_confidence(
            textblob_sentiment,
            vader_scores['compound'],
            textblob_subjectivity
        )
        
        result = {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'textblob_score': round(textblob_sentiment, 3),
            'vader_score': round(vader_scores['compound'], 3),
            'weighted_score': round(weighted_score, 3),
            'sentiment': self._get_sentiment_label(weighted_score),
        }
        
        if include_confidence:
            result.update({
                'confidence_score': round(confidence_score, 3),
                'textblob_subjectivity': round(textblob_subjectivity, 3),
                'vader_details': {
                    'pos': round(vader_scores['pos'], 3),
                    'neu': round(vader_scores['neu'], 3),
                    'neg': round(vader_scores['neg'], 3)
                }
            })
            
        return result
    
    def _calculate_confidence(self, textblob_score: float, vader_score: float, subjectivity: float) -> float:
        """
        Calculate confidence score based on agreement between analyzers and subjectivity.
        """
        # Calculate agreement between analyzers (normalized difference)
        score_diff = abs(textblob_score - vader_score)
        agreement = 1 - (score_diff / 2)  # Scale to 0-1
        
        # Consider subjectivity (high subjectivity reduces confidence)
        confidence = (agreement * 0.7) + ((1 - subjectivity) * 0.3)
        return confidence
        
    @staticmethod
    def _get_sentiment_label(score: float) -> str:
        """
        Convert sentiment score to human-readable label with fine-grained categories.
        """
        if score > 0.5:
            return 'Very Positive'
        elif score > 0.1:
            return 'Positive'
        elif score < -0.5:
            return 'Very Negative'
        elif score < -0.1:
            return 'Negative'
        return 'Neutral'

class SentimentAnalysisSystem:
    def __init__(self, bearer_token: str):
        """Initialize the complete sentiment analysis system."""
        self.twitter_client = TwitterClient(bearer_token)
        self.sentiment_analyzer = SentimentAnalyzer()
        
    def analyze_topic(self, query: str, max_results: int = 100) -> Dict:
        """
        Analyze sentiments for a given topic on Twitter.
        Returns comprehensive analysis results.
        """
        tweets = self.twitter_client.fetch_tweets(query, max_results)
        results = []
        sentiment_counts = {'Very Positive': 0, 'Positive': 0, 'Neutral': 0, 'Negative': 0, 'Very Negative': 0}
        
        for tweet in tweets:
            analysis = self.sentiment_analyzer.analyze_text(tweet['text'])
            results.append({**tweet, **analysis})
            sentiment_counts[analysis['sentiment']] += 1
            
        return {
            'query': query,
            'total_tweets': len(tweets),
            'sentiment_distribution': sentiment_counts,
            'detailed_results': results
        }