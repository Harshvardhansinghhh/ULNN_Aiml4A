import logging
import asyncio
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer


TELEGRAM_BOT_TOKEN = "8098449383:AAFZ5a5NKJkqM6v5M2q-n-9KwkC1jVpzyJE"
NEWS_API_KEY = "f94b03eb9d6944d49fd3e535c4a78b40"
MAX_ARTICLES = 50
ARTICLES_TO_SHOW = 10
SUMMARY_SENTENCES = 3


logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class NewsProcessor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.summarizer = LsaSummarizer()
    
    def cluster_articles(self, articles):
        try:
            texts = [f"{art['title']} {art['description'] or ''}" for art in articles]
            
            if len(texts) < 3:
                return [articles]
                
            tfidf = self.vectorizer.fit_transform(texts)
            n_clusters = min(10, len(texts) // 2)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(tfidf)
            
            return [[articles[j] for j in range(len(articles)) if clusters[j] == i] 
                   for i in range(n_clusters)]
        except Exception as e:
            logger.error(f"Clustering error: {e}")
            return [articles]
    
    def summarize_text(self, text):
        try:
            if not text or len(text.split()) < 10:
                return "Summary not available for short content"
                
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            return " ".join(str(s) for s in 
                          self.summarizer(parser.document, SUMMARY_SENTENCES))
        except Exception as e:
            logger.warning(f"Summarization failed: {e}")
            return text[:300] + ("..." if len(text) > 300 else "")

news_processor = NewsProcessor()

def get_news(topic):
    try:
        response = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                'q': topic + " AND (India OR Indian)",
                'apiKey': NEWS_API_KEY,
                'language': 'en',
                'pageSize': MAX_ARTICLES,
                'sortBy': 'publishedAt'
            },
            timeout=15
        )
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') != 'ok':
            logger.error(f"API Error: {data.get('message')}")
            return []
            
        articles = data.get('articles', [])
        logger.info(f"Found {len(articles)} articles for '{topic}'")
        return [art for art in articles if art.get('title') and art.get('url')]
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    return []

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📰 Indian News Bot\n\n"
        "Type any topic (e.g. 'cricket', 'AI startups', 'elections')\n"
        "I'll fetch the latest Indian news with summaries!\n\n"
        "Try: technology, business, bollywood",
        parse_mode='Markdown'
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    topic = update.message.text.strip()
    
    if not topic or len(topic) < 2:
        await update.message.reply_text("Please enter a valid topic (min 2 characters)")
        return
    
    logger.info(f"User @{user.username} requested: {topic}")
    
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing"
    )
    
    articles = get_news(topic)
    
    if not articles:
        await update.message.reply_text(
            f"⚠ No Indian news found for '{topic}'\n"
            "Try different keywords like:\n"
            "- 'technology' instead of 'AI'\n"
            "- 'modi' instead of 'politics'"
        )
        return
    
    clustered_articles = news_processor.cluster_articles(articles)
    sent_count = 0
    
    for cluster in clustered_articles:
        if sent_count >= ARTICLES_TO_SHOW:
            break
            
        article = cluster[0]
        message = format_article(article)
        
        try:
            await update.message.reply_text(
                text=message,
                parse_mode='Markdown',
                disable_web_page_preview=False
            )
            sent_count += 1
            await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Failed to send message: {e}")

def format_article(article):
    title = article.get('title', 'No title')
    url = article.get('url', '')
    source = article.get('source', {}).get('name', 'Source')
    content = article.get('content') or article.get('description', '')
    
    summary = news_processor.summarize_text(content)
    
    return (
        f"📰 {title}\n"
        f"{source}\n\n"
        f"🔹 Summary: {summary}\n\n"
        f"📎 [Read more]({url})"
    )

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Update {update} caused error: {context.error}")
    if update.effective_message:
        await update.effective_message.reply_text(
            "❌ An error occurred. Please try again later."
        )

def main():
    test_articles = get_news("test")
    logger.info(f"Startup test - Found {len(test_articles)} articles")
    
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)
    
    logger.info("Bot is running...")
    app.run_polling()


if __name__ == '__main__':
    main()
