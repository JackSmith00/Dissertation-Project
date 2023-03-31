"""DailyMailWebScraper is for downloading Daily Mail news
articles to a corpus based on the sitemap provided at:
https://www.dailymail.co.uk/robots.txt

@author: Jack Smith
"""
from bs4 import BeautifulSoup
from web_scraper.WebScraper import WebScraper
from web_scraper.errors import *


class DailyMailWebScraper(WebScraper):
    """Creates objects that can scrape relevant text
    from Daily Mail articles and save them to a
    predefined location"""

    def isolate_text(self, soup: BeautifulSoup) -> [str]:
        """Takes the HTML of an article from the Daily Mail as a BeautifulSoup
        object and extracts the relevant information: title, headings and paragraphs.
        This is returned as a list of Strings, starting with the title and
        each following entry being a heading or paragraph
        :param soup: A BeautifulSoup object containing the HTML of a Daily Mail article
        :returns: A list containing each line of relevant text in the article"""
        article = soup.find(id="content")

        try:
            text = [article.h2.get_text().strip()]
        except AttributeError:
            raise HeaderNotFound
        else:
            if article.h1.get_text().startswith("Let's try again"):
                raise PageNotFound

        text.extend([
            text.get_text().strip()
            for text in article.find(id="js-article-text").findAll(["p", "strong"])
            if text.name == "strong" or (text.has_attr("class") and text["class"][0] == "mol-para-with-font")
        ])

        return text

    def filter_articles(self, sitemap_data):
        """Removes any unwanted articles from the list as
        to not retrieve them. Unwanted articles are those
        not on the subject of news.
        :param sitemap_data: The original list containing all urls retrieved from the sitemap"""
        duplicate_data = sitemap_data.copy()
        for article in duplicate_data:  # filter sports and video articles
            if not article.loc.get_text().startswith("https://www.dailymail.co.uk/news"):
                # if the article is not a news article
                sitemap_data.remove(article)


if __name__ == '__main__':
    daily_mail_web_scraper = DailyMailWebScraper("https://www.dailymail.co.uk/google-news-sitemap.xml",
                                                 "/Volumes/24265241/News Corpus/Daily Mail Corpus")
    daily_mail_web_scraper.scrape()
