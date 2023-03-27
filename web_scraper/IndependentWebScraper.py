"""IndependentWebScraper is for downloading news articles
from The Independent to a corpus based on the sitemap provided at:
https://www.independent.co.uk/robots.txt

@author: Jack Smith
"""
from bs4 import BeautifulSoup

from web_scraper.WebScraper import WebScraper
from web_scraper.errors import *


class IndependentWebScraper(WebScraper):
    """Creates objects that can scrape relevant text
    from 'The Independent' articles and save to a predefined location"""

    def isolate_text(self, soup: BeautifulSoup) -> [str]:
        """Takes the HTML of an article from 'The Independent' as a BeautifulSoup
        object and extracts the relevant information: title, headings and paragraphs.
        This is returned as a list of Strings, starting with the title and
        each following entry being a heading or paragraph
        :param soup: A BeautifulSoup object containing the HTML of an Independent article
        :returns: A list containing each line of relevant text in the article"""
        article = soup.find("article")  # isolate article

        try:
            text = [article.h1.get_text().strip()]  # store article header as 1st line
        except AttributeError:  # article has no header - not an article (maybe a live feed)
            raise HeaderNotFound
        else:
            if article.h1.get_text().startswith("This page doesn’t exist"):
                raise PageNotFound

        text.extend([
            p.get_text().strip()
            for p in article.find(id="main").findAll("p")  # get all main paragraphs
            if len(p.get_text().strip()) > 0  # ensure they are not blank (sometimes found where pic are located)
        ])

        return text

    def filter_articles(self, sitemap_data):
        """Removes any unwanted articles from the list as
        to not retrieve them. Unwanted articles are those
        on the subject of sports (found in /sports and /f1)
        or video articles with little text data
        :param sitemap_data: The original list containing all urls retrieved from the Om sitemap"""
        duplicate_data = sitemap_data.copy()
        for article in duplicate_data:  # filter sports and video articles
            if article.loc.get_text().startswith("https://www.independent.co.uk/sport") \
                    or article.loc.get_text().startswith("https://www.independent.co.uk/f1")\
                    or article.loc.get_text().startswith("https://www.independent.co.uk/tv"):
                # if the article is a sports article
                sitemap_data.remove(article)

    def page_is_irrelevant(self, page_text: [str]) -> bool:
        pass


if __name__ == '__main__':
    independent_web_scraper = IndependentWebScraper("https://www.independent.co.uk/sitemaps/googlenews",
                                                    "/Volumes/24265241/News Corpus/Independent Corpus")
    independent_web_scraper.scrape()
