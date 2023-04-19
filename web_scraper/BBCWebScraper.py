"""BBCWebScraper is for downloading BBC news articles
to a corpus based on the sitemap provided at:
https://www.bbc.co.uk/robots.txt

@author: Jack Smith
"""
from bs4 import BeautifulSoup
from web_scraper.WebScraper import WebScraper
from web_scraper.errors import *


class BBCWebScraper(WebScraper):
    """Creates objects that can scrape relevant text
    from BBC articles and save to a predefined location"""

    def isolate_text(self, soup: BeautifulSoup) -> [str]:
        """Takes the HTML of a BBC article as a BeautifulSoup object and
        extracts the relevant information: title, headings and paragraphs.
        This is returned as a list of Strings, starting with the title and
        each following entry being a heading or paragraph
        :param soup: A BeautifulSoup object containing the HTML of a BBC article
        :returns: A list containing each line of relevant text in the article"""
        article = soup.find("article")  # isolate article

        try:
            text = [article.h1.get_text().strip()]  # store article header as 1st line
        except AttributeError:  # article has no header - not an article (maybe a live feed)
            raise HeaderNotFound
        else:
            if article.h1.get_text().startswith('\n404\n'):
                raise PageNotFound  # page doesn't exist - no article

        text_blocks = [  # retrieve all blocks containing text or headings
            div for div in article.find_all("div")
            if div.attrs.get("data-component") == "text-block" or div.attrs.get("data-component") == "subheadline-block"
        ]
        # this method prevents irrelevant paragraphs being captured, such as related
        # article snippets and video player error warnings

        for div in text_blocks:
            for p in div.find_all(["p", "h2"]):  # seperate each paragraph and heading
                """
                The passing of multiple arguments to find_all() was researched at the following resource:

                PROJECTPRO, 2022. How to pass attributes in the find functions of beautiful Soup [online].
                Available from: https://www.projectpro.io/recipes/pass-attributes-in-find-and-find-all-functions-of-beautiful-soup
                [Accessed 16 February 2023]

                The code was not copied verbatim; it was simply used as a guide how to better use the find_all() function.
                """
                if p.get_text() != "":  # ignore blank lines
                    if p.find("i") is not None:
                        # ignore irrelevant information (content warnings and social media links)
                        continue
                    text.append(p.get_text().strip())  # add current line to the end of the current list

        return text  # return all article text

    def filter_articles(self, sitemap_data):
        """Removes any unwanted articles from the list as
        to not retrieve them. Unwanted articles are those
        not in English and not from BBC News (CBBC Newsround,
        BBC Sport, BBC Cymru Fyw, etc.)
        :param sitemap_data: The original list containing all urls retrieved from the BBC sitemap"""
        duplicate_data = sitemap_data.copy()
        for article in duplicate_data:  # loop each article in the list
            if article.language.get_text() != "en" or article.find("name").get_text() != "BBC News":
                # if it is not in English or not from BBC news, remove it from the list
                sitemap_data.remove(article)


if __name__ == '__main__':

    bbc_web_scraper = BBCWebScraper("https://www.bbc.co.uk/sitemaps/https-index-uk-news.xml",
                                    "/Volumes/24265241/News Corpus/BBC Corpus")
    bbc_web_scraper.scrape()
