"""WebScraper holds a class and functions necessary
for saving webpage data to a corpus.

@author: Jack Smith
"""
import requests
from bs4 import Tag, BeautifulSoup
from web_scraper.errors import *
from abc import abstractmethod


def retrieve_sitemap_data(sitemap_url: str) -> [Tag]:
    """Recursively searches a sitemap to retrieve
    all urls from all sublevels of the sitemap
    :param sitemap_url: The url location of the
    sitemap to retrieve urls from"""
    page = requests.get(sitemap_url)
    sitemap_xml = BeautifulSoup(page.content, "lxml-xml")
    if sitemap_xml.find("sitemapindex") is not None:  # sitemap to other sitemaps
        urls = []  # master list for urls
        for sitemap in sitemap_xml.find_all("sitemap"):
            # for each sub-sitemap, repeat this search and add urls to master list
            urls.extend(retrieve_sitemap_data(sitemap.loc.get_text()))
        return urls  # return master list
    else:  # base level sitemap actual data
        return sitemap_xml.find_all("url")  # return urls from sitemap


def extract_page(url: str) -> BeautifulSoup:
    """Retrieves the HTML of a web page and
    returns it as a BeautifulSoup object

    :param url: The url address of the webpage
    :returns: A BeautifulSoup object representing the website HTML
    """
    page = requests.get(url)  # get html content
    return BeautifulSoup(page.content, "html.parser")  # return it as a BeautifulSoup object


def save(lines: [str], save_location: str):
    """Saves given lines of text to a specified location
    :param lines: the text to write to the file
    :param save_location: the path to which to save the file"""
    with open(save_location, "w") as f:  # will automatically close file
        for line in lines:
            # write each line on a new line in the file
            f.write(line + "\n")


class WebScraper:
    """Object that can scrape all pages from a provided
    sitemap. It should be extended to apply filters and
    formatting for specific websites"""

    def __init__(self, sitemap, save_location):
        self.sitemap = sitemap
        self.save_location = save_location

    @abstractmethod
    def filter_articles(self, sitemap_data):
        """Can be overwritten to apply a filter to any
        unwanted pages from the sitemap as to not retrieve them.
        :param sitemap_data: The original list containing all urls retrieved from the sitemap"""
        pass

    def page_is_irrelevant(self, page_text: [str]) -> bool:
        """Can be extended to check the contents of page text
        and determine if the page is relevant or not
        :returns: True when the page should be ignored"""
        return False

    def isolate_text(self, soup: BeautifulSoup) -> [str]:
        """Takes the HTML of a webpage as a BeautifulSoup object and
        extracts all text from the page. This is returned as a list of Strings.
        :param soup: A BeautifulSoup object containing the HTML of a webpage
        :returns: A list containing each line of text from the page"""
        return list(soup.strings)

    def scrape(self):
        """Used to recursively scrape webpages from the set sitemap
        and save to a predefined location. Will also filter articles
        and isolate relevant text where applicable"""
        # Step 1: Retrieve all page urls from the sitemap
        sitemap_data = retrieve_sitemap_data(self.sitemap)
        self.filter_articles(sitemap_data)

        # Step 2: Extract pages
        for page_data in sitemap_data:
            # get page data
            page_url = page_data.loc.get_text()
            page_html = extract_page(page_url)

            # Step 3: Filter relevant data
            try:
                page_text = self.isolate_text(page_html)
            except HeaderNotFound:
                print("No header found for this article:")
                print(page_url)
                print("It has been disregarded\n")
            except PageNotFound:
                print("No page found for this article:")
                print(page_url)
                print("It has been disregarded\n")
            else:
                if self.page_is_irrelevant(page_text):
                    continue

                # Step 4: Save page to corpus
                save_name = page_url[str(page_url).rfind("/"):] + ".txt"
                save(page_text, self.save_location + save_name)
