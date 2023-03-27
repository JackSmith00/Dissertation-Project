"""WebScraper holds a class and functions necessary
for saving webpage data to a corpus.

@author: Jack Smith
"""
import os.path

import requests
from bs4 import Tag, BeautifulSoup
from web_scraper.errors import *
from abc import abstractmethod
from datetime import date


def retrieve_sitemap_data(sitemap_url: str, limit: int = -1) -> [Tag]:
    """Recursively searches a sitemap to retrieve
    all urls from all sublevels of the sitemap
    :param sitemap_url: The url location of the sitemap to retrieve urls from
    :param limit: A maximum number of urls to be retrieved, defaults
    to -1 to indicate no limit"""
    urls = []  # master list for urls
    page = requests.get(sitemap_url)
    sitemap_xml = BeautifulSoup(page.content, "lxml-xml")
    if sitemap_xml.find("sitemapindex") is not None:  # sitemap to other sitemaps
        for sitemap in sitemap_xml.find_all("sitemap"):
            # for each sub-sitemap, repeat this search and add urls to master list
            urls.extend(retrieve_sitemap_data(sitemap.loc.get_text(), limit - len(urls)))
            if len(urls) >= limit > -1:  # if limit reached, stop here and return list
                return urls[:limit]
    else:  # base level sitemap actual data
        urls.extend(sitemap_xml.find_all("url"))  # urls from current sitemap
        if len(urls) >= limit > -1:  # if limit reached, stop here and return list
            return urls[:limit]

    return urls  # limit not reached, return all that is found


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
        if os.path.exists(save_location + "/timestamp.md"):  # check if corpus has a timestamp
            with open(save_location + "/timestamp.md") as f:  # if so, open it and retrieve dates
                lines = f.readlines()
                self.oldest_page = date.fromisoformat(lines[1][lines[1].find(":") + 1:].strip())
                self.latest_page = date.fromisoformat(lines[2][lines[2].find(":") + 1:].strip())
        else:  # if not, no dates to check
            self.oldest_page = None
            self.latest_page = None

    def update_date_range(self, page_data: BeautifulSoup):
        """Used to update the date range of the corpus
        for storage in the timestamp
        :param page_data: The BeautifulSoup object containing
        all info on a url from which the date of the page can
        be retrieved"""
        # retrieve publish date of page
        publish_date = date.fromisoformat(page_data.find("publication_date").get_text()[:10])

        # check timestamps already exist, if not, set them
        if self.latest_page is None:
            self.latest_page = publish_date
        # otherwise, compare and set
        elif publish_date > self.latest_page:
            self.latest_page = publish_date

        if self.oldest_page is None:
            self.oldest_page = publish_date
        elif publish_date < self.oldest_page:
            self.oldest_page = publish_date

    def timestamp(self):
        to_save = [f"Last collected: {str(date.today())}"]
        if self.oldest_page is not None and self.latest_page is not None:
            to_save.append(f"Retrieved from: {str(self.oldest_page)}"
                           f"\n\t    to: {str(self.latest_page)}")
        save(to_save, self.save_location + "/timestamp.md")

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

    def scrape(self, limit: int = -1):
        """Used to recursively scrape webpages from the set sitemap
        and save to a predefined location. Will also filter articles
        and isolate relevant text where applicable"""
        # Step 1: Retrieve all page urls from the sitemap
        sitemap_data = retrieve_sitemap_data(self.sitemap, -1 if limit == -1 else limit * 2)
        # retrieve double the urls needed as some may be disregarded
        self.filter_articles(sitemap_data)

        count = 0  # to track when limit is reached

        try:  # ensures timestamp is always created, even if unexpected error occurs
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
                else:  # when text is successfully isolated
                    if self.page_is_irrelevant(page_text):
                        continue
                    else:  # page is relevant
                        # Step 4: Save page to corpus
                        save_name = page_url[str(page_url).rfind("/"):] + ".txt"
                        save(page_text, self.save_location + save_name)

                        self.update_date_range(page_data)  # update timestamp
                        count += 1  # increase successful extraction count
                        if count >= limit != -1:
                            break
        finally:
            self.timestamp()  # timestamp the corpus
