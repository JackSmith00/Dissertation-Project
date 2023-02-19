import requests
from errors import *
from bs4 import BeautifulSoup
from bs4.element import Tag


def extract_page(url: str) -> BeautifulSoup:
    """Retrieves the HTML of a web page and
    returns it as a BeautifulSoup object

    :param url: The url address of the webpage
    :returns: A BeautifulSoup object representing the website HTML
    """
    page = requests.get(url)  # get html content
    return BeautifulSoup(page.content, "html.parser")  # return it as a BeautifulSoup object


def isolate_bbc_text(soup: BeautifulSoup) -> [str]:
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
                text.append(p.get_text().strip())  # add current line to the end of the current list
                if p.find("i") is not None:
                    pass
                    # print(p)

    return text  # return all article text


def save_to_corpus(lines: [str], save_location: str):
    """Saves given lines of text to a specified location
    :param lines: the text to write to the file
    :param save_location: the path to which to save the file"""
    with open(save_location, "w") as f:  # will automatically close file
        for line in lines:
            # write each line on a new line in the file
            f.write(line + "\n")


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


def filter_bbc_articles(sitemap_data: [Tag]) -> [Tag]:
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

    return sitemap_data  # return the filtered list


if __name__ == '__main__':

    corpus_path = "/Volumes/24265241/BBC Corpus/"

    # Step 1: Retrieve all relevant article urls from the BBC sitemap
    bbc_sitemap = "https://www.bbc.co.uk/sitemaps/https-index-uk-news.xml"
    bbc_sitemap_data = retrieve_sitemap_data(bbc_sitemap)
    filter_bbc_articles(bbc_sitemap_data)

    # Step 2: Extract articles
    for article_data in bbc_sitemap_data:
        # get page data
        article_url = article_data.loc.get_text()
        article_html = extract_page(article_url)

        try:
            # Step 3: Filter relevant data
            article_text = isolate_bbc_text(article_html)
            if len(article_text) < 5:
                # skip short articles that are likely videos with a few
                # summary lines, rather than actual articles
                continue

            # Step 4: Save article to corpus
            save_name = article_url[str(article_url).rfind("/"):] + ".txt"
            save_to_corpus(article_text, corpus_path + save_name)

        # Handle bad articles
        except HeaderNotFound:
            print("No header found for this article:")
            print(article_url)
            print("It has been disregarded\n")
        except PageNotFound:
            print("No page found for this article:")
            print(article_url)
            print("It has been disregarded\n")
