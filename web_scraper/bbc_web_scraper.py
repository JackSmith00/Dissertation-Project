import requests
from bs4 import BeautifulSoup


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
    text = [article.h1.get_text().strip()]  # store article header as 1st line

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

    return text  # return all article text


def save_to_corpus(lines: [str], save_location: str):
    """Saves given lines of text to a specified location
    :param lines: the text to write to the file
    :param save_location: the path to which to save the file"""
    with open(save_location, "w") as f:  # will automatically close file
        for line in lines:
            # write each line on a new line in the file
            f.write(line + "\n")


def retrieve_sitemap_data(sitemap_url: str):
    pass
    page = requests.get(sitemap_url)
    sitemap_xml = BeautifulSoup(page.content, "xml.parser")


if __name__ == '__main__':
    pass
