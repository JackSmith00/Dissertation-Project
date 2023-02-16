from cgitb import text

import requests
from bs4 import BeautifulSoup


def extract_page(url: str) -> BeautifulSoup:
    # get html content
    page = requests.get(url)
    return BeautifulSoup(page.content, "html.parser")


def isolate_bbc_text(soup: BeautifulSoup) -> [str]:
    article = soup.find("article")
    article_text = [article.h1.get_text().strip()]
    text_blocks = [
        div for div in article.find_all("div")
        if div.attrs.get("data-component") == "text-block" or div.attrs.get("data-component") == "subheadline-block"
    ]
    for div in text_blocks:
        for p in div.find_all(["p", "h2"]):  # https://www.projectpro.io/recipes/pass-attributes-in-find-and-find-all-functions-of-beautiful-soup
            if p.get_text() != "":
                article_text.append(p.get_text().strip())

    return article_text


def save_to_corpus(lines: [str], save_location: str):
    with open(save_location, "w") as f:
        for line in lines:
            f.write(line + "\n")


if __name__ == '__main__':
    with open("/Users/Jack/Documents/UNI/Project/Code/Example Soup/world-europe-64626783.html", "r") as file:
        a1 = BeautifulSoup(file.read(), "html.parser")
    with open("/Users/Jack/Documents/UNI/Project/Code/Example Soup/cw5pvgn7356o.html", "r") as file:
        a2 = BeautifulSoup(file.read(), "html.parser")
    with open("/Users/Jack/Documents/UNI/Project/Code/Example Soup/uk-england-12457929.html", 'r') as file:
        a3 = BeautifulSoup(file.read(), "html.parser")

    new_page = extract_page("https://www.bbc.co.uk/news/education-63283289")
    page_text = isolate_bbc_text(new_page)
    save_to_corpus(page_text, "/Users/Jack/Documents/UNI/Project/Code/Example Soup/education-63283289.txt")

    isolate_bbc_text(a1)
    isolate_bbc_text(a2)
    isolate_bbc_text(a3)
