# --- Beautiful Soup ---
from bs4 import BeautifulSoup
from urllib.request import urlopen

url = "http://olympus.realpython.org/profiles"
html = urlopen(url).read().decode("utf-8")
soup = BeautifulSoup(html, "html.parser")
print(soup.prettify())

# for link in soup.find_all("a"):
#     print(url[:url.find("/profiles")] + link["href"])