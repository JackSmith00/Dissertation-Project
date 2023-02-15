from urllib.request import urlopen

# --- Read page html ---
url = "http://olympus.realpython.org/profiles/aphrodite"
page = urlopen(url)

html_bytes = page.read()
html = html_bytes.decode("utf-8")

# Get the page title
title_index = html.find("<title>")
start_index = title_index + len("<title>")
end_index = html.find("</title>")
title = html[start_index:end_index]
# print(title)
# NOT RELIABLE - html formatting may not be consistent

# --- Using Regular Expressions ---
import re

# read in new page
url = "http://olympus.realpython.org/profiles/dionysus"
page = urlopen(url)
html = page.read().decode("utf-8")

# extract title based on regular expression
re_pattern = "<title.*?>.*?</title.*?>"
match_results = re.search(re_pattern, html, re.IGNORECASE)
title = match_results.group()
title = re.sub("<.*?>", "", title)  # remove the <title> tags
# print(title)

# --- Practice Exercise ---
url = "http://olympus.realpython.org/profiles/dionysus"
html = urlopen(url).read().decode("utf-8")

start_index = html.find("Name:") + len("Name:")
end_index = html.find("<", start_index)
name = html[start_index:end_index].strip()

start_index = html.find("Favorite Color:") + len("Favorite Color:")
end_index = html.find("<", start_index)
fav_color = html[start_index:end_index].strip()

print(f"Name: {name}\nFav Colour: {fav_color}")
