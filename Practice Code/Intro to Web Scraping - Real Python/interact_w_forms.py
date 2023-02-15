# --- Mechanical Soup ---
import mechanicalsoup

# create a browser instance and access its html
browser = mechanicalsoup.Browser()
url = "http://olympus.realpython.org/login"
page = browser.get(url)
html = page.soup

# fill in the form
form = html.select("form")[0]
form.select("input")[0]["value"] = "zeus"
form.select("input")[1]["value"] = "ThunderDude"

# submit the form
next_page = browser.submit(form, page.url)
print(next_page.url)

# access links on new page
links = next_page.soup.select("a")
for link in links:
    link_name = link.text
    link_add = link["href"]
    print(f"{link_name}: {link_add}")

# --- Practice Exercise ---

url = "http://olympus.realpython.org/login"
page = browser.get(url)
html = page.soup

form = html.select("form")[0]
form.select("input")[0]["value"] = "zeus"
form.select("input")[1]["value"] = "ThunderDude"

next_page = browser.submit(form, url)
print(next_page.soup.title)
