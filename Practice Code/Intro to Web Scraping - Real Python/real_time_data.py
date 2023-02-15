import mechanicalsoup
import time

browser = mechanicalsoup.Browser()

for i in range(4):
    page = browser.get("http://olympus.realpython.org/dice")
    tag = page.soup.select("#result")[0]
    result = tag.text
    print(f"Roll #{i+1}: {result}")
    time.sleep(5)
else:
    print("----------")
