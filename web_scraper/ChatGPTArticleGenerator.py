"""This code is for automatically requesting and downloading
unbiased news articles generated by ChatGPT to a corpus to
serve as a baseline for comparison of how biased other media
outlets are when classified using a supervised approach.

@author: Jack Smith
"""
import openai
import time
from tqdm import tqdm
from datetime import date


def generate_article() -> str:
    """
    Makes a request to the OpenAI API and asks ChatGPT to
    generate an unbiased news article. The output of this
    is returned in a string.
    :return: A string response from ChatGPT containing the
    generated unbiased news article
    """
    # access saved API keys
    with open("/Volumes/24265241/openaiKeys.txt") as f:
        openai_organisation_key = f.readline().strip()
        openai_secret_key = f.readline().strip()

    """
    The below code that accesses the OpenAI was researched at the following resource:
    
    OpenAI, 2023. OpenAI API [online].
    Available from: https://platform.openai.com/docs/api-reference/chat/create
    [Accessed 19 April 2023].
    """

    # set API keys
    openai.organization = openai_organisation_key
    openai.api_key = openai_secret_key

    # send request to ChatGPT to generate an unbiased news article
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user",
                   "content": "Generate an objective news article with a heading that contains no bias"
                   }]
    )

    # return only the relevant response data (the generated article)
    return completion.choices[0].message.content


if __name__ == '__main__':

    try:
        for i in tqdm(range(996, 1000)):  # loop process to generate 500 articles
            request_made = time.time()  # track time so that requests are not made too frequently
            article = generate_article()  # request an article generation
            lines = article.split("\n")  # split the article by lines

            # remove common labeling added by ChatGPT that are surplus to requirements
            if lines[0].startswith("Title:"):
                lines[0] = lines[0].replace("Title: ", "")
            if lines[0].startswith("Headline:"):
                lines[0] = lines[0].replace("Headline: ", "")
            if lines[0].startswith('"') and lines[0].endswith('"'):
                lines[0] = lines[0][1:-1]
            if lines[0].startswith("Possible objective news article:"):
                lines.pop(0)
            if lines[0].startswith("Possible news article:"):
                lines.pop(0)

            # save generated article to corpus
            with open(f"/Volumes/24265241/News Corpus/ChatGPT Corpus/article{i}.txt", "w") as f:
                for line in lines:
                    if line != "":  # ignore blank lines - matches formatting of other corpora
                        f.write(line + "\n")

            while time.time() - request_made < 10:
                # wait at least 10 secs from previous request as to exceed limit
                pass

    finally:  # ensure timestamp is created even if an error occurs
        with open("/Volumes/24265241/News Corpus/ChatGPT Corpus/timestamp.md", "w") as f:
            f.write(f"Generated on: {str(date.today())}")
