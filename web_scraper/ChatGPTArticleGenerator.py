import openai
import time
from tqdm import tqdm
from datetime import date


def generate_article():
    with open("/Volumes/24265241/openaiKeys.txt") as f:
        openai_organisation_key = f.readline().strip()
        openai_secret_key = f.readline().strip()

    openai.organization = openai_organisation_key
    openai.api_key = openai_secret_key

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Generate an objective news article with a heading that contains no bias"}]
    )

    return completion.choices[0].message.content


if __name__ == '__main__':

    try:
        for i in tqdm(range(64)):
            request_made = time.time()
            article = generate_article()
            lines = article.split("\n")

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

            with open(f"/Volumes/24265241/News Corpus/ChatGPT Corpus/article{i + 136}.txt", "w") as f:
                for line in lines:
                    if line != "":
                        f.write(line + "\n")

            while time.time() - request_made < 10:
                # wait at least 10 secs from previous request as to exceed limit
                pass

    finally:
        with open("/Volumes/24265241/News Corpus/ChatGPT Corpus/timestamp.md", "w") as f:
            f.write(f"Generated on: {str(date.today())}")
