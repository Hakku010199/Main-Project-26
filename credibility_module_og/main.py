from src.inference.url_checker import check_news_url

if __name__ == "__main__":
    url = input("Enter news URL: ")
    result = check_news_url(url)
    print(result)
