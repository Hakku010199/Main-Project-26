from src.inference.url_checker import check_url_for_propaganda

if __name__ == "__main__":
    url = input("Enter news URL: ")
    result = check_url_for_propaganda(url)
    print("\nFinal Result:")
    print(result)
