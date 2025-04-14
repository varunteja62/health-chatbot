import nltk

def download_nltk_data():
    """Download required NLTK data"""
    print("Downloading NLTK data...")
    
    # List of required NLTK data
    required_data = [
        'punkt',
        'wordnet',
        'stopwords',
        'averaged_perceptron_tagger'
    ]
    
    # Download each dataset
    for data in required_data:
        print(f"Downloading {data}...")
        nltk.download(data)
    
    print("NLTK data download complete!")

if __name__ == "__main__":
    download_nltk_data() 