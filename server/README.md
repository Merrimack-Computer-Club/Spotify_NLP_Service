# Spotify Emotions Server Management


## Dependency Management
To install all dependencies/imports run cmd in the server directory.
```pip install -r requirements.txt```

After adding new dependencies run cmd in the server directory to regenerate the file
```pipreqs . --force```


## Starting the Server
To start the server run the following cmd in the server directory
```python .\server.py``` 

## Configuring the server
To Configure the server look towards the ```config.toml``` file.
You can currently edit your musixmatch API key, http & https proxies, and
if you want to use webscraping (use_full_lyrics=true) or if you want to use 30% of the lyrics
from the musixmatch free lyrics api (use_full_lyrics=true).
If you have a pytorch model saved that was compiled using model.py you can load it using the 
model_path={path option.