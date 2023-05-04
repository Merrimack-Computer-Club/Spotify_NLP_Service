from imports import *

class Song:
    
    def __init__(self, name, artists, isrc):
        self.name = name
        self.artists = artists
        self.isrc = isrc
        self.lyrics = []

    def __str__(self):
        return self.name + " " + self.artists + " " + self.isrc + " " + self.lyrics
    
    def toJson(self):
        return {'name': self.name, 'artists': self.artists, 'isrc': self.isrc, "lyrics": self.lyrics}


'''Constructs a list of the top songs using the 'Song' class.'''
def read_top_songs(data):
    ret = []
    for val in data:
        song = Song(val['name'], val['artists'], val['isrc'])
        ret.append(song)

    return ret

'''
Returns the lyrics to a song based on its isrc
Input is a 'Song' class object.
'''
def scrape_song(song):

    # Load in the config file.
    config = toml.load('config.toml')

    # Determine if we should web-scrape the full lyrics or use the Musixmatch lyrics API.
    if(not config['server']['use_full_lyrics']):
        get_lyrics_thirty(song, config['server']['musix_match_api_key'])
    else:
        # Request the session
        s = requests.Session()

        track_url = get_musixmatch_share_url(song, config['server']['musix_match_api_key'])

        # Assure that the track was found
        if(track_url == None):
            return None

        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36', "Upgrade-Insecure-Requests": "1","DNT": "1","Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8","Accept-Language": "en-US,en;q=0.5","Accept-Encoding": "gzip, deflate"}

        # Get the page from the track_url 
        print(track_url)
        s.headers.update(headers)
        s.proxies.update({"http":config['server']['http_proxy'], "https":config['server']['https_proxy']}) # https://free-proxy-list.net/
        page = s.get(track_url)

        if(page.status_code != 200):
            return None

        # Print the status code
        print(f'Status Code: {page.status_code}')

        # Get the `Lyrics` and create the Beautiful Soup obj from import.
        soup = BeautifulSoup(page.content, "html.parser")
        lyrics_set = soup.findAll("span", class_="lyrics__content__ok")

        # Construct the lyrics from the set of lyrics.
        # Assign the lyrics to this song.
        for string in lyrics_set:
            var = string.text.strip()
            var = var.split('\n')
            song.lyrics = [ret for ret in var if ret]

        # If empty print out to console that the IP's may be blocked
        if not song.lyrics:
            print("[server] Your Proxies' IPs are blocked, please enter new proxies in the config.toml to refresh or set 'use_full_lyrics' in config.toml to false.\n\tOr your song's had no lyrics / no MusixMatch URL found.")

'''Gets the thirty percent of lyrics incase IP's are blocked'''
def get_lyrics_thirty(song, musixmatch_api_key):
    # Get the Request
    request  = requests.get(f'https://api.musixmatch.com/ws/1.1/track.lyrics.get?apikey={musixmatch_api_key}&track_isrc={song.isrc}', headers={'Accept': 'application/json'})
    
    # Print out the status code
    print(f'Lyrics API Status: {request.status_code}')

    # Check for status code 200
    if(request.status_code != 200):
        return None

    # Construct the content
    musixmatch_song_content = request.content

    msc_json = json.loads(musixmatch_song_content.decode('utf-8'))

    # If none is found return.
    if(msc_json['message']['body']['lyrics'] is None):
        return None

    # Assign the 30% of song lyrics incase the proxies break mid-session
    lyrics = msc_json['message']['body']['lyrics']['lyrics_body'].split('\n')
    song.lyrics = [ret for ret in lyrics if ret and '******* This Lyrics is NOT for Commercial use *******' not in ret] 


'''Gets share url for a song based on the musixmatch API'''
def get_musixmatch_share_url(song, musixmatch_api_key):
    # Get the Request
    request  = requests.get(f'https://api.musixmatch.com/ws/1.1/track.get?apikey={musixmatch_api_key}&track_isrc={song.isrc}', headers={'Accept': 'application/json'})
    
    # Print out the status code
    print(f'Share URL Status: {request.status_code}')

    # Check for status code 200
    if(request.status_code != 200):
        return None

    # Construct the content
    musixmatch_song_content = request.content

    msc_json = json.loads(musixmatch_song_content.decode('utf-8'))

    track_share_url = msc_json['message']['body']['track']['track_share_url']

    return track_share_url

