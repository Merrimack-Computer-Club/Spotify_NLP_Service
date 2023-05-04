from imports import *
musixmatch_api_key = "ce0064c0705a30e2b136cfa78dd75eea"

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
    # Request the session
    s = requests.Session()

    track_url = get_musixmatch_share_url(song)
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36', "Upgrade-Insecure-Requests": "1","DNT": "1","Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8","Accept-Language": "en-US,en;q=0.5","Accept-Encoding": "gzip, deflate"}

    # Get the page from the track_url 
    print(track_url)
    s.headers.update(headers)
    s.proxies.update({"http":"104.211.29.96:80", "http":"188.226.188.71:3128"}) # https://free-proxy-list.net/
    page = s.get(track_url)

    # Print the status code
    print(f'Status Code: {page.status_code}')

    # Get the Lyrics and create the Beautiful Soup obj from import.
    soup = BeautifulSoup(page.content, "html.parser")
    lyrics_set = soup.findAll("span", class_="lyrics__content__ok")

    # Construct the lyrics from the set of lyrics.
    # Assign the lyrics to this song.
    for string in lyrics_set:
        string = string.text.strip()
        song.lyrics = [ret for ret in string.split('\n') if ret]

    #print(song.lyrics)

'''Gets share url for a song based on the musixmatch API'''
def get_musixmatch_share_url(song):
    musixmatch_song_content = requests.get(f'https://api.musixmatch.com/ws/1.1/track.get?apikey={musixmatch_api_key}&track_isrc={song.isrc}', headers={'Accept': 'application/json'}).content  
    
    msc_json = json.loads(musixmatch_song_content.decode('utf-8'))

    track_share_url = msc_json['message']['body']['track']['track_share_url']

    return track_share_url

