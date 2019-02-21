import sys
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
import spotipy.util as util
import pprint
#import plotly.plotly as py
#import plotly.graph_objs as go
#import plotly
import matplotlib

# plotly creds
#plotly.tools.set_credentials_file(username='kjsnow11', api_key='CdXOsPqwq2G3aMCiHpjJ')

# Spotify creds
client_id = '2d0cc7a4ed9d44a69c9ad358b216dd7e'
client_secret = 'bc85301ff7114dca9f2a195804b16ddc'

# Read in credentials
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

if len(sys.argv) > 1:
    search_str = sys.argv[1]
else:
    search_str = 'Underachievers'

# Returns dict of tracks dict
result = sp.search(search_str, limit=20)

# Print all keys in dict
for x in result:
    print(x)


# Goals:
# Search Song
# Features?
# Instruments?
# Tempo/Key...