import React from "react";

const clientId = process.env.REACT_APP_SPOTIFY_API_KEY;
const redirectUri = process.env.REACT_APP_SPOTIFY_REDIRECT_URL;

export default function log() {
  console.log("Default function from SpotifyOath");
}

export async function authorize() {
    let codeVerifier = generateRandomString(128);
    localStorage.setItem('code_verifier', codeVerifier);


    generateCodeChallenge(codeVerifier).then(codeChallenge => {

        let state = generateRandomString(16);
        let scope = 'user-read-private user-read-email user-read-playback-state user-modify-playback-state user-read-recently-played user-top-read user-read-private user-read-email user-top-read user-library-read ';

        let args = new URLSearchParams({
            response_type: 'code',
            client_id: clientId,
            scope: scope,
            redirect_uri: redirectUri,
            state: state,
            code_challenge_method: 'S256',
            code_challenge: codeChallenge
        });

        window.location = 'https://accounts.spotify.com/authorize?' + args;
    });
}

export async function getResponse() {

  const urlParams = new URLSearchParams(window.location.search);
  let code = urlParams.get('code');

  let codeVerifier = localStorage.getItem('code_verifier');

  let body = new URLSearchParams({
    grant_type: 'authorization_code',
    code: code,
    redirect_uri: redirectUri,
    client_id: clientId,
    code_verifier: codeVerifier
  });

  // Get the profile and if it is not a status 200 then get a new access token
  //const test_resp = await getProfile();
  //if(test_resp.status == 200)
  // return;

  const response = fetch('https://accounts.spotify.com/api/token', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            body: body
            })
            .then(response => {
                if (!response.ok) {
                throw new Error('HTTP status ' + response.status);
                }
                window.location = 'localhost:3000';
                return response.json();
            })
            .then(data => {
                localStorage.setItem('access_token', data.access_token);
            })
            .catch(error => {
                console.error('Error:', error);
        });
}

export async function getTopSongs(time_range, song_limit) {
  let accessToken = localStorage.getItem('access_token');

  let body = new URLSearchParams({
    time_range: time_range === undefined ? 'medium_term' : time_range,
    limit: song_limit === undefined ? '15' : song_limit,
    offset: '0'
  });

  const response = await fetch(`https://api.spotify.com/v1/me/top/tracks?` + body, {
      method: 'GET',
      headers: {
        Authorization: 'Bearer ' + accessToken
      },
  });

  const data = await response.json();
  console.log(data);

  // Get all of the song names
  let names = data.items.map(n => n.name + " | " + n.external_ids.isrc);
  console.log(names)

  return data;

}

export async function getProfile() {
    let accessToken = localStorage.getItem('access_token');

    const response = await fetch('https://api.spotify.com/v1/me', {
      headers: {
        Authorization: 'Bearer ' + accessToken
      }
    });
  
    const data = await response.json();

    // If there was an error getting the profile, authorize again with the code. 
    //if(data.error !== undefined) {
    //    const resp = await getResponse();
    //    return;
    //}


    return response;
}

/**
 * Generates a challenge based off of the verifier (random string or the IV)
 * @param {*} codeVerifier 
 * @returns 
 */
async function generateCodeChallenge(codeVerifier) {
  function base64encode(string) {
    return btoa(String.fromCharCode.apply(null, new Uint8Array(string)))
      .replace(/\+/g, '-')
      .replace(/\//g, '_')
      .replace(/=+$/, '');
  }

  const encoder = new TextEncoder();
  const data = encoder.encode(codeVerifier);
  const digest = await window.crypto.subtle.digest('SHA-256', data);

  return base64encode(digest);
}


/**
 * Constructs an IV based off of the length
 * @param {} length 
 * @returns 
 */
function generateRandomString(length) {
    let text = '';
    let possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  
    for (let i = 0; i < length; i++) {
      text += possible.charAt(Math.floor(Math.random() * possible.length));
    }
    return text;
}

export async function getTopSongsData(time_range, song_limit) {

  const data = await getTopSongs(time_range, song_limit);

  const list = []

  data.items.forEach(item => {
    // Construct the track object.
    var obj = {
      name: "",
      artists: "",
      isrc: "",
    };

    // Get the data for the track
    let name = "";
    if(item.name != undefined)
      obj.name = item.name;

    if(item.artists != undefined)
      item.artists.forEach(p => obj.artists += p.name + " ")
    
    if(item.external_ids.isrc != undefined)
      obj.isrc = item.external_ids.isrc;

    list.push(obj); 
  });

  // Print out the JSON data
  const json = JSON.stringify(list);
  
  return json;
}

/**
 * Gets the data for a single song based on its song name
 *  1st. Fetch the search function from the spotify endpoint
 *  2nd. Get the song that is the most similar to the search params
 *  3rd. Format the song for the server endpoint
 *  4th. Add the song to a list
 *  5th. Return the list.
 */
export async function getSingleSongData(song_name) {

  let accessToken = localStorage.getItem('access_token');

  let body = new URLSearchParams({
    q: (!song_name) ? "Skyline" : song_name.replaceAll('/\s/g', '+'),
    offset: '0',
    market: 'ES',
    type: 'track',
    limit: '1',
    include_external: 'audio',
  });

  const response = await fetch(`https://api.spotify.com/v1/search?` + body, {
      method: 'GET',
      headers: {
        Authorization: 'Bearer ' + accessToken
      },
  });

  const data = await response.json();

  // Create the object
  var obj = {
    name: "",
    artists: "",
    isrc: "",
  };

  // Get the data for the track
  if(data.tracks.items[0].name != undefined)
    obj.name = data.tracks.items[0].name;

  if(data.tracks.items[0].artists != undefined)
    data.tracks.items[0].artists.forEach(p => obj.artists += p.name + " ")
  
  if(data.tracks.items[0].external_ids.isrc != undefined)
    obj.isrc = data.tracks.items[0].external_ids.isrc;

  const list = [];
  list.push(obj);

  // Stringify the list
  const json = JSON.stringify(list);
  return json;

}

/**
 * Gets the information for a single song based on its song name
 *  1st. Fetch the search function from the spotify endpoint
 *  2nd. Get the song that is the most similar to the search params
 *  3rd. Format the song to the (this.song) class for display.
 *  4th. Add the song to a list
 *  5th. Return the list.
 */
export async function getSingleSongInfo(song_name) {

  let accessToken = localStorage.getItem('access_token');

  let body = new URLSearchParams({
    q: (!song_name) ? "Skyline" : song_name.replaceAll('/\s/g', '+'),
    offset: '0',
    market: 'ES',
    type: 'track',
    limit: '1',
    include_external: 'audio',
  });

  const response = await fetch(`https://api.spotify.com/v1/search?` + body, {
      method: 'GET',
      headers: {
        Authorization: 'Bearer ' + accessToken
      },
  });

  const data = await response.json();

  let name = "";
  if(data.tracks.items[0].name != undefined)
    name = data.tracks.items[0].name;

  let img_url = "";
  if(data.tracks.items[0].album.images[0].url != undefined)
    img_url = data.tracks.items[0].album.images[0].url;

  let artist = "| ";
  if(data.tracks.items[0].artists != undefined)
    data.tracks.items[0].artists.forEach(p => artist += p.name + " ")

  let url = "";
  if(data.tracks.items[0].preview_url != undefined)
    url = data.tracks.items[0].preview_url;
  
  var list = [];
  list.push(new song(name, img_url, artist, url));
  return list;

}

/**
 * Gets the top songs
 * @returns a pair of <song_name, image_url>
 */
export async function getTopSongsInfo(time_range, song_limit) {

  const data = await getTopSongs(time_range, song_limit);
  
  const song_objs = [];
  // Get all of the song names
  data.items.forEach(item => {

    let name = "";
    if(item.name != undefined)
      name = item.name;

    let img_url = "";
    if(item.album.images[0].url != undefined)
      img_url = item.album.images[0].url;

    let artist = "| ";
    if(item.artists != undefined)
      item.artists.forEach(p => artist += p.name + " ")

    let url = "";
    if(item.preview_url != undefined)
      url = item.preview_url;
  
    let add = true;
    if(song_objs.forEach(song => {
      if(song.name === name)
        add = false;
    }));
      
    if(add)
      song_objs.push(new song(name, img_url, artist, url));
  });

  return song_objs;
}
  
// Constructs a song from a song, image pair.
function song(name, image, artist, url) {
  this.name = name;
  this.image = image;
  this.artist = artist;
  this.url = url;
}